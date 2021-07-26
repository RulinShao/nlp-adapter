import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model, is_model, model_entrypoint
from timm.models import create_model
from timm.models.helpers import load_pretrained, load_checkpoint

from models.layers.weight_init import lecun_normal_
from args import args


_logger = logging.getLogger(__name__)


# Configurations for ViT
def _cfg(path='', url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    # patch models (weights ported from timm impl)
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),

    # patch models (weights ported from official Google JAX impl)
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    # patch models, imagenet21k (weights ported from official Google JAX impl)
    'vit_base_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    # deit models (FB weights)
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    # SAM pretrained vit
    'vit_base_patch16_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz'),
}


class BasicAdapter(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features // 2
        self.down_proj = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.up_proj = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = x + self.up_proj(self.act(self.down_proj(x)))
        return x


class MultiAdapter(nn.Module):
    def __init__(self, num_adapters=1):
        self.num_adapters = num_adapters
        self.weight = [nn.Parameter() for _ in range(self.num_adapters)]

    def forward(self, x, alpha=None):
        assert len(alpha) == self.num_adapters
        new_weight = torch.sum(alpha[i] * self.adapter[i].weight for i in range(len(alpha)))
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_adapter=False, capacity=None):
        super().__init__()

        self.use_adapter = use_adapter
        self.capacity = capacity  # use limited capacity.

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # TODO: add a function to add Adapters to the list according to capacity arg.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.capacity:
            assert self.capacity > 0; "Invalid capacity for adapters."
            self.adapter1 = nn.ModuleList([BasicAdapter(in_features=dim, act_layer=act_layer) for i in range(self.capacity)])
            self.adapter2 = nn.ModuleList([BasicAdapter(in_features=dim, act_layer=act_layer) for i in range(self.capacity)])
        else:
            self.adapter1 = BasicAdapter(in_features=dim, act_layer=act_layer)
            self.adapter2 = BasicAdapter(in_features=dim, act_layer=act_layer)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, alpha=None, soft=False):
        if self.use_adapter:
            if self.capacity:
                x_ = self.drop_path(self.attn(self.norm1(x)))
                if not soft:
                    alpha = torch.argmax(alpha, dim=1)
                    x = x + self.adapter1[alpha[0]](x_)
                    x_ = self.drop_path(self.mlp(self.norm2(x)))
                    x = x + self.adapter2[alpha[1]](x_)
                else:
                    assert int(alpha.size()[1]) == self.capacity;
                    f"Incompatible alpha for limited capacity, expected {self.capacity} got {alpha.size()[1]}"
                    # TODO: add adapters first to save multiplications
                    for i in range(self.capacity):
                        x = x + (alpha[0][i]/self.capacity) * self.adapter1[i](x_)
                    x_ = self.drop_path(self.mlp(self.norm2(x)))
                    for i in range(self.capacity):
                        x = x + (alpha[1][i]/self.capacity) * self.adapter2[i](x_)
            else:
                x = x + self.adapter1(self.drop_path(self.attn(self.norm1(x))))
                x = x + self.adapter2(self.drop_path(self.mlp(self.norm2(x))))
            return x
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', use_adapter=args.train_adapter, capacity=args.capacity, soft_alpha=args.soft_alpha):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.depth = depth
        self.use_adapter = use_adapter
        if capacity is not None and capacity > 0:
            self.capacity = capacity
            self.alpha = nn.Parameter(torch.ones((self.depth, 2, self.capacity)))
            self.soft_alpha = soft_alpha
            self.adapter_count = torch.zeros_like(self.alpha)
        else:
            self.capacity = None
            self.alpha = None

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # TODO: set only the adapter related to task trainable
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                use_adapter=self.use_adapter, capacity=self.capacity)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_list = None

        # Weight init
        assert weight_init in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in weight_init else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if weight_init.startswith('jax'):
            # leave cls token as zeros to match jax impl
            for n, m in self.named_modules():
                _init_vit_weights(m, n, head_bias=head_bias, jax_impl=True)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def choose_head_from_list(self, task_id):
        if task_id >= 0:
            self.head = self.head_list[task_id]
        else:
            self.head = None
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def reset_classifier_list(self, num_classes):
        self.num_classes = num_classes
        self.head = None
        self.head_list = nn.ModuleList(
            [nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity() for _ in range(args.num_tasks)]
        )

    def set_adapter(self, new_head=None):
        # Set adapter and norm layers trainable.
        # Reset the head layer when dimension of new head is given
        self.activate_adapter()
        for name, param in self.named_parameters():
            if 'adapter' in name or 'norm' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        if new_head:
            if self.use_adapter and self.capacity is not None:
                self.reset_classifier_list(new_head)
                self.head_list.requires_grad = True
            else:
                self.reset_classifier(new_head)

    def update_alpha(self, alpha=None, soft_alpha=False):
        if alpha is None:
            alpha = nn.Parameter(torch.ones((self.depth, 2, self.capacity)))
            soft_alpha = True
        self.alpha.data.copy_(alpha)
        self.soft_alpha = soft_alpha

    def count_alpha(self, new_alpha=None):
        if new_alpha is None:
            new_alpha = self.alpha.data
        self.adapter_count = self.adapter_count + new_alpha.cpu()

    def train_alpha(self, train_alpha=True):
        if train_alpha:
            self.alpha.requires_grad_(True)
        else:
            self.alpha.requires_grad_(False)

    def set_layer(self, layer_num=0, new_head=None):
        self.remove_adapter()

        for name, param in self.named_parameters():
            param.requires_grad = False

        if new_head:
            self.reset_classifier(new_head)

        # layer_num >= 0
        self.head.requires_grad = True
        self.head.weight.requires_grad=True
        self.head.bias.requires_grad = True

        act_layer = []
        if layer_num > 0:
            # layer_num >= 1
            self.norm.requires_grad = True
            self.pre_logits.fc.requires_grad = True
            self.pre_logits.fc.weight.requires_grad = True
            self.pre_logits.fc.bias.requires_grad = True

            # layer_num >= 2
            if layer_num > 1:
                for i in range(layer_num):
                    assert i <= self.depth; "Exceed maximum layers"
                    act_layer.append(str(self.depth-i))
                for name, param in self.named_parameters():
                    if 'adapter' not in name and (len(name.split('.')) > 1 and name.split('.')[1] in act_layer):
                        param.requires_grad = True

    def remove_adapter(self):
        self.use_adapter = False

    def activate_adapter(self):
        self.use_adapter = True

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for i in range(self.depth):
            if self.capacity:
                assert int(self.alpha.size()[0]) == self.depth
                x = self.blocks[i](x, self.alpha[i], soft=self.soft_alpha)
            else:
                x = self.blocks[i](x)
        x = self.norm(x)
        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        # assert self.head is not None
        if self.use_adapter and self.capacity is not None and self.task >= 0:
            x = self.head_list[self.task](x)
            return x
        x = self.head(x)
        return x


def _init_vit_weights(m, n: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    TODO: initialize adapter parameters to near zeros. (Maybe just set smaller std for adapter init)
    """
    if isinstance(m, nn.Linear):
        if n.startswith('head'):
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, head_bias)
        elif n.startswith('pre_logits'):
            lecun_normal_(m.weight)
            nn.init.zeros_(m.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if 'mlp' in n:
                        nn.init.normal_(m.bias, std=1e-6)
                    else:
                        nn.init.zeros_(m.bias)
            else:
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif jax_impl and isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


def create_model(
        model_name,
        pretrained=False,
        num_classes=1000,
        in_chans=3,
        checkpoint_path='',
        use_adapter=False,
        **kwargs):

    model_args = dict(pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if is_model(model_name):
        create_fn = model_entrypoint(model_name)
        model = create_fn(**model_args, **kwargs)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, strict=(not use_adapter))

    return model


@register_model
def vit_small_patch16_224_adapter(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3., use_adapter=args.train_adapter, capacity=args.capacity, soft_alpha=args.soft_alpha, **kwargs)
    model.default_cfg = default_cfgs['vit_small_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, strict=False)
    return model


@register_model
def vit_base_patch16_224_in21k_adapter(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=768, use_adapter=args.train_adapter, capacity=args.capacity, soft_alpha=args.soft_alpha,
                              **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224_in21k']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter,
            strict=False)
    return model


# @register_model
# def vit_base_patch16_sam_224(pretrained=False, **kwargs):
#     """ ViT-Base (ViT-B/16) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548
#     """
#     # model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=768, **kwargs)
#     # model = _create_vision_transformer('vit_base_patch16_sam_224', pretrained=pretrained, **model_kwargs)
#     # return model
#     if pretrained:
#         kwargs.setdefault('qk_scale', 768 ** -0.5)
#     model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12,representation_size=768, use_adapter=args.train_adapter, capacity=args.capacity, soft_alpha=args.soft_alpha, **kwargs)
#     model.default_cfg = default_cfgs['vit_base_patch16_sam_224']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, strict=False)
#     return model


@register_model
def vit_base_patch16_sam_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=768, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_sam_224', pretrained=pretrained, **model_kwargs)
    return model


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    from timm.models.helpers import build_model_with_cfg
    # from timm.models.vision_transformer import checkpoint_filter_fn, resize_pos_embed

    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


@register_model
def deit_tiny_patch16_224_adapter(pretrained=False, **kwargs):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    # model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    # model = _create_vision_transformer('deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    
    if pretrained:
        kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = VisionTransformer(patch_size=16, embed_dim=192, depth=12, num_heads=3, use_adapter=args.train_adapter, capacity=args.capacity, soft_alpha=args.soft_alpha,
                              **kwargs)
    model.default_cfg = default_cfgs['deit_tiny_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter,
            strict=False)
    return model


@register_model
def deit_small_patch16_224_adapter(pretrained=False, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    # model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    # model = _create_vision_transformer('deit_small_patch16_224', pretrained=pretrained, **model_kwargs)

    if pretrained:
        kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, use_adapter=args.train_adapter, capacity=args.capacity, soft_alpha=args.soft_alpha,
                              **kwargs)
    model.default_cfg = default_cfgs['deit_small_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter,
            strict=False)
    return model


@register_model
def deit_base_patch16_224_adapter(pretrained=False, **kwargs):
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    # model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    # model = _create_vision_transformer('deit_base_patch16_224', pretrained=pretrained, **model_kwargs)

    if pretrained:
        kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, use_adapter=args.train_adapter, capacity=args.capacity, soft_alpha=args.soft_alpha,
                              **kwargs)
    model.default_cfg = default_cfgs['deit_base_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter,
            strict=False)
    return model


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load pretrained model
    model = create_model(args.model_name,
                         pretrained=True,
                         num_classes=1000,
                         in_chans=3,)
    model.eval().to(device)
    print(dict(model.named_parameters()).keys())
    print(model.depth)


if __name__ == '__main__':
    main()


