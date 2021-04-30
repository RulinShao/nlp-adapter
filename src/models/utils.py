import os
import torch
from args import args
import utils

from timm.models import load_checkpoint, create_model
import models.vit


def get_backbone(head_dim=1000//args.num_tasks, no_head=False):
    # Get the model.
    model = create_model(args.model_name,
                         pretrained=args.pretrained,
                         num_classes=1000,
                         in_chans=3, )
    if not no_head:
        if hasattr(model, "module"):
            model.module.set_head(new_head=head_dim)
        else:
            model.set_head(new_head=head_dim)

    # Put the model on the GPU,
    model = utils.set_gpu(model)

    return model


def get_task_model(model, num_tasks_learned, idx, task_length):
    """
    :param num_tasks_learned:
    :param idx:
    :param task_length:
    :return:
    """
    _modify_model(model, idx, task_length)

    if args.resume:
        _resume_from_ckpt(model)

    # Put the model on the GPU,
    model = utils.set_gpu(model)

    for p in model.parameters():
        p.grad = None

    params = []
    if (
            args.train_weight_tasks < 0
            or num_tasks_learned < args.train_weight_tasks
    ):
        for n, p in model.named_parameters():
            # train all weights if train_weight_tasks is -1, or num_tasks_learned < train_weight_tasks
                p.requires_grad = True
                params.append(p)
    else:
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            params.append(p)

    return model, params


def _resume_from_ckpt(model):
    # Optionally resume from a checkpoint.
    # TODO: resume script for loading adapters
    assert args.resume is not None, "No checkpoint found to resume!"
    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(
            args.resume, map_location=f"cuda:{args.multigpu[0]}"
        )
        best_acc1 = checkpoint["best_acc1"]
        pretrained_dict = checkpoint["state_dict"]
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)

        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
    else:
        print(f"=> No checkpoint found at '{args.resume}'")

    return model, best_acc1


def _modify_model(model, idx, task_length):

    # Tell the model which task it is trying to solve -- in Scenario NNs this is ignored.
    model.apply(lambda m: setattr(m, "task", idx))

    # Change classifier head dimension and set corresponding paramenters (e.g. adapter&norm&head) trainable.
    if args.train_adapter:
        # set_adapter() will automatically activate the adapter layers.
        if hasattr(model, "module"):
            model.module.set_adapter(new_head=task_length)
        else:
            model.set_adapter(new_head=task_length)
    elif args.train_head:
        # set_head() will automatically remove the dapter layers.
        if hasattr(model, "module"):
            model.module.set_head(new_head=task_length)
        else:
            model.set_head(new_head=task_length)
    else:
        if hasattr(model, "module"):
            model.module.remove_adapter()
            model.module.reset_classifier(num_classes=task_length)
        else:
            model.remove_adapter()
            model.reset_classifier(num_classes=task_length)

    return model