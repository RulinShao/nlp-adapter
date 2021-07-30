import os
import torch
from args import args

from timm.models import load_checkpoint, create_model


def get_backbone(img_size=224):
    # Get the model.
    model = create_model(args.model_name,
                         pretrained=args.pretrained,
                         num_classes=1000,
                         in_chans=3,
                         img_size=img_size)
    return model


def get_task_model(model, num_tasks_learned, idx):
    """
    :param num_tasks_learned:
    :param idx:
    :param task_length:
    :return:
    """
    # TODO:
    #  1. return best_acc1 when resuming from a ckpt.
    #  2. load test related adapters when resume.

    # modify_model(model, idx, task_length)

    # Tell the model which task it is trying to solve -- in Scenario NNs this is ignored.
    model.apply(lambda m: setattr(m, "task", idx))

    if args.resume:
        _resume_from_ckpt(model)

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
        return model, params
    else:
        if args.train_adapter and args.capacity is not None:
            adapter_params = {}
            for d in range(model.module.depth):
                for i in range(2):
                    for c in range(model.module.capacity):
                        params_prefix = f"blocks.{d}.adapter{i+1}.{c}."
                        params_list = []
                        for n, p in model.named_parameters():
                            if params_prefix in n:
                                params_list.append(p)
                        assert len(params_list) > 0; "Error in getting adapter parameters"
                        adapter_params.update({params_prefix: params_list})

            # model.module.choose_head_from_list(idx)
            # model.module.head.requires_grad_(True)
            for n, p in model.named_parameters():
                if not p.requires_grad or 'adapter' in n:
                    continue
                params.append(p)
            return model, params, adapter_params
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


def modify_model(model, task_length):

    # Change classifier head dimension and set corresponding paramenters (e.g. adapter&norm&head) trainable.
    if args.train_adapter:
        # set_adapter() will automatically activate the adapter layers.
        if hasattr(model, "module"):
            model.module.set_adapter(new_head=task_length)
        else:
            model.set_adapter(new_head=task_length)
    elif args.train_layer >= 0:
        # set_layer() will automatically remove the dapter layers.
        assert args.train_layer >= 0;
        "invalid number of trainable layers"
        if hasattr(model, "module"):
            model.module.set_layer(layer_num=args.train_layer, new_head=task_length)
        else:
            model.set_layer(layer_num=args.train_layer, new_head=task_length)
    else:
        if hasattr(model, "module"):
            model.module.remove_adapter()
            model.module.reset_classifier(num_classes=task_length)
        else:
            model.remove_adapter()
            model.reset_classifier(num_classes=task_length)

    return model


def save_model(model, best_acc1, curr_acc1, run_base_dir, idx):
    if args.save == "full":
        torch.save(
            {
                "epoch": args.epochs,
                "arch": args.model,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "curr_acc1": curr_acc1,
                "args": args,
            },
            run_base_dir / f"task{idx}_full_final.pt",
        )
    elif args.save == "adapter":
        torch.save(
            {
                "epoch": args.epochs,
                "arch": args.model,
                "state_dict": {k: v for k, v in model.state_dict().items()
                               if 'bn' in k or 'adapter' in k or 'head' in k},
                "curr_acc1": curr_acc1,
                "args": args,
            },
            run_base_dir / f"task{idx}_adapter_final.pt",
        )
    elif args.save == "layer":
        assert args.train_layer >= 0;
        "invalid number of trainable layers"
        if args.train_layer > 0:
            act_layer = ''
            # TODO: add depth to args.py and replace 8 with args.depth when other models introduced
            for i in range(args.train_layer):
                act_layer += str(8 - i)
        if args.train_layer == 0:
            torch.save(
                {
                    "epoch": args.epochs,
                    "arch": args.model,
                    "state_dict": {k: v for k, v in model.state_dict().items()
                                   if 'head' in k},
                    "curr_acc1": curr_acc1,
                    "args": args,
                },
                run_base_dir / f"task{idx}_head_final.pt",
            )
        else:
            torch.save(
                {
                    "epoch": args.epochs,
                    "arch": args.model,
                    "state_dict": {k: v for k, v in model.state_dict().items()
                                   if ('head' in k) or ('norm.' in k) or (len(k.split('.'))>1 and k.split('.')[1] in act_layer and 'adapter' not in k)},
                    "curr_acc1": curr_acc1,
                    "args": args,
                },
                run_base_dir / f"task{idx}_head_final.pt",
            )


def count_trainable_parameters(model):
    if not args.train_adapter:
        count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(n)
    else:
        count = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(n)
    print(f"Number of trainable parameters: {count}")
    return count


if __name__ == "__main__":
    import models.avit
    model = get_backbone()
    model = modify_model(model, 1000//args.num_tasks)
    count  = count_trainable_parameters(model)
