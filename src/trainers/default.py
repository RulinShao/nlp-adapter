import torch
import torch.nn as nn


def init(args):
    pass


def train(model, writer, train_loader, optimizer, criterion, epoch, task_idx, data_loader=None, args=None):
    if args is None:
        from args import args

    model.zero_grad()
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.iter_lim < 0 or len(train_loader) * (epoch - 1) + batch_idx < args.iter_lim:
            data, target = data.to(args.device), target.to(args.device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                num_samples = batch_idx * len(data)
                num_epochs = len(train_loader.dataset)
                percent_complete = 100.0 * batch_idx / len(train_loader)
                print(
                    f"Train Epoch: {epoch} [{num_samples}/{num_epochs} ({percent_complete:.0f}%)]\t"
                    f"Loss: {loss.item():.6f}"
                )

                t = (len(train_loader) * epoch + batch_idx) * args.batch_size
                writer.add_scalar(f"train/task_{task_idx}/loss", loss.item(), t)


def test(model, writer, criterion, test_loader, epoch, task_idx, args=None):
    if args is None:
        from args import args

    model.zero_grad()
    model.eval()
    test_loss = 0
    correct = 0
    logit_entropy = 0.0

    with torch.no_grad():

        for data, target in test_loader:
            if type(data) == list:
                data = data[0]
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            if len(output.shape) == 1:
                output = output.unsqueeze(0)
            logit_entropy += (
                -(output.softmax(dim=1) * output.log_softmax(dim=1))
                .sum(1)
                .mean()
                .item()
            )
            test_loss += criterion(output, target).item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    logit_entropy /= len(test_loader)
    test_acc = float(correct) / len(test_loader.dataset)

    if epoch is None or epoch == -1:
        return test_acc

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({test_acc:.4f})\n")

    writer.add_scalar(f"test/task_{task_idx}/loss", test_loss, epoch)
    writer.add_scalar(f"test/task_{task_idx}/acc", test_acc, epoch)
    writer.add_scalar(f"test/task_{task_idx}/entropy", logit_entropy, epoch)

    return test_acc


def infer(model, task_id, writer, criterion, train_loader, use_soft=False, args=None):
    if args is None:
        from args import args

    if task_id < model.module.capacity:
        print(f"=> Using the {task_id}-th adapters in all layers..")
        alpha = torch.zeros_like(model.module.alpha)
        for d in range(model.module.depth):
            alpha[d][0][task_id] = 1
            alpha[d][1][task_id] = 1
        model.module.update_alpha(alpha, soft_alpha=use_soft)
    else:
        print(f"=> Initializing alpha to size of {(model.module.depth, 2, model.module.capacity)}..")
        model.module.update_alpha() # Initialize the alpha to torch.ones. soft_alpha holds True
        model.module.train_alpha(True) # Set the alpha trainable
        model.eval()
        grad = torch.zeros_like(model.module.alpha.data)
        print(f"=> Infering alpha using whole training data..")
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            loss_alpha = criterion(model(data), target)
            grad += torch.autograd.grad(loss_alpha, model.module.alpha)[0].detach()
        del loss_alpha
        new_alpha = nn.functional.softmin(grad, dim=2)
        if not use_soft:
            max_idx = torch.argmax(new_alpha, dim=2)
            # new_alpha = max_idx
            new_alpha = torch.zeros_like(new_alpha)
            for i in range(new_alpha.size()[0]):
                for j in range(new_alpha.size()[1]):
                    new_alpha[i][j][max_idx[i][j]] = 1
        model.module.update_alpha(new_alpha, soft_alpha=use_soft) # Fix the alpha and pass the soft_alpha
        model.module.train_alpha(False)
        model.train()


def adapt_test(model, writer, criterion, test_loader, task_idx, alpha_list, use_soft=False, args=None):
    if args is None:
        from args import args

    remeber_alpha = True
    if len(alpha_list) > 0:
        if remeber_alpha:
            model.module.update_alpha(alpha_list[task_idx], use_soft)
        else:
            infer(model, writer, criterion, test_loader, use_soft)

    model.zero_grad()
    model.eval()
    test_loss = 0
    correct = 0
    logit_entropy = 0.0

    with torch.no_grad():

        for data, target in test_loader:
            if type(data) == list:
                data = data[0]
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            if len(output.shape) == 1:
                output = output.unsqueeze(0)
            logit_entropy += (
                -(output.softmax(dim=1) * output.log_softmax(dim=1))
                .sum(1)
                .mean()
                .item()
            )
            test_loss += criterion(output, target).item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

        if len(alpha_list) > 0:
            alpha_diff = torch.sum(abs(alpha_list[task_idx] - model.module.alpha))

    test_loss /= len(test_loader)
    logit_entropy /= len(test_loader)
    test_acc = float(correct) / len(test_loader.dataset)

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({test_acc:.4f})\n")
    if len(alpha_list) > 0:
        return test_acc, alpha_diff
    else:
        return test_acc, None
