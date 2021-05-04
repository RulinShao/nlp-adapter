import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicAdapter(nn.Module):

    def __init__(self, in_dim, h_dim=None, out_dim=None, act_layer=nn.ReLU):
        super(BasicAdapter, self).__init__()
        self.in_chans = in_dim
        if h_dim:
            self.h_dim = h_dim
        else:
            self.h_dim = in_dim // 2
        if out_dim:
            self.out_dim = out_dim
        else:
            self.out_dim = in_dim

        self.res = nn.Identity()
        self.down_proj = nn.Conv2d(self.in_chans, self.h_dim, kernel_size=1, stride=1, bias=False)
        self.act = act_layer()
        self.up_proj = nn.Conv2d(self.h_dim, self.out_dim, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        return self.up_proj(self.act(self.down_proj(x)))+self.res(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_adapter=False, mode="train"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.mode = mode

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.use_adapter = use_adapter
        if self.use_adapter:
            self.adapter = BasicAdapter(self.expansion * planes)

    def forward(self, x_adapter, x_normal=None):
        if self.mode == "train":
            assert len(x_adapter) == 2; "Expected inputs for adapters in [clean, adv] form"
            adapter_out_clean = self.conv2(F.relu(self.bn1(self.conv1(x_adapter[0]))))
            adapter_out_adv = self.conv2(F.relu(self.bn1(self.conv1(x_adapter[1]))))
            normal_out = self.conv2(F.relu(self.bn1(self.conv1(x_normal))))

            # TODO: add adapters before or after the bn2
            adapter_out_clean = self.bn2(self.adapter(adapter_out_clean))
            adapter_out_adv = self.bn2(self.adapter(adapter_out_adv))
            normal_out = self.bn2(normal_out)

            adapter_out_clean = F.relu(adapter_out_clean + self.shortcut(x_adapter[0]))
            adapter_out_adv = F.relu(adapter_out_adv + self.shortcut(x_adapter[1]))
            normal_out = F.relu(normal_out + self.shortcut(x_normal))

            # out = F.relu(self.shortcut(x) + self.bn2(self.adapter(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
            return [adapter_out_clean, adapter_out_adv], normal_out
        else:
            out = F.relu(self.bn1(self.conv1(x_adapter)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x_adapter)
            out = F.relu(out)
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_adapter=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.use_adapter = use_adapter

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_adapter=self.use_adapter))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(use_adapter=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], use_adapter=use_adapter)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())