import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        # ----- backbone 部分 -----
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # 用一个 Sequential 包起来，后面方便保存 / 调用
        self.backbone = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(inplace=True),
            self.layer1,
            self.layer2,
            self.layer3,
        )

        # ----- 分类头 -----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 等价于你原来的 avg_pool2d(out, 8)
        self.fc      = nn.Linear(64 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 提供一个显式的特征提取接口
    def extract_features(self, x):
        x = self.backbone(x)               # (B, 64, 8, 8)
        x = self.avgpool(x)                # (B, 64, 1, 1)
        x = torch.flatten(x, 1)            # (B, 64)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.fc(x)
        return x


def ResNet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)

#net = ResNet20().to(device)

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """
    经典的 LeNet-5 架构，针对 28x28 灰度图 (MNIST) 进行了调整。
    
    原始 LeNet-5 的输入是 32x32，这里使用 28x28，
    但整体的 Conv -> Pool -> Conv -> Pool -> FC 结构保持不变。
    """
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        
        # 1. 卷积层 C1
        # 输入: (1, 28, 28)
        # 输出: (6, 28, 28)  (6个 5x5 卷积核, padding=2 保持尺寸)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # 激活函数: 使用 ReLU 取代 sigmoid/tanh，以加速训练
        
        # 2. 池化层 S2 (平均池化)
        # 输入: (6, 28, 28)
        # 输出: (6, 14, 14)  (2x2 池化，步长为 2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 3. 卷积层 C3
        # 输入: (6, 14, 14)
        # 输出: (16, 10, 10) (16个 5x5 卷积核, 无 padding)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # 4. 池化层 S4 (平均池化)
        # 输入: (16, 10, 10)
        # 输出: (16, 5, 5)   (2x2 池化，步长为 2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 计算进入全连接层时的展平尺寸: 16 * 5 * 5 = 400
        
        # 5. 全连接层 F5
        # 输入: 16 * 5 * 5 = 400
        # 输出: 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        
        # 6. 全连接层 F6
        # 输入: 120
        # 输出: 84
        self.fc2 = nn.Linear(120, 84)
        
        # 7. 输出层 Output
        # 输入: 84
        # 输出: num_classes (10)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # C1: Conv -> ReLU -> Pool
        x = self.pool1(F.relu(self.conv1(x)))
        
        # C3: Conv -> ReLU -> Pool
        x = self.pool2(F.relu(self.conv2(x)))
        
        # 展平操作 (Flatten): (Batch, 16, 5, 5) -> (Batch, 400)
        x = x.view(-1, 16 * 5 * 5)
        
        # F5: FC -> ReLU
        x = F.relu(self.fc1(x))
        
        # F6: FC -> ReLU
        x = F.relu(self.fc2(x))
        
        # Output: FC
        x = self.fc3(x)
        
        # 注意: 训练时通常直接输出 logits，不在这里加 softmax
        return x
    
#net = Net(num_classes=10).to(device)


import torch
import torch.nn as nn
import torch.nn.functional as F

class Gate_resnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super().__init__()
        self.in_planes = 16

        # ----- backbone 部分 -----
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # 用一个 Sequential 包起来，后面方便保存 / 调用
        self.backbone = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(inplace=True),
            self.layer1,
            self.layer2,
            self.layer3,
        )

        # ----- 分类头 -----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 等价于你原来的 avg_pool2d(out, 8)
        
        feat_dim = 64*block.expansion
        self.fc= nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, num_classes)
        )


        self._initialize_weights()
        
                # 读取两个 state_dict
        state_dict1 =torch.load("/home/zhy/Zhou/mixture_of_experts/_image_run/saved_cnn/_f5cifar10_cnn_backbone.pth", map_location="cpu")
        state_dict2 = torch.load("/home/zhy/Zhou/mixture_of_experts/_image_run/saved_cnn/_l5cifar10_cnn_backbone.pth", map_location="cpu")

        # 创建一个新的平均 state_dict
        state_dict = {}

        for key in state_dict1.keys():
            # 只对共有的参数做平均
            if key in state_dict2:
                state_dict[key] = (state_dict1[key] + state_dict2[key]) / 2
            else:
                state_dict[key] = state_dict1[key]
        self.backbone.load_state_dict(state_dict)
        self.backbone.requires_grad_(False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 提供一个显式的特征提取接口
    def extract_features(self, x):
        x = self.backbone(x)               # (B, 64, 8, 8)
        x = self.avgpool(x)                # (B, 64, 1, 1)
        x = torch.flatten(x, 1)            # (B, 64)
        return x

    def forward(self, x,train=True):
        x = self.extract_features(x)
        x = self.fc(x)
        clean,nosiy =x,x
        return x,clean,nosiy
    
    



def Gate_net_CIFAR10(num_classes=2):
    """
    ResNet-20: 包含3个Stage, 每个Stage有3个BasicBlock (3x2x3 + 2 = 20 层)
    """
    # num_blocks = [3, 3, 3]
    return Gate_resnet(BasicBlock, [3, 3, 3], num_classes=num_classes)




    
