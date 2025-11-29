# initialize the network
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 定义核心：基础残差块 (BasicBlock) ---
class BasicBlock(nn.Module):
    """
    ResNet的基本残差块，适用于ResNet-18/34 (使用两个3x3卷积层)
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        # 第一个卷积层: 负责降维/改变特征图大小 (如果 stride > 1)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 第二个卷积层: 保持尺寸不变
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 定义快捷连接 (Shortcut): 只有当输入和输出特征图尺寸或通道数不匹配时才需要
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # 卷积 -> BN -> ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # 卷积 -> BN
        out = self.bn2(self.conv2(out))
        # 核心：残差连接 (快捷连接 + 卷积输出)
        out += self.shortcut(x)
        # 激活
        out = F.relu(out)
        return out


# --- 2. 定义主模型：ResNet ---
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16  # CIFAR-10 ResNet从16个通道开始

        # 1. 初始卷积层：针对32x32输入，使用3x3, stride=1, 不使用MaxPool
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 2. 残差层 Stage 1-3：通道数从16 -> 32 -> 64
        # 第一个stage保持尺寸 (32x32)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        # 第二个stage尺寸减半 (16x16)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        # 第三个stage尺寸再减半 (8x8)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # 3. 分类层
        self.linear = nn.Linear(64 * block.expansion, num_classes)
        
        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        # 确定是否需要降采样 (downsample)
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # (B, 16, 32, 32)
        out = self.layer1(out)               # (B, 16, 32, 32)
        out = self.layer2(out)               # (B, 32, 16, 16)
        out = self.layer3(out)               # (B, 64, 8, 8)
        
        # 全局平均池化 (Global Average Pooling): 尺寸从 (B, 64, 8, 8) -> (B, 64, 1, 1)
        out = F.avg_pool2d(out, 8) 
        
        # 展平: (B, 64)
        out = out.view(out.size(0), -1)
        # 全连接层分类
        out = self.linear(out)
        return out


# --- 3. 工厂函数：创建 ResNet-20 实例 ---
def ResNet20(num_classes=10):
    """
    ResNet-20: 包含3个Stage, 每个Stage有3个BasicBlock (3x2x3 + 2 = 20 层)
    """
    # num_blocks = [3, 3, 3]
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
    """
    基于 LeNet-5 架构，修改以适应 CIFAR-10 (3x32x32, 10 类别) 的输入。
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(Gate_resnet, self).__init__()
        self.in_planes = 16  # CIFAR-10 ResNet从16个通道开始

        # 1. 初始卷积层：针对32x32输入，使用3x3, stride=1, 不使用MaxPool
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 2. 残差层 Stage 1-3：通道数从16 -> 32 -> 64
        # 第一个stage保持尺寸 (32x32)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        # 第二个stage尺寸减半 (16x16)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        # 第三个stage尺寸再减半 (8x8)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # 3. 分类层
        self.linear = nn.Linear(64 * block.expansion, num_classes)
        
        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        # 确定是否需要降采样 (downsample)
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x,train=True):
        out = F.relu(self.bn1(self.conv1(x))) # (B, 16, 32, 32)
        out = self.layer1(out)               # (B, 16, 32, 32)
        out = self.layer2(out)               # (B, 32, 16, 16)
        out = self.layer3(out)               # (B, 64, 8, 8)
        
        # 全局平均池化 (Global Average Pooling): 尺寸从 (B, 64, 8, 8) -> (B, 64, 1, 1)
        out = F.avg_pool2d(out, 8) 
        
        # 展平: (B, 64)
        out = out.view(out.size(0), -1)
        # 全连接层分类
        out = self.linear(out)
        clean,noisy_stddev=x,x
        return out,clean,noisy_stddev
    
    



def Gate_net_CIFAR10(num_classes=2):
    """
    ResNet-20: 包含3个Stage, 每个Stage有3个BasicBlock (3x2x3 + 2 = 20 层)
    """
    # num_blocks = [3, 3, 3]
    return Gate_resnet(BasicBlock, [3, 3, 3], num_classes=num_classes)