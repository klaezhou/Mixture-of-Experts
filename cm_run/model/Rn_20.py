import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# --- 1. 定义核心：基础残差块 (BasicBlock) ---
class BasicBlock(nn.Module):
    """
    ResNet的基本残差块，适用于ResNet-18/34 (使用两个3x3卷积层)
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        # 第一个卷积层: 负责降维/改变特征图大小 (如果 stride > 1)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 第二个卷积层: 保持尺寸不变
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 定义快捷连接 (Shortcut): 只有当输入和输出特征图尺寸或通道数不匹配时才需要
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
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
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 2. 残差层 Stage 1-3：通道数从16 -> 32 -> 64
        # 第一个stage保持尺寸 (32x32)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        # 第二个stage尺寸减半 (16x16)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        # 第三个stage尺寸再减半 (8x8)
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

        # self.tau1, self.tau2 = nn.Parameter(torch.tensor(1e-5)), nn.Parameter(torch.tensor(1e-5))
        # self.tau1.requires_grad, self.tau2.requires_grad = True, True

        # 3. 分类层ss
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 等价于你原来的 avg_pool2d(out, 8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        
        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        # 确定是否需要降采样 (downsample)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def soft_pool(self, x):
        k=32
        x = torch.flatten(x, 2)  # (N, C, H*W)

        diff = torch.unsqueeze(x,-1) - torch.unsqueeze(x,-2)    # (N, C, H*W, H*W)
        sigma = torch.sigmoid(-diff / self.tau1)                # (N, C, H*W, H*W)
        row_sum = sigma.sum(dim=-1) - 0.5                       # (N, C, H*W)
        r_tilde = 1 + row_sum                                   # (N, C, H*W)
        eps = 0.5
        a = torch.sigmoid((k + eps - r_tilde) / self.tau2)  

        x = x * a

        # weights = weights / (weights.sum(dim=(2,3), keepdim=True) + 1e-8)

        # 执行加权池化
        y = x.sum(dim=-1, keepdim=True)/k
        return y

    def hard_pool(self, x):
        k=16
        x = torch.flatten(x, 2)
        value,index= torch.topk(x, k, dim=-1)

        # 执行加权池化
        y = value.sum(dim=-1, keepdim=True)/k
        return y
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


# --- 3. 工厂函数：创建 ResNet-20 实例 ---
def ResNet20(num_classes=10):
    """
    ResNet-20: 包含3个Stage, 每个Stage有3个BasicBlock (3x2x3 + 2 = 20 层)
    """
    # num_blocks = [3, 3, 3]
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


