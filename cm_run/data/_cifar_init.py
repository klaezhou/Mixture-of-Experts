import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt

class data_generator():
    def __init__(self,class_index):



        # CIFAR-10 的所有类别
        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # 1. 定义我们想要的类别及其索引 (0到4)
        TARGET_CLASSES_NAMES = [classes[i] for i in class_index]
        TARGET_CLASSES_INDICES = list(class_index)  
        CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR10_STD = (0.2023, 0.1994, 0.2010)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

        batch_size = 128
        trainset_full = torchvision.datasets.CIFAR10(root='CIFAR10/', 
                                                    train=True,
                                                    download=True, 
                                                    transform=train_transform)

        indices_to_keep = []
        for i, label in enumerate(trainset_full.targets):
            # trainset_full.targets 是一个包含所有样本标签（0-9）的列表
            if label in TARGET_CLASSES_INDICES:
                indices_to_keep.append(i)

        trainset_subset = Subset(trainset_full, indices_to_keep)

        # 4. 载入筛选后的数据
        self.trainloader = torch.utils.data.DataLoader(trainset_subset, 
                                                batch_size=batch_size,
                                                shuffle=True, 
                                                num_workers=2)

        # 5. 可选：检查结果
        print(f"原始训练集大小: {len(trainset_full)}")
        print(f"筛选后的训练集大小: {len(trainset_subset)}")
        print(f"筛选后的类别: {TARGET_CLASSES_NAMES}")

        # --- 测试集保持不变（使用您原来的代码，假设您不筛选测试集）---
        testset = torchvision.datasets.CIFAR10(root='CIFAR10/', train=False,
                                            download=True, transform=test_transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        
    def get_data(self): 
        return self.trainloader,self.testloader 