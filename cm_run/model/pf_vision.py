import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from model.Rn_20 import ResNet20

class GatingNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,depth=1,output_size=1,activation=nn.ReLU()):
        self.depth=depth
        super(GatingNetwork, self).__init__()
        
        self.activation=activation
        layer_list = []
        layer_list.append(nn.Linear(input_size,hidden_size))  #[I,H]
        for i in range(self.depth-1):
            layer_list.append(nn.Linear(hidden_size,hidden_size))  #[H,H]
        layer_list.append(nn.Linear(hidden_size,output_size,bias=False))  #[H,I] zhou's model
        self.net = nn.ModuleList(layer_list)
        self.softmax_eps=nn.Parameter(torch.tensor(0.0))
        self.softmax_eps.requires_grad=True
        self._init_weights()
        self.conv=1
        self.conti_conv= nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding='same',bias=False) # groups = in_channels
        self.conti_conv.requires_grad=False
    def _init_weights(self):
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)  # Xavier 正态分布初始化
                    if m.bias is not None:
                        init.zeros_(m.bias)

    def _set_conv(self,kernel_size=11):
        self.conv=kernel_size
        if kernel_size==0:
            return
        self.conti_conv=nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size, stride=1, padding='same',bias=False).to(self.args.device)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                    C_in = m.in_channels
                    K = m.kernel_size[0]
                    # 基础平均值：考虑了通道数
                    val = 1.0 / (C_in * K**2)
                    
                    # 将权重全部初始化为固定值 1/k
                    init.constant_(m.weight, val)
        self.conti_conv.requires_grad=False

    
    def forward(self, y):
        if self.conv>0:
            y=self.conti_conv(y)
            
        y = y.view(y.size(0), -1) #3072
        for i, layer in enumerate(self.net[:-1]):
            y = layer(y)
            y = self.activation(y)
        y = self.net[-1](y)
        y= F.sigmoid(y/torch.exp(self.softmax_eps))
        return y
    
    
    
class PF_moe(nn.Module):
    def __init__(self,args):
        super(PF_moe, self).__init__()
        self.args=args
        
        self.expert1=ResNet20(num_classes=args.class_num).to(args.device)
        self.expert2=ResNet20(num_classes=args.class_num).to(args.device)
        self.expert1.load_state_dict(torch.load(args.path1))
        self.expert1.requires_grad_(False)
        
        self.expert2.load_state_dict(torch.load(args.path2))
        self.expert2.requires_grad_(False)
        self.gating_network=GatingNetwork(args.input_size,args.gating_hidden_size,args.gating_depth).to(args.device)
        self.gating_network.args=args
        self.report_numbertrainable()
    
    def report_numbertrainable(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
    def forward(self, x):
        gating_output=self.gating_network(x)  #[N,1]
        expert_output1=self.expert1(x)        #[N,O]
        expert_output2=self.expert2(x)        #[N,O]
        output=expert_output1*gating_output + expert_output2*(1-gating_output)  #[N,O] element-wise multiplication with broadcasting
        return output