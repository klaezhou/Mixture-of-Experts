from Pipeline.pl import Pipeline
import argparse
import torch
import numpy as np
import logging

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO, # 设置级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s', # 时间 - 级别 - 内容
    handlers=[
        logging.FileHandler("train.log"), # 保存到文件
        logging.StreamHandler()            # 输出到屏幕
    ]
)

logging.info("模型开始构建...")
def parse_args():
        parser = argparse.ArgumentParser(description="Train a Mixture-of-Experts model.")
        parser.add_argument("--epochs", type=int, default=6000, help="Number of training epochs.")
        parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate.")
        parser.add_argument("--device", type=str, default="cuda:7" if torch.cuda.is_available() else "cpu", help="Device to train on.")    
        parser.add_argument("--input_size", type=int, default=1,help="Input size (funcion approximation x) ")
        parser.add_argument("--output_size", type=int, default=1,help="Output size (funcion approximation y=u(x)) ")
        parser.add_argument("--num_experts", type=int, default=2,help="Number of experts")
        parser.add_argument("--hidden_size", type=int, default=10,help="Hidden size of the E")
        parser.add_argument("--gating_hidden_size", type=int, default=10,help="Hidden size of the G")
        parser.add_argument("--gating_depth", type=int, default=1,help="Depth of the G")
        parser.add_argument("--depth", type=int, default=4,help="Depth of the E")
        parser.add_argument("--log_interval", type=int, default=600, help="Logging interval.")
        parser.add_argument("--num_samples", type=int, default=1000)
        parser.add_argument("--smooth_steps",type=int,default=200,help="number of steps for smooth mode")
        parser.add_argument("--smooth_lb",type=int,default=20000,help="number lower bound of steps for smooth mode")
        parser.add_argument("--path",type=str,default="/home/zhy/Zhou/mixture_of_experts/cm_run/log_model/cm_model.pth",help="path to save the model")
        parser.add_argument("--seed",type=int,default=1234) #1234
        parser.add_argument("--epsilon",type=float,default=0)
        parser.add_argument("--eps_lb",type=float,default=-3)
        return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    Pipeline = Pipeline(args)
    Pipeline.data_loader()
    model = Pipeline.build_model()
    # Pipeline.load_model()
    print(model)
    
    Pipeline.Conv_data(91,20)
    Pipeline.train_model()
    Pipeline.Conv_data(51,20)
    Pipeline.train_model()
    Pipeline.Conv_data(31,15)
    Pipeline.train_model()
    args.epochs=20000
    Pipeline.Conv_data(1,1)
    
    Pipeline.train_model()
    
    

    
    
