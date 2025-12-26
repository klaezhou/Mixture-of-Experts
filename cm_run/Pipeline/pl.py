import torch
import torch.nn as nn
import numpy as np 
from torch.distributions.normal import Normal
import torch.nn.functional as F
from functorch import make_functional, vmap
import logging
import torch.nn.init as init
from model.cm_model import PF_moe, Conv_nn
from data._data import data_generator, plot_data

class Pipeline():
    def __init__(self,args):
        self.args=args
        
    
    def data_loader(self):
        data_gen=data_generator(num_samples=self.args.num_samples,device=self.args.device)
        x,y=data_gen.get_data()
        
        self.x=x
        self.y=y
        self.train_data=(x,y)
        
    
    def Conv_data(self,kernel_size,stride):
        input_size=self.args.num_samples
        Conv=Conv_nn(input_size,kernel_size,stride)
        self.Conv=Conv.to(self.args.device)
        print(self.y.shape)
        y=self.y.T
        y_conv=self.Conv(y)
        y_conv=y_conv.T
        x=self.x.T
        x_conv=self.Conv.get_input(x)
        x_conv=x_conv.T
        print(f"After convolution, x shape: {x_conv.shape}, y shape: {y_conv.shape}")
        
        self.train_data=(x_conv,y_conv)
        
    def build_model(self):
        CMM=PF_moe(self.args.input_size,self.args.output_size,self.args.hidden_size,self.args.depth,self.args.gating_hidden_size,self.args.gating_depth)
        self.model=CMM.to(self.args.device)
        return self.model

    def load_model(self):
        self.model.load_state_dict(torch.load(self.args.path))
        self.model.to(self.args.device)
    def train_model(self):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        x, y = self.train_data[0], self.train_data[1]
        step_count=self.args.smooth_steps
        loss_fn = nn.MSELoss()
        for epoch in range(self.args.epochs):
            step_count -=1
            if step_count<=0 and epoch<=self.args.smooth_lb:
                with torch.no_grad():
                    self.model.gating_network.softmax_eps.copy_(torch.tensor(self.args.eps_lb
                                                                , device=x.device)) 
                step_count=self.args.smooth_steps
            
            optimizer.zero_grad()
            outputs = self.model(x)
            # well_potential= loss_fn(outputs, )
            reg=self.args.epsilon*(1/torch.exp(self.model.gating_network.softmax_eps))
            loss = loss_fn(outputs, y)
            total_loss=loss+reg 
            total_loss.backward()
            optimizer.step()
            if (epoch+1) % self.args.log_interval == 0:
                logging.info(f'Epoch [{epoch+1}/{self.args.epochs}], Loss: {loss.item():.8f},reg: {reg.item():.8f},softmax_eps: {torch.exp(self.model.gating_network.softmax_eps).item():.8f}')
                plot_data(x, y, self.model).plot(1)
                plot_data(x,y,self.model.gating_network).plot(2)
                
                
            
                
        torch.save(self.model.state_dict(), self.args.path)
        
    