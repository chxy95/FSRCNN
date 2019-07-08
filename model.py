# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:54:40 2019

@author: chxy
"""

#FSRCNN_s (32, 5, 1), scale = 3, input_size = 11, output_size = 19

import torch
import torch.nn as nn
#import torch.nn.functional as F

class FSRCNN_net(torch.nn.Module):
    def __init__(self, num_channels=1):
        super(FSRCNN_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=5, stride=1, padding=0, bias=True)
        self.relu1 = nn.PReLU(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=5, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu2 = nn.PReLU(5)
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu3 = nn.PReLU(5)
        self.conv4 = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu4 = nn.PReLU(32)
        self.deconv = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=9, stride=3, padding=4, bias=True)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = self.deconv(out)
        return out
    
    def weight_init(self):
        nn.init.normal(self.conv1.weight, mean=0, std=0.05)
        nn.init.normal(self.conv2.weight, mean=0, std=0.6325)
        nn.init.normal(self.conv3.weight, mean=0, std=0.2108)
        nn.init.normal(self.conv4.weight, mean=0, std=0.25)
        nn.init.normal(self.deconv.weight, mean=0, std=0.001)
        
    
#net = FSRCNN_net()
#inputs = torch.autograd.Variable(torch.randn(1, 1, 11, 11))
#outputs = net(inputs)
#print(outputs.size())