# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:14:27 2019

@author: chxy
"""

import torch
from model import FSRCNN_net
import scipy.io
import pickle
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

#class test_net(torch.nn.Module):
#    def __init__(self, num_channels):
#        super(test_net, self).__init__()
#        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=9, stride=1, padding=4, bias=True)
#        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)
#        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2, bias=True)
#
#    def forward(self, x):
#        out = F.relu(self.conv1(x))
#        out = F.relu(self.conv2(out))
#        out = self.conv3(out)
#        return out

net = FSRCNN_net(num_channels=1)
net.load_state_dict(torch.load('./epoch_2000_train_loss_0.001252_test_loss_0.000565_net_params.pkl', map_location='cpu'))
#print(net)
#scipy.io.savemat('3_9_1_5.mat', mdict={'net':net})

#f = open('./ckpt/epoch_500_loss_0.2217_net_params.pkl')  
#data = pickle.load(f)
#print(data)

#print(net.state_dict().keys())

for name in net.state_dict():
   print(name)
   print(net.state_dict()[name].data.cpu().numpy().dtype)


weight = dict()
weight['weights_conv1'] = net.state_dict()['conv1.weight'].data.cpu().numpy().astype(np.float64).transpose(1,2,3,0).reshape(1, 25, 32)
weight['weights_conv2'] = net.state_dict()['conv2.weight'].data.cpu().numpy().astype(np.float64).transpose(1,2,3,0).reshape(32, 1, 5)
weight['weights_conv3'] = net.state_dict()['conv3.weight'].data.cpu().numpy().astype(np.float64).transpose(1,2,3,0).reshape(5, 9, 5)
weight['weights_conv4'] = net.state_dict()['conv4.weight'].data.cpu().numpy().astype(np.float64).transpose(1,2,3,0).reshape(5, 1, 32)
weight['weights_deconv'] = net.state_dict()['deconv.weight'].data.cpu().numpy().astype(np.float64).transpose(1,2,3,0).reshape(1, 81, 32)

#weight['weights_conv1'] = net.state_dict()['conv1.weight'].data.cpu().numpy().astype(np.float64).transpose(1,3,2,0).reshape(1, 25, 32)
#weight['weights_conv2'] = net.state_dict()['conv2.weight'].data.cpu().numpy().astype(np.float64).transpose(1,3,2,0).reshape(32, 1, 5)
#weight['weights_conv3'] = net.state_dict()['conv3.weight'].data.cpu().numpy().astype(np.float64).transpose(1,3,2,0).reshape(5, 9, 5)
#weight['weights_conv4'] = net.state_dict()['conv4.weight'].data.cpu().numpy().astype(np.float64).transpose(1,3,2,0).reshape(5, 1, 32)
#weight['weights_deconv'] = net.state_dict()['deconv.weight'].data.cpu().numpy().astype(np.float64).transpose(1,3,2,0).reshape(1, 81, 32)

weight['biases_conv1'] = net.state_dict()['conv1.bias'].data.cpu().numpy().astype(np.float64)
weight['biases_conv2'] = net.state_dict()['conv2.bias'].data.cpu().numpy().astype(np.float64)
weight['biases_conv3'] = net.state_dict()['conv3.bias'].data.cpu().numpy().astype(np.float64)
weight['biases_conv4'] = net.state_dict()['conv4.bias'].data.cpu().numpy().astype(np.float64)
weight['biases_deconv'] = net.state_dict()['deconv.bias'].data.cpu().numpy().astype(np.float64)

weight['prelu1'] = net.state_dict()['relu1.weight'].data.cpu().numpy().astype(np.float64)
weight['prelu2'] = net.state_dict()['relu2.weight'].data.cpu().numpy().astype(np.float64)
weight['prelu3'] = net.state_dict()['relu3.weight'].data.cpu().numpy().astype(np.float64)
weight['prelu4'] = net.state_dict()['relu4.weight'].data.cpu().numpy().astype(np.float64)
weight['prelu5'] = np.array([])

for key in weight.keys():
    print(weight[key].shape)
    
scipy.io.savemat('mine_2000.mat', mdict=weight)