# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:37:42 2019

@author: chxy
"""

import torch
import torch.nn as nn
from model import FSRCNN_net
from data_loader import loadtraindata, loadtestdata
from torch.autograd import Variable

def train(num_channels=1, learning_rate=1e-3, epochs=2000):
    trainloader = loadtraindata()
    testloader = loadtestdata()
    net = FSRCNN_net(num_channels=num_channels).cuda()
    net.weight_init()
    
    deconv_params = list(map(id, net.deconv.parameters()))
    base_params = filter(lambda p: id(p) not in deconv_params, net.parameters())
    optimizer = torch.optim.Adam([
            {'params': base_params},
            {'params': net.deconv.parameters(), 'lr': learning_rate * 0.1}
            ], lr=learning_rate)

    #optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss().cuda()
    #train_loss_list = []
    #test_loss_list = []
    for epoch in range(epochs):
        train_loss = 0.0
        test_loss = 0.0
        for i, data in enumerate(trainloader):
            imgLR, label = data
            imgLR, label = imgLR.cuda(), label.cuda()
            imgLR, label = Variable(imgLR), Variable(label)
            imgHR = net(imgLR)
            loss = loss_fn(imgHR, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
#                pbar.set_description("MSE_loss: {:.3f}".format(running_loss))
#                pbar.update(trainloader.batch_size)
        #print("epoch : {}, train_loss : {:.3f}".format(epoch+1, running_loss))
        train_loss = train_loss / (i+1)
        for i, data in enumerate(testloader):
            imgLR, label = data
            imgLR, label = imgLR.cuda(), label.cuda()
            imgLR, label = Variable(imgLR), Variable(label)
            imgHR = net(imgLR)
            loss = loss_fn(imgHR, label)
            test_loss += loss.item()
        test_loss = test_loss / (i+1)
        print("epoch : {}, train_loss : {:.6f}, test_loss : {:.6f}".format(epoch+1, train_loss, test_loss))
        
        if (epoch+1) % 25 == 0: 
            torch.save(net.state_dict(), './ckpt/epoch_{}_train_loss_{:.6f}_test_loss_{:.6f}_net_params.pkl'.format(epoch+1, train_loss, test_loss))
        #train_loss_list.append(train_loss)
        #test_loss_list.append(test_loss)
        with open('loss.txt', 'a') as f:
            f.write("{:.6f} {:.6f}\n".format(train_loss, test_loss))

torch.cuda.set_device(7)
train()
#trainloader = loadtraindata()
#print(len(trainloader.dataset) // 100)
#print(trainloader.batch_size)
