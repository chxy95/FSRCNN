# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:03:01 2019

@author: chxy
"""

import matplotlib.pyplot as plt
import numpy as np

train_loss=[]
test_loss=[]
with open('loss2.txt', 'r') as f:
    for line in f:
        train_loss.append(line[:-1].split(' ')[0])
        test_loss.append(line[:-1].split(' ')[1])

x = []
y1 = []
y2 = []
for i in range(10, len(train_loss)+1, 50):
    x.append(i)
    y1.append(train_loss[i])
    y2.append(test_loss[i])
x = np.array(x).astype(np.int)
y1 = np.array(y1).astype(np.float32)
y2 = np.array(y2).astype(np.float32)

plt.figure()
plt.title('loss vs. epoches')
plt.xlabel('epoch')
plt.ylabel('mse_loss')
#plt.plot(x, y1, color='b', label = 'train_loss')
plt.plot(x, y2, color='g', label = 'test_loss')
plt.legend()
plt.savefig('test_loss.png', format='png', bbox_inches = 'tight')
plt.show()