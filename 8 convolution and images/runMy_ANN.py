# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:50:23 2023

@author: hohle
"""

import numpy as np
import My_ANN_Loss_Backprob_Optimizer_ReadImds_Conv as My_ANN
import matplotlib.pyplot as plt

minibatch_size = 7
ANN_size       = [128,128]

[M, C] = My_ANN.Read_Scale_Imds.Read_Scale(minibatch_size, ANN_size)

Conv1 = My_ANN.ConvLayer()
Conv2 = My_ANN.ConvLayer()
Conv3 = My_ANN.ConvLayer()

Conv1.forward(M,2,1)
Conv2.forward(Conv1.output)
Conv3.forward(Conv2.output,0,3)

plt.imshow(M[:,:,0,1], cmap = 'gray')
plt.show()
plt.imshow(M[:,:,1,1], cmap = 'gray')
plt.show()
plt.imshow(M[:,:,2,1], cmap = 'gray')
plt.show()


plt.imshow(Conv1.output[:,:,0,1], cmap = 'gray')
plt.show()
plt.imshow(Conv1.output[:,:,1,1], cmap = 'gray')
plt.show()
plt.imshow(Conv1.output[:,:,2,1], cmap = 'gray')
plt.show()

plt.imshow(Conv2.output[:,:,0,1], cmap = 'gray')
plt.show()
plt.imshow(Conv2.output[:,:,1,1], cmap = 'gray')
plt.show()
plt.imshow(Conv2.output[:,:,2,1], cmap = 'gray')
plt.show()

plt.imshow(Conv3.output[:,:,0,1], cmap = 'gray')
plt.show()
plt.imshow(Conv3.output[:,:,1,1], cmap = 'gray')
plt.show()
plt.imshow(Conv3.output[:,:,2,1], cmap = 'gray')
plt.show()






