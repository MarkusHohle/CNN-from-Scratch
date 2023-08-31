# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:50:23 2023

@author: hohle
"""

import numpy as np
import My_ANN_Loss_Backprob_Optimizer_ReadImds_Conv_MP as My_ANN
import matplotlib.pyplot as plt

minibatch_size = 7
ANN_size       = [128,128]

[M, C] = My_ANN.Read_Scale_Imds.Read_Scale(minibatch_size, ANN_size)

Conv1 = My_ANN.ConvLayer()
MP1   = My_ANN.Max_Pool()

Conv1.forward(M,2,1)
MP1.forward(Conv1.output,5,5)

MP1in  = MP1.inputs
MP1out = MP1.output
Mask   = MP1.mask

plt.imshow(MP1in[:,:,1,1], cmap = 'gray')
plt.show()
plt.imshow(MP1out[:,:,1,1], cmap = 'gray')
plt.show()
plt.imshow(Mask[:,:,1,1], cmap = 'gray')
plt.show()

I1 = MP1in[:,:,1,1]
I2 = MP1out[:,:,1,1]
I3 = Mask[:,:,1,1]

