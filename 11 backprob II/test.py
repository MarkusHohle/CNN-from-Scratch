# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:14:52 2023

@author: hohle
"""

import My_ANN_BPFlat_Sigm_MaxPool_Conv as My_ANN
import matplotlib.pyplot as plt

[M, C] = My_ANN.Read_Scale_Imds.Read_Scale(4, [128,128])

#initializing layers

Conv1 = My_ANN.ConvLayer()
Conv2 = My_ANN.ConvLayer()
Conv3 = My_ANN.ConvLayer()

RL1   = My_ANN.Activation_ReLU()
RL2   = My_ANN.Activation_ReLU()
RL3   = My_ANN.Activation_ReLU()

MP1   = My_ANN.Max_Pool()
MP2   = My_ANN.Max_Pool()
MP3   = My_ANN.Max_Pool()

#forward part

Conv1.forward(M,2,1)
RL1.forward(Conv1.output)
MP1.forward(RL1.output,4,4)

Conv2.forward(MP1.output,1,1)
RL2.forward(Conv2.output)
MP2.forward(RL2.output,1,4)

Conv3.forward(MP2.output,0,3)
RL3.forward(Conv3.output)
MP3.forward(RL3.output,2,2)

#backward part

MP3.backward(MP3.output)
RL3.backward(MP3.dinputs)
Conv3.backward(RL3.dinputs)

MP2.backward(Conv3.dinputs)
RL2.backward(MP2.dinputs)
Conv2.backward(RL2.dinputs)

MP1.backward(Conv2.dinputs)
RL1.backward(MP1.dinputs)
Conv1.backward(RL1.dinputs)


#just checking

plt.imshow(M[:,:,1,1])

plt.imshow(Conv1.output[:,:,1,1])
plt.imshow(RL1.output[:,:,1,1])
plt.imshow(MP1.output[:,:,1,1])


plt.imshow(Conv1.dinputs[:,:,1,1])
plt.imshow(RL1.dinputs[:,:,1,1])
plt.imshow(MP1.dinputs[:,:,1,1])




