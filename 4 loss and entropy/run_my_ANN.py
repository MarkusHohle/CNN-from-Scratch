# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 11:41:55 2022

@author: hohle
"""

import numpy as np
import matplotlib.pyplot as plt

from nnfs.datasets import spiral_data

nc = 3 #number of different classes
N  = 200 #number of data points per class

[x, y] = spiral_data(N,nc) #x: actual data; y: vector containing the classes
S      = np.shape(x)



#1) calling the network
import My_ANN_Loss as My_ANN

#2)initializing the Layer
dense1 = My_ANN.Layer_Dense(S[1],64)
dense2 = My_ANN.Layer_Dense(64,nc)

activation1 = My_ANN.Activation_ReLU()
activation2 = My_ANN.Activation_Softmax()

loss_function = My_ANN.Loss_CategoricalCrossEntropy()



#3)feeding the data to the layer
dense1.forward(x)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

#calculating cross entropy aka how certain the ANN is about its decissions
loss = loss_function.calculate(activation2.output, y)

predictions = np.argmax(activation2.output, axis = 1)

if len(y.shape) == 2:
        y = np.argmax(y,axis = 1)
        
accuracy = np.mean(predictions == y)


















