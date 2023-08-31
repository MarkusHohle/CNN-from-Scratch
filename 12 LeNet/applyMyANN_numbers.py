# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:33:18 2022

@author: MMH_user
"""

def applyMyANN_numbers(M):
    
    import numpy as np
    
    #calling the ANN
    import ANN_MMH_L2 as My_ANN
    
    
    #initializing layers
    Conv1  = My_ANN.ConvLayer(5,5,6)
    Conv2  = My_ANN.ConvLayer(5,5,16)
    Conv3  = My_ANN.ConvLayer(5,5,120)
    
    AP1    = My_ANN.Average_Pool()
    AP2    = My_ANN.Average_Pool()
    
    T1 = My_ANN.Tanh()
    T2 = My_ANN.Tanh()
    T3 = My_ANN.Tanh()
    T4 = My_ANN.Tanh()

    F      = My_ANN.Flat()
    
    #calling weights/biases
    Conv1.weights = np.load('weightsC1.npy')
    Conv2.weights = np.load('weightsC2.npy')
    Conv3.weights = np.load('weightsC3.npy')
    
    Conv1.biases = np.load('biasC1.npy')
    Conv2.biases = np.load('biasC2.npy')
    Conv3.biases = np.load('biasC3.npy')
    
    
    Conv1.forward(M,0,1)
    T1.forward(Conv1.output)
    AP1.forward(T1.output,2,2)
    
    Conv2.forward(AP1.output,0,1)
    T2.forward(Conv2.output)
    AP2.forward(T2.output,2,2)

    Conv3.forward(AP2.output,2,3)
    T3.forward(Conv3.output)

    #flattening
    F.forward(T3.output)
    x = F.output

    P = x.shape[1]

    dense1          = My_ANN.Layer_Dense(P, 84)
    dense2          = My_ANN.Layer_Dense(len(dense1.biases.T), 10)
    
    dense1.weights = np.load('weights1.npy')
    dense2.weights = np.load('weights2.npy')
    
    dense1.biases = np.load('bias1.npy')
    dense2.biases = np.load('bias2.npy')

    dense1.forward(x)
    T4.forward(dense1.output)
    dense2.forward(T4.output)
    
    result = dense2.output
    
    #calculating the classes
    exp_values  = np.exp(result - np.max(result, axis = 1,\
                                  keepdims = True))
    probabilities = exp_values/np.sum(exp_values, axis = 1,\
                                  keepdims = True)
    
    predictions = np.argmax(probabilities, axis = 1)
    
    return(predictions,probabilities)












