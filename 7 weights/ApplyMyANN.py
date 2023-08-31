# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:51:59 2023

@author: hohle
"""

def ApplyMyANN(x_new):
    
    import numpy as np
    #1) calling the network
    import My_ANN_Loss_Backprob_Optimizer as My_ANN
    
    #2) loading weights and biases from previous training session
    w1 = np.load('weights1.npy')
    w2 = np.load('weights2.npy')
    
    b1 = np.load('biases1.npy')
    b2 = np.load('biases2.npy')
    
    #3) initializing the ANN
    s    = w1.shape
    nrow = s[0] 
    ncol = s[1]
    
    nc = 3
    
    dense1 = My_ANN.Layer_Dense(nrow,ncol)
    dense2 = My_ANN.Layer_Dense(len(dense1.biases.T),nc)

    activation1 = My_ANN.Activation_ReLU()
    
    #4) transferring the weights/biases
    dense1.weights = w1
    dense2.weights = w2
    
    dense1.biases  = b1
    dense2.biases  = b2
    
    #5) feeding the ANN with the new data set
    dense1.forward(x_new)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    result = dense2.output
    
    #6) calculating the probabilities
    exp_values    = np.exp(result - np.max(result, axis = 1, keepdims = True))
    
    probabilities = exp_values/np.sum(exp_values, axis = 1, \
                                      keepdims = True)
    
    predictions   = np.argmax(probabilities, axis = 1) 
    
    
    return(probabilities, predictions)
    
    
    
    
    