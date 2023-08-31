# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 11:51:13 2022

@author: hohle
"""
I = [[1,-2,3,-4], [-4,6,0,-2]]

n_inputs   = 4  #number of features per data point
n_neurons1 = 10 #number of neurons in our 1st dense layer
n_neurons2 = 3 #number of neurons in our 2nd dense layer aka number of classes


#1) calling the network
import My_ANN as My_ANN

#2)initializing the Layer
dense1 = My_ANN.Layer_Dense(n_inputs,n_neurons1)
dense2 = My_ANN.Layer_Dense(n_neurons1,n_neurons2)

activation1 = My_ANN.Activation_ReLU()
activation2 = My_ANN.Activation_Softmax()


#3)feeding the data to the layer
dense1.forward(I)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)


#4) calling the result
out = activation2.output