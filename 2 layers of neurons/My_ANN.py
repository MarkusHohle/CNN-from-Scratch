# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 11:29:38 2022

@author: hohle
"""

import numpy as np

class Layer_Dense:
    
    
    
    def __init__(self, n_inputs,n_neurons):
        
        self.weights = np.random.rand(n_inputs,n_neurons)
        self.biases  = np.zeros((1,n_neurons))
        
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
        