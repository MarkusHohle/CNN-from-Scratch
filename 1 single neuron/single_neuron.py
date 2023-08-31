# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:35:12 2022

@author: hohle
"""

def single_neuron(inputs):
    
    import numpy as np
    
    l = len(inputs)
    
    weights = np.random.rand(1,l)
    bias    = np.random.rand(1,1)
    
    out = np.dot(weights,inputs) + bias
    
    return(out)