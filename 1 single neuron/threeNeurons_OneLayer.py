# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:57:41 2022

@author: hohle
"""

def threeNeurons_OneLayer(inputs):
    
    import numpy as np
    
    l = len(inputs)
    
    weights = np.random.rand(3,l)
    bias    = np.random.rand(3)
    
    out = np.dot(weights,inputs) + bias
    
    return(out)