# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:36:43 2023

@author: hohle
"""

def ConvSelfMade(*Image, K, padding = 0, stride = 1):
    
    import matplotlib.pyplot as plt
    import numpy as np
    #from scipy.signal import convolve as Conv # N-Dim convolution
    
    if Image:
        
        for Image in Image:
            plt.imshow(Image)
            plt.show()
            
    else:
        Image = plt.imread('C:/Users/hohle/Desktop/QBM/courses/Python/ANN/pet pics/Cat/2.jpg')
        plt.imshow(Image)
        plt.show()
        
    xImgShape = Image.shape[0]
    yImgShape = Image.shape[1]
    numChans  = Image.shape[2]
    
    xK = K.shape[0]
    yK = K.shape[1]
    
    #calculating the size of the output image
    xOutput = int(((xImgShape - xK + 2*padding)/stride) + 1)
    yOutput = int(((yImgShape - yK + 2*padding)/stride) + 1)
    
    #initializing empty output matrix
    output = np.zeros((xOutput, yOutput, numChans))
    
    imagePadded = np.zeros((xImgShape + 2*padding, yImgShape + 2*padding, \
                            numChans))
    imagePadded[int(padding):int(padding+xImgShape),\
                int(padding):int(padding+yImgShape),:] = Image
        
    
    for c in range(numChans):# loop over the color channels
        for y in range(yOutput):# loop over y axis of the output
            for x in range(xOutput):# loop over x axis of the output
            
                #finding the corners of the current slice
                y_start = y*stride
                y_end   = y_start + yK
                x_start = x*stride
                x_end   = x_start + xK
                
                current_slice = imagePadded[x_start:x_end, y_start:y_end,c]
                
                #the actual convolution part
                s             = np.multiply(current_slice,K)
                output[x,y,c] = np.sum(s)
            
    
    plt.imshow(output.sum(2), cmap = 'gray')
    
    return(imagePadded)





























