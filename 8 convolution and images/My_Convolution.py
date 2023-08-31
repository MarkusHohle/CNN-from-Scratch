# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:25:48 2023

@author: hohle
"""

def My_Convolution(*Image):
    #  * indicates an optional input argument
    
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import convolve as Conv # N-Dim convolution
    
    if Image:
        
        for Image in Image:
            plt.imshow(Image)
            
    else:
        Image = plt.imread('C:/Users/hohle/Desktop/QBM/courses/Python/ANN/pet pics/Cat/2.jpg')
        plt.imshow(Image)
        
    K1  = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    #edges
    K2  = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
    K3  = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    K4  = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    #sharpen
    K5  = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    #blur
    K6  = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    K6  = K6/9
    K7  = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    
   #skeewed
    K8  = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
   #misc
    K9  = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    K10 = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    K11 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    K12 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    K13 = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
    
    K = np.dstack((K1,K2,K3,K4,K5,K6,K7,K8,K9,K10,K11,K12,K13))
    
    
    NK = K.shape[2]
    NC = Image.shape[2]
    
    PS = np.math.ceil(NK**0.5)
    
    for i in range(NC):
        
        plt.figure(figsize = (15,12))
        plt.suptitle('after convolution of channel ' + str(i+1), \
                     fontsize = 20, y = 0.95)
        plt.subplots_adjust(hspace = 0.5)
        
        
        for k in range(NK):
            
            plt.subplot(PS,PS,k+1)
            Out = Conv(Image[:,:,i],K[:,:,k]) #the actual convolution
            plt.imshow(Out, cmap = 'gray')
            
        plt.show()
        
        
        
        
        
        
    
    
    