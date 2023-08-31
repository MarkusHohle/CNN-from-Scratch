# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 11:29:38 2022

@author: hohle
"""

import numpy as np
import glob as gl # for using ls like in linux
import random     # for picking mini batches randomly
from PIL import Image # for resizing images

class Read_Scale_Imds:
    
    def Read_Scale(minibatch_size, ANN_size):
        
        path_dogs = 'C:/Users/hohle/Desktop/QBM/courses/Python/ANN/pet pics/Dog/*.jpg' 
        path_cats = 'C:/Users/hohle/Desktop/QBM/courses/Python/ANN/pet pics/Cat/*.jpg'
        
        #creating a list of the images, like ls in linux
        Cats = gl.glob(path_cats)
        Dogs = gl.glob(path_dogs)
        
        Ld = len(Dogs)
        Lc = len(Cats)
        
        D = np.zeros((Ld,1))
        C = np.zeros((Lc,1)) + 1
        
        All_class = np.array(np.vstack((D,C)))
        All_imds  = np.hstack((Dogs,Cats))
        
        idx = random.sample(range(Ld+Lc),minibatch_size)
        
        classes = All_class[idx]
        imds    = All_imds[idx]
        
        #initializing a matrix for all the images
        #size: ANN size x ANN size x RGB x minibatch size
        ImdsMatrix = np.zeros((ANN_size[0],ANN_size[1],3,minibatch_size))
        
        for i in range(minibatch_size):
            
            I       = Image.open(imds[i])
            Ire     = I.resize((ANN_size[0],ANN_size[1]))
            Iar     = np.array(Ire)
            
            if len(Iar.shape) != 3:
                I3D        = np.zeros((ANN_size[0],ANN_size[1],3))
                I3D[:,:,0] = Iar
                I3D[:,:,1] = Iar
                I3D[:,:,2] = Iar
                Iar        = I3D
            
            ImdsMatrix[:,:,:,i] = Iar
            
        ImdsMatrix.astype(np.uint8)
        
        return(ImdsMatrix,classes)
        
    
###############################################################################
###############################################################################
class Creat_Kernels:
    
    def kernel_library():
        
        #creating kernels
        
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
        
        #later, the values in K are lernables, hence weights
        #K =  0.1*np.random.randn(xK,yK,NK)
        
        return(K)


###############################################################################
###############################################################################


class Layer_Dense:
    
    
    def __init__(self, n_inputs,n_neurons):
        
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases  = np.zeros((1,n_neurons))
        
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
        self.inputs = inputs
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dinputs  = np.dot(dvalues,self.weights.T)
        self.dbiases  = np.sum(dvalues, axis = 0, keepdims = True)
        
        
###############################################################################
###############################################################################

class Activation_ReLU:
    
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)
        self.inputs  = inputs
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0#!!!
        
        
###############################################################################
###############################################################################

class Activation_Softmax:
    
    def forward(self, inputs):
        
        exp_values    = np.exp(inputs - np.max(inputs))
        probabilities = exp_values/np.sum(exp_values, axis = 1, \
                                          keepdims = True)
        self.output   = probabilities
        
    def backward(self, dvalues):
        
        self.dinputs = np.empty_like(dvalues)
        
        for i, (single_output, single_dvalues) in enumerate(zip(self.output,dvalues)):
            
            single_output = single_output.reshape(-1,1)
            
            jacobMatr = np.diagflat(single_output) - \
                        np.dot(single_output,single_output.T)
            
            self.dinputs[i] = np.dot(jacobMatr,single_dvalues)
        
###############################################################################
###############################################################################

class Loss:
    
    def calculate(self, output, y):
        #output is the output from the softmax layer
        #vector/matrix of the actual class from the training data
        
        sample_losses = self.forward(output,y)
        data_loss     = np.mean(sample_losses)
        
        return(data_loss)
        
###############################################################################
###############################################################################

class Loss_CategoricalCrossEntropy(Loss):
    
    def forward(self, prob_pred, y_true):
        
        Nsamples = len(prob_pred)
        y_pred_clipped = np.clip(prob_pred, 1e-7, 1-1e-7)
        
        #checking if we have one hot or sparse
        if len(y_true.shape) == 1:#sparse [0,2,3,1,1,0]
            correct_confidences = y_pred_clipped[range(Nsamples),y_true]
            
        elif len(y_true.shape) == 2:# one hot [[0,1,0], [1,0,0]]
            correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        
        return(negative_log_likelihoods)
    
    
    def backward(self, dvalues, y_true):
        
        Nsamples = len(dvalues)
        
        if len(y_true.shape) == 1:
            Nlabels = len(dvalues[0])
            y_true  = np.eye(Nlabels)[y_true]
        
        self.dinputs = - y_true/dvalues/Nsamples
        
###############################################################################
###############################################################################

class CalcSoftmaxLossGrad:
    
    def __init__(self):
        
        self.activation = Activation_Softmax()
        self.loss       = Loss_CategoricalCrossEntropy()
        
    def forward(self, inputs, y_true):
        #inputs comes from the last dense layer (output) --> needed for 
        #softmax
        #y_true: vector/matrix of the true classes/labels
        self.activation.forward(inputs)
        self.output = self.activation.output
        
        return(self.loss.calculate(self.output, y_true))
    
    def backward(self, dvalues, y_true):
        #output from softmax layer
        Nsamples = len(dvalues)
        
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
        
        self.dinputs = dvalues.copy()
        #calculating the gradient
        self.dinputs[range(Nsamples),y_true] -= 1
        self.dinputs = self.dinputs/Nsamples
        
###############################################################################
###############################################################################

class Optimizer_SGD:
    
    def __init__(self, learning_rate = 0.1, decay = 0, momentum = 0):
        self.learning_rate         = learning_rate
        self.decay                 = decay
        self.current_learning_rate = learning_rate
        self.iterations            = 0
        self.momentum              = momentum
        
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate *\
                (1/(1 + self.iterations * self.decay))
        
    def update_params(self, layer):
        
        if self.momentum:
            
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums   = np.zeros_like(layer.biases)
            
            weight_updates = self.momentum * layer.weight_momentums - \
                            self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * layer.bias_momentums - \
                            self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        
        else:
            
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates   = -self.current_learning_rate * layer.dbiases
        
        layer.weights += weight_updates
        layer.biases  += bias_updates
        
    def post_update_params(self):
        self.iterations += 1























        
        
        
        
        