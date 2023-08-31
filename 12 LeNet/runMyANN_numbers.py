# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:48:05 2022

@author: MMH_user
"""

def runMyANN_numbers(minibatch_size = 128, iterations = 20, epochs = 50):
    
    #calling the ANN
    import ANN_MMH_L2 as My_ANN
    import matplotlib.pyplot as plt
    import random 
    import numpy as np
    
    #calling the MNIST data set (numbers, black/white)
    #pip install keras
    #pip install tensorflow
    from keras.datasets import mnist

    (train_x, train_y), (test_X, test_y) = mnist.load_data()

    train_x   = train_x.transpose(1,2,0)#turning matrix in right direction, so
                                        #that it fits our ANN
                                    
    #our ANN wants 3D images
    S         = train_x.shape
    train_X3D = np.zeros((S[0],S[1],3,S[2]))
    
    train_X3D[:,:,0,:] = train_x
    train_X3D[:,:,1,:] = train_x
    train_X3D[:,:,2,:] = train_x
    
    #picking data randomly
    idx = random.sample(range(len(train_y)), minibatch_size)
    M   = train_X3D[:,:,:,idx]
    C   = train_y[idx]

    #making sure, that C has len(shape) = 0
    C = C.reshape(minibatch_size)
    C = C.astype(int)

###############################################################################
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
    
###############################################################################
    ###calling weights/biases
    ##if weights:
        
    #Conv1.weights = np.load('weights numbers/weightsC1.npy')
    #Conv2.weights = np.load('weights numbers/weightsC2.npy')
    #Conv3.weights = np.load('weights numbers/weightsC3.npy')
    
    #Conv1.biases = np.load('weights numbers/biasC1.npy')
    #Conv2.biases = np.load('weights numbers/biasC2.npy')
    #Conv3.biases = np.load('weights numbers/biasC3.npy')
###############################################################################


    #going through the layers at least once, in oder to determine P
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
    
###############################################################################
    ##calling weights/biases
    ##if weights:
        
    #dense1.weights = np.load('weights numbers/weights1.npy')
    #dense2.weights = np.load('weights numbers/weights2.npy')
    
    #dense1.biases = np.load('weights numbers/bias1.npy')
    #dense2.biases = np.load('weights numbers/bias2.npy')
###############################################################################
    
    optimizer       = My_ANN.Optimizer_SGD(learning_rate = 0.1, decay = 0.001,\
                                           momentum = 0.5)
    loss_activation = My_ANN.Activation_Softmax_Loss_CategoricalCrossentropy()
    
    
    
    ie      = iterations*epochs
    Monitor = np.zeros((ie,3))
    ct      = 0
    
    for Nep in range(epochs):
        
        if Nep >1:
            #calling new minibatch
            idx = random.sample(range(len(train_y)), minibatch_size)
            M   = train_X3D[:,:,:,idx]
            C   = train_y[idx]

            #making sure, that C has len(shape) = 0
            C = C.reshape(minibatch_size)
            C = C.astype(int)
        
        for it in range(iterations):
            
            Conv1.forward(M,0,1)
            T1.forward(Conv1.output)
            AP1.forward(T1.output,2,2)
            
            Conv2.forward(AP1.output,0,1)
            T2.forward(Conv2.output)
            AP2.forward(T2.output,2,2)
            
            #print(Conv3.weights[:,:,1])

            Conv3.forward(AP2.output,2,3)
            T3.forward(Conv3.output)

            #flattening
            F.forward(T3.output)
            x = F.output

                
            dense1.forward(x)
            T4.forward(dense1.output)
            dense2.forward(T4.output)
            loss = loss_activation.forward(dense2.output, C)
             
            predictions = np.argmax(loss_activation.output, axis = 1)
            if len(C.shape) == 2:
                C = np.argmax(C ,axis = 1)
            accuracy = np.mean(predictions == C)
               
            #backward passes
            loss_activation.backward(loss_activation.output, C)
            dense2.backward(loss_activation.dinputs)
            T4.backward(dense2.dinputs)
            dense1.backward(T4.dinputs)
                
            F.backward(dense1.dinputs)
            
            T3.backward(F.dinputs)
            Conv3.backward(T3.dinputs)

            AP2.backward(Conv3.dinputs)
            T2.backward(AP2.dinputs)
            Conv2.backward(T2.dinputs)
                        
            AP1.backward(Conv2.dinputs)
            T1.backward(AP1.dinputs)
            Conv1.backward(T1.dinputs)
         
            optimizer.pre_update_params()#decaying learning rate
                
            optimizer.update_params(dense1)
            optimizer.update_params(dense2)
                                
            optimizer.update_params(Conv1)
            optimizer.update_params(Conv2)
            optimizer.update_params(Conv3)
                
            optimizer.post_update_params()#just increasing iteration by one
                
            Monitor[ct,0] = accuracy
            Monitor[ct,1] = loss
            Monitor[ct,2] = optimizer.current_learning_rate
            
            ct += 1
                
   
            print(f'epoch: {Nep}, ' +
                 f'iteration: {it}, ' +
                 f'accuracy: {accuracy:.3f}, ' +
                 f'loss: {loss:.3f}, ' +
                 f'current learning rate: {optimizer.current_learning_rate:.5f}')
    
        #saving learnables
        
        np.save('weights1.npy', dense1.weights)
        np.save('weights2.npy', dense2.weights)
        
        np.save('bias1.npy', dense1.biases)
        np.save('bias2.npy', dense2.biases)
        
        np.save('weightsC1.npy', Conv1.weights)
        np.save('weightsC2.npy', Conv2.weights)
        np.save('weightsC3.npy', Conv3.weights)
        
        np.save('biasC1.npy', Conv1.biases)
        np.save('biasC2.npy', Conv2.biases)
        np.save('biasC3.npy', Conv3.biases)
        
        np.savetxt('Monitor1.txt',Monitor)
    
    
    
    Vie   = np.arange(ie)
    VieIt = np.arange(epochs) *iterations
    
    fig, ax = plt.subplots(3, 1,sharex=True)
    ax[0].plot(Vie, Monitor[:,0]*100)
    for xh in VieIt:
        ax[0].axvline(x=xh,ymin=0, ymax=100, color = 'black')
    ax[0].set_ylabel('accuracy [%]')
    ax[1].plot(Vie, Monitor[:,1])
    for xh in VieIt:
        ax[1].axvline(x=xh,ymin=0, ymax=100, color = 'black')
    ax[1].set_ylabel('loss')
    ax[2].plot(Vie, Monitor[:,2])
    ax[2].set_ylabel(r'$\alpha$')
    ax[2].set_xlabel('epoch * iterations')
    #plt.xscale('log',base=10) 