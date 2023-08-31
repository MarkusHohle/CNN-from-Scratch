# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 11:16:49 2022

@author: hohle
"""

def run_my_ANN(Nsteps):
    
    import numpy as np
    import matplotlib.pyplot as plt

    from nnfs.datasets import spiral_data

    nc = 3 #number of different classes
    N  = 600 #number of data points per class

    [x, y] = spiral_data(N,nc) #x: actual data; y: vector containing the classes
    S      = np.shape(x)

    #1) calling the network
    import My_ANN_Loss_Backprob_Optimizer as My_ANN

    #2)initializing the Layer
    dense1 = My_ANN.Layer_Dense(S[1],64)
    dense2 = My_ANN.Layer_Dense(64,nc)

    activation1 = My_ANN.Activation_ReLU()

    loss_function = My_ANN.CalcSoftmaxLossGrad()
    
    optimizer    = My_ANN.Optimizer_SGD(0.2,decay = 0.001, momentum = 0.9)
###############################################################################

    Monitor = np.zeros((Nsteps,3))

    for epoch in range(Nsteps):
        
        #3)feeding the data to the layer
        dense1.forward(x)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        #calculating cross entropy aka how certain the ANN is about its decissions
        loss = loss_function.forward(dense2.output, y)


        predictions = np.argmax(loss_function.output, axis = 1)

        if len(y.shape) == 2:
                y = np.argmax(y,axis = 1)
                
        accuracy = np.mean(predictions == y)

        #3)the backpropagation part
        loss_function.backward(loss_function.output, y)
        dense2.backward(loss_function.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        
        #5)updating the parameters
        optimizer.pre_update_params()
        
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        
        optimizer.post_update_params()
        
        Monitor[epoch,0] = accuracy
        Monitor[epoch,1] = loss
        Monitor[epoch,2] = optimizer.current_learning_rate
        
        if not epoch % 1000:
            print(f'epoch: {epoch}, ' +
                  f'accuracy: {accuracy: .3f}, ' +
                  f'loss: {loss: .3f}')
        
    np.save('weights1.npy',dense1.weights)
    np.save('weights2.npy',dense2.weights)
    
    np.save('biases1.npy',dense1.biases)
    np.save('biases2.npy',dense2.biases)
    
    v = np.arange(Nsteps)
    
    idx0 = np.argwhere(y==0)
    idx1 = np.argwhere(y==1)
    idx2 = np.argwhere(y==2)
    
    idxp0 = np.argwhere(predictions==0)
    idxp1 = np.argwhere(predictions==1)
    idxp2 = np.argwhere(predictions==2)
    
    
    fig, ax = plt.subplots(3,1, sharex = True)
    ax[0].plot(v, 100*Monitor[:,0])
    ax[0].set_ylabel('accurac [%]')
    ax[1].plot(v, Monitor[:,1])
    ax[1].set_ylabel('loss')
    ax[2].plot(v, Monitor[:,2])
    ax[2].set_ylabel(r'$\alpha$')
    ax[2].set_xlabel('epoch')
    plt.xscale('log',base =10)
    
    plt.show()
    
    plt.scatter(x[idx0,0],x[idx0,1], color = 'black')
    plt.scatter(x[idx1,0],x[idx1,1], color = [0.2, 0, 0.8])
    plt.scatter(x[idx2,0],x[idx2,1], color = [0.7, 0.13, 0.13])
    
    plt.scatter(x[idxp0,0],x[idxp0,1], marker = 'o', edgecolors = 'black',\
                facecolors = 'none')
    plt.scatter(x[idxp1,0],x[idxp1,1], marker = 'o', \
                edgecolors = [0.2, 0, 0.8], facecolors = 'none')
    plt.scatter(x[idxp2,0],x[idxp2,1], marker = 'o', \
                edgecolors = [0.7, 0.13, 0.13], facecolors = 'none')
    















