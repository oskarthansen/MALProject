# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:19:39 2020

@author: valde
"""

def build_keras_base(hidden_layers = [64, 64, 64], dropout_rate = 0, 
                     l2_penalty = 0.1, optimizer = 'adam',
                     n_input = 100, n_class = 2):
 
    from keras.models import Sequential
    from keras.layers.normalization import BatchNormalization
    from keras.layers.core import Dense, Dropout, Activation
    from keras.layers.advanced_activations import PReLU
    model = Sequential()   
    for index, layers in enumerate(hidden_layers):       
        if not index:
            # specify the input_dim to be the number of features for the first layer
            model.add(Dense(layers, input_dim = n_input))
        else:
            model.add(Dense(layers))
        
        # insert BatchNorm layer immediately after fully connected layers
        # and before activation layer
        model.add(BatchNormalization())
        model.add(PReLU())        
        if dropout_rate:
            model.add(Dropout(p = dropout_rate))
    
    model.add(Dense(n_class))
    model.add(Activation('sigmoid'))
    
    # the loss for binary and muti-class classification is different 
    loss = 'binary_crossentropy'
    if n_class > 2:
        loss = 'categorical_crossentropy'
    
    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])   
    return model


