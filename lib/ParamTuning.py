# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 09:19:39 2020

@author: valde
"""

def build_keras_model(hidden_layers = [64, 64, 64], dropout_rate = [], 
                     l2_penalty = 0.1, optimizer= "adam",
                     n_input = 100, n_class = 1, default_dropout=0.2, metrics=["accuracy"]):
 
    from keras.models import Sequential
    from keras.layers.normalization import BatchNormalization
    from keras.layers.core import Dense, Dropout, Activation
    model = Sequential()   
    for index, layer in enumerate(hidden_layers):       
        if not index:
            model.add(Dense(layer, input_dim = n_input, kernel_initializer="he_normal", activation="elu"))
        else:
            model.add(Dense(layer, kernel_initializer="he_normal", activation="elu"))
        
        model.add(BatchNormalization())    
        if dropout_rate and index < len(dropout_rate):
            model.add(Dropout(rate = dropout_rate[index]))
        else:
            model.add(Dropout(rate = default_dropout))
            
    model.add(Dense(n_class))
    model.add(Activation('sigmoid'))
    
    loss = 'mean_squared_error'
    model.compile(loss = loss, optimizer=optimizer, metrics=metrics)   
    return model


