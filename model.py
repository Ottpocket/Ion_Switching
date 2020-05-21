import os
#os.system('ls -l')
#os.system('pip install tensorflow_addons')
import tensorflow as tf
from tensorflow.keras.layers import *
import pandas as pd
import numpy as np
import random
from IPython.display import FileLink
from time import time
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import losses, models, optimizers
#import tensorflow_addons as tfa #Figure out how to get this working!
import gc

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
###############################################################################
#Classifier: The nn classifier
#INPUTS:
    #shape_: automatically done in cv function
    #args: dictionary with the following keys
        #rnn: (int) # of rnn layers at the end. -1 if none used
        #multitask: (int) predict the signal for the point n timestamps ahead
        #activation_penalty: (Boolean) include a penalty for activations on last 
               #non_softmax layer.  
        #LR: the learning rate.
        #wn:architecture: (int) arch of wavenet section.
            #1 is standard, 
            #2 is mini
            #3 is ...
        
###############################################################################
def Classifier(shape_):    
    def cbr(x, out_layer, kernel, stride, dilation):
        x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x
    
    def wave_block(x, filters, kernel_size, n):
        dilation_rates = [2**i for i in range(n)]
        x = Conv1D(filters = filters,
                   kernel_size = 1,
                   padding = 'same')(x)
        res_x = x
        for dilation_rate in dilation_rates:
            tanh_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same', 
                              activation = 'tanh', 
                              dilation_rate = dilation_rate)(x)
            sigm_out = Conv1D(filters = filters,
                              kernel_size = kernel_size,
                              padding = 'same',
                              activation = 'sigmoid', 
                              dilation_rate = dilation_rate)(x)
            x = Multiply()([tanh_out, sigm_out])
            x = Conv1D(filters = filters,
                       kernel_size = 1,
                       padding = 'same')(x)
            res_x = Add()([res_x, x])
        return res_x
    
    inp = Input(shape = (shape_))
    x = cbr(inp, 64, 7, 1, 1)
    #Commented for faster prototyping.  Get rid of comments when actually submitting code
    
    x = BatchNormalization()(x)
    x = wave_block(x, 16, 3, 12)
    
    x = BatchNormalization()(x)
    x = wave_block(x, 32, 3, 8)
    x = BatchNormalization()(x)
    x = wave_block(x, 64, 3, 4)
    x = BatchNormalization()(x)
    x = wave_block(x, 128, 3, 1)
    x = cbr(x, 32, 7, 1, 1)
    x = BatchNormalization()(x)
    x = wave_block(x, 64, 3, 1)
    x = cbr(x, 32, 7, 1, 1)
    
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    out = Dense(11, activation = 'softmax', name = 'out')(x)
    
    model = models.Model(inputs = inp, outputs = out)
    
    opt = Adam(lr = args['LR'])
    model.compile(loss = losses.CategoricalCrossentropy(), optimizer = opt, metrics = ['accuracy'])
    return model

# function that decrease the learning as epochs increase (i also change this part of the code)
def lr_schedule(epoch):
    if epoch < 30:
        lr = args['LR']
    elif epoch < 40:
        lr = args['LR'] / 3
    elif epoch < 50:
        lr = args['LR'] / 5
    elif epoch < 60:
        lr = args['LR'] / 7
    elif epoch < 70:
        lr = args['LR'] / 9
    elif epoch < 80:
        lr = args['LR'] / 11
    elif epoch < 90:
        lr = args['LR'] / 13
    else:
        lr = args['LR'] / 100
    return lr

# class to get macro f1 score. This is not entirely necessary but it's fun to check f1 score of each epoch (be carefull, if you use this function early stopping callback will not work)
class MacroF1(Callback):
    def __init__(self, model, inputs, targets, val_inputs, val_targets):
        self.model = model
        self.inputs = inputs
        self.targets = np.argmax(targets, axis = 2).reshape(-1)
        self.val_inputs = val_inputs
        self.val_targets = np.argmax(val_targets, axis = 2).reshape(-1)
        self.time = time()
        
    def on_epoch_end(self, epoch, logs):
        pred = np.argmax(self.model.predict(self.inputs), axis = 2).reshape(-1)
        score = f1_score(self.targets, pred, average = 'macro')
        logs['F1_tr'] = score
        pred2 = np.argmax(self.model.predict(self.val_inputs), axis = 2).reshape(-1)
        score2 = f1_score(self.val_targets, pred2, average = 'macro')
        logs['F1_val'] = score2
        logs['time'] = self.time - time()
        print(f'F1 Macro Score, Train: {score:.5f}, Val: {score2:.5f}')
        gc.collect()
