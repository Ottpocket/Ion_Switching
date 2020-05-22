import os
#os.system('ls -l')
#os.system('pip install tensorflow_addons')
import tensorflow as tf
from tensorflow.keras.layers import *
import pandas as pd
from IPython.display import FileLink
from time import time
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import losses, models, optimizers
#import tensorflow_addons as tfa #Figure out how to get this working!
import gc


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
        #Rnn: (int) # of rnn layers at the end. -1 if none used
        #Multitask: (list of ints) predict the channels for the point n timestamps 
                #ahead.  If an int is negative, it predicts a previous channel.
                #Empty list if no predictions desired
        #Multi_Weights: (int) the weight given to a multitask loss.  
        #Activation_penalty: (Boolean) include a penalty for activations on last 
               #non_softmax layer.  
        #LR: the learning rate.
        #Wn:architecture: (int) arch of wavenet section.
            #1 is standard, 
            #2 is mini
            #3 is ...
        
###############################################################################
def Classifier(shape_, args):    
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
    #Returns a list of convolution softmax heads depending on the number of 
    #multitask predictions desired
    def Multitask_Head(fork, num_preds):
        if num_preds == 0:
            return []
        heads = []
        for i in range(num_preds):
            pred = cbr(fork, 32, 7, 1, 1)
            pred = BatchNormalization()(pred)
            pred = Dropout(0.2)(pred)
            pred = Dense(11, activation = 'softmax', name = 'multout_{}'.format(i+1))(pred)
            heads.append(pred)
        return heads
    #Returns the weights of the heads for the classifier. multi_weight is the 
    # weight given to each multitask prediction.  
    def Get_Weights(num_losses, multi_weight):
        if num_losses ==1:
            return [1.]
        else:
            return [1. - multi_weight*(num_losses-1)] +[multi_weight for i in range(num_losses - 1 )]

    inp = Input(shape = (shape_))
    x = cbr(inp, 64, 7, 1, 1)
    #Commented for faster prototyping.  Get rid of comments when actually submitting code
    '''
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
    '''
    fork = cbr(x, 32, 7, 1, 1)
    multitask_list = Multitask_Head(fork, len(args['Multitask']))
    x = BatchNormalization()(fork)
    x = Dropout(0.2)(x)
    out = Dense(11, activation = 'softmax', name = 'out')(x)
    outputs = [out] + multitask_list
    model = models.Model(inputs = inp, outputs = outputs)
    
    opt = Adam(lr = args['LR'])
    losses_ = [losses.CategoricalCrossentropy() for i in range(len(outputs))]
    loss_weights_ = Get_Weights(len(losses_), args['Multi_Weights'])
    model.compile(loss = losses_, optimizer = opt, metrics = ['accuracy'],
                  loss_weights = loss_weights_)
    return model


'''
args = {'Lag': [1],
             'Lead':[1,2,3,4],
             'Diff': True,
             'Rfc': True,
             'GROUP_BATCH_SIZE': 50000,
             'Tallest': 2,
             'Lowest': 2,
             'Rnn': False,
             'Multitask': [1,2,-1],
             'Multi_Weights': .05,
             'Activation_penalty': False,
             'LR': .015,
             'Wn':1,
             'Epochs':25,
             'Minibatch_Size': 16}
model = Classifier([None, 22], args)
tf.keras.utils.plot_model(model)
'''