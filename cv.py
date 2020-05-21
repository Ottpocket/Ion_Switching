# -*- coding: utf-8 -*-
"""
Created on Thu May 21 07:04:10 2020

@author: andre
"""

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

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

EPOCHS = 60 #180 
NNBATCHSIZE = 16
GROUP_BATCH_SIZE = 50000#4000
SEED = 321
LR = 0.0015
SPLITS = 5 #

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

# model function (very important, you can try different arquitectures to get a better score. I believe that top public leaderboard is a 1D Conv + RNN style)
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
    
    opt = Adam(lr = LR)
    #opt = tfa.optimizers.SWA(opt) #Figure out how to get this running!
    model.compile(loss = losses.CategoricalCrossentropy(), optimizer = opt, metrics = ['accuracy'])
    return model

# function that decrease the learning as epochs increase (i also change this part of the code)
def lr_schedule(epoch):
    if epoch < 30:
        lr = LR
    elif epoch < 40:
        lr = LR / 3
    elif epoch < 50:
        lr = LR / 5
    elif epoch < 60:
        lr = LR / 7
    elif epoch < 70:
        lr = LR / 9
    elif epoch < 80:
        lr = LR / 11
    elif epoch < 90:
        lr = LR / 13
    else:
        lr = LR / 100
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

# main function to perfrom groupkfold cross validation (we have 1000 vectores of 4000 rows and 8 features (columns)). Going to make 5 groups with this subgroups.
###############################################################################
#run_cv_model_by_batch: runs the cv and prints the results to kaggle
#INPUTS:
    #train:
    #test:
    #feats: the features used for training. 'features' from data_wrangling
    #sample_submission: the regular sample submissiong from Ion-Transfer
    #name: the name of the model used. 
    #nn_epochs: how many epochs the nn will run
    #nn_batch_size: minibatch size
    #batch_col: the column useds to split the data into batched. 'group'
    #splits: The number of Folds for the cross val
#OUTPUTS:
    #
###############################################################################
def run_cv_model_by_batch(train, test,  feats, sample_submission, name, nn_epochs = 60, 
                          nn_batch_size=16, batch_col='groups', splits=5):
    
    name = '{}_{}_{}'.format(name, nn_epochs, nn_batch_size)
    training_time = time()
    seed_everything(SEED)
    K.clear_session()
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    oof_ = np.zeros((len(train), 11)) # build out of folds matrix with 11 columns, they represent our target variables classes (from 0 to 10)
    preds_ = np.zeros((len(test), 11))
    target = ['open_channels']
    group = train['group']
    kf = GroupKFold(n_splits=5)
    splits = [x for x in kf.split(train, train[target], group)]

    new_splits = []
    for sp in splits:
        new_split = []
        new_split.append(np.unique(group[sp[0]]))
        new_split.append(np.unique(group[sp[1]]))
        new_split.append(sp[1])    
        new_splits.append(new_split)
    # pivot target columns to transform the net to a multiclass classification estructure (you can also leave it in 1 vector with sparsecategoricalcrossentropy loss function)
    tr = pd.concat([pd.get_dummies(train.open_channels), train[['group']]], axis=1)

    tr.columns = ['target_'+str(i) for i in range(11)] + ['group']
    target_cols = ['target_'+str(i) for i in range(11)]
    train_tr = np.array(list(tr.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)
    train = np.array(list(train.groupby('group').apply(lambda x: x[feats].values)))
    test = np.array(list(test.groupby('group').apply(lambda x: x[feats].values)))
    Training_df = []
    for n_fold, (tr_idx, val_idx, val_orig_idx) in enumerate(new_splits[0:], start=0):
        train_x, train_y = train[tr_idx], train_tr[tr_idx]
        valid_x, valid_y = train[val_idx], train_tr[val_idx]
        print(f'Our training dataset shape is {train_x.shape}')
        print(f'Our validation dataset shape is {valid_x.shape}')

        gc.collect()
        shape_ = (None, train_x.shape[2]) # input is going to be the number of feature we are using (dimension 2 of 0, 1, 2)
        model = Classifier(shape_)
        # using our lr_schedule function
        cb_lr_schedule = LearningRateScheduler(lr_schedule)
        H = model.fit(train_x,train_y,
                  epochs = nn_epochs,
                  callbacks = [cb_lr_schedule, MacroF1(model, train_x, train_y, valid_x, valid_y)], # adding custom evaluation metric for each epoch
                  batch_size = nn_batch_size,verbose = 2,
                  validation_data = (valid_x,valid_y))
        preds_f = model.predict(valid_x)
        #f1_score_ = f1_score(np.argmax(valid_y, axis=2).reshape(-1),  np.argmax(preds_f, axis=2).reshape(-1), average = 'macro') # need to get the class with the biggest probability
        print('Training fold {} completed. macro f1 score : {:1.5f}'.format(n_fold+1,H.history['F1_val'][-1] ))
        preds_f = preds_f.reshape(-1, preds_f.shape[-1])
        oof_[val_orig_idx,:] += preds_f
        te_preds = model.predict(test)
        model.save("model-wavenet_fold{}.h5".format(n_fold+1))
        te_preds = te_preds.reshape(-1, te_preds.shape[-1])           
        preds_ += te_preds / SPLITS
        
        #Creating a dataframe of the training dynamics of this fold
        df = pd.DataFrame.from_dict(H.history)
        df['Fold'] = [n_fold]*df.shape[0]
        Training_df.append(df)
        
        #Getting some space in memory
        del([model, train_x, train_y, valid_x, valid_y])
        gc.collect()
        
    print('Training completed...')
    print(f'Training time: {time() - training_time}')
    # calculate the oof macro f1_score
    print('Collection final submissions...')
    f1_score_ = f1_score(np.argmax(train_tr, axis = 2).reshape(-1),  np.argmax(oof_, axis = 1), average = 'macro') # axis 2 for the 3 Dimension array and axis 1 for the 2 Domension Array (extracting the best class)
    print(f'Training completed. oof macro f1 score : {f1_score_:1.5f}')
    sample_submission['open_channels'] = np.argmax(preds_, axis = 1).astype(int)
    sample_submission.to_csv('{}.csv'.format(name), index=False, float_format='%.4f')
    
    #create the datafrane for graphing training dynamics
    Training_dynamics = pd.concat(Training_df)
    Training_dynamics.to_csv('Training_by_Epoch_{}.csv'.format(name), index=False)
    
    
    #Reducing the data footprint, compressing, and saving softmax probs 
    # of val and test data as numpy compressed files
    save_start = time()
    oof_ = oof_.astype(np.float16)
    preds_ = preds_.astype(np.float16)
    #Saving the validation predictions and test predictions for a stacknet
    print('Saving Validation Probs and test Probs to npz')
    np.savez_compressed("Train_probs_{}.npz".format(name), train_probs=oof_)
    np.savez_compressed("Test_probs_{}.npz".format(name), test_probs=preds_)
    print('Done Saving.  Took {} seconds'.format(time() - save_start))
 