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
from time import time
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import losses, models, optimizers

import gc

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


###############################################################################
#MacroF1: gets F1 score for val and train.  Also stems memory leak in TF2.  Get
#   it together google.
#INPUTS:
    #model: the tf model
    #inputs: train data
    #targets: the correct answers
    #val_inputs: val data
    #val_targets: val correct answers
###############################################################################
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
        

def lr_schedule(epoch, args):
    LR = args['LR']
    if epoch < 15:
        lr = LR
    elif epoch < 30:
        lr = LR / 3
    elif epoch < 40:
        lr = LR / 5
    elif epoch < 50:
        lr = LR / 7
    elif epoch < 60:
        lr = LR / 9
    elif epoch < 70:
        lr = LR / 11
    elif epoch < 80:
        lr = LR / 13
    else:
        lr = LR / 100
    return lr

# main function to perfrom groupkfold cross validation (we have 1000 vectores of 4000 rows and 8 features (columns)). Going to make 5 groups with this subgroups.
###############################################################################
#run_cv_model_by_batch: runs the cv and prints the results to kaggle.  The 
    #predictions are an ensembled from each of the models trained on the 
    #K folds.  
#INPUTS:
    #args: (dict) huge list of hyperparameters and model specifications
    #train:
    #test:
    #folds: (int) the number of folds for the cross val
    #feats:(list of strings) columns of train and test to be used for classification
    #sample_submission:
    #nn_epochs: the number of epochs for NN training
    #nn_batch_size: Batch size.  
#OUTPUTS:
    #submission_wavenet.csv: the submission sent to kaggle
    #Train_probs.npz: the probabilities for each train item as given by the 
                    #the trained NN.  Used for training the StackNet.
    #Test_probs.npz: the probabilites outputted by the NN for each test item.
    #model-wavenet_fold{}.h5: a trained model after the completion of the fold.
###############################################################################
def run_cv_model_by_batch(args, train, test, folds, batch_col, feats, sample_submission, nn_epochs, nn_batch_size):
    training_time = time()
    seed_everything(args['Seed'])
    K.clear_session()
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    oof_ = np.zeros((len(train), 11)) # build out of folds matrix with 11 columns, they represent our target variables classes (from 0 to 10)
    preds_ = np.zeros((len(test), 11))
    target = ['open_channels']
    group = train['group']
    kf = GroupKFold(folds=args['Folds'])
    splits = [x for x in kf.split(train, train[target], group)]

    
    new_splits = []
    for sp in splits:
        new_split = []
        new_split.append(np.unique(group[sp[0]]))
        new_split.append(np.unique(group[sp[1]]))
        new_split.append(sp[1])    
        new_splits.append(new_split)
    # pivot target columns to transform the net to a multiclass classification estructure (you can also leave it in 1 vector with sparsecategoricalcrossentropy loss function)
   #Getting the list of correct channels for the predictions
    train_tr_list = []
    tr = pd.concat([pd.get_dummies(train.open_channels), train[['group']]], axis=1)
    tr.columns = ['target_'+str(i) for i in range(11)] + ['group']
    target_cols = ['target_'+str(i) for i in range(11)]
    train_tr = np.array(list(tr.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)
    train_tr_list.append(train_tr)
    del train_tr
    #Getting the list of correct channels for the multitask predictions
    for shift_ in args['Multitask']:
        #Shifting the predictions by the correct ammount
        tr_copy = tr.copy()
        tr_copy[target_cols] = tr_copy.loc[:,target_cols].shift(shift_).fillna(0)
        train_tr = np.array(list(tr_copy.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)
        train_tr_list.append(train_tr)
        del train_tr
        gc.collect()
    
    start = time()
    for i in range(len(train_tr_list)):
        np.savez_compressed('train_tr_{}'.format(i), a=train_tr_list[i])
    print(f'Took {time() - start} to clock')
    del train_tr_list
    gc.collect()
    train_tr  = [np.load('/kaggle/working/train_tr_{}.npz'.format(i)) for i in range(4)]#The compressed targets!
    
    train = np.array(list(train.groupby('group').apply(lambda x: x[feats].values)))
    test = np.array(list(test.groupby('group').apply(lambda x: x[feats].values)))
    Training_df = []
    for n_fold, (tr_idx, val_idx, val_orig_idx) in enumerate(new_splits[0:], start=0):
        train_x  = train[tr_idx]
        train_y  = [train_tr[i]['a'][tr_idx] for i in range(len(train_tr))]
        valid_x  = train[val_idx]
        valid_y  = [train_tr[i]['a'][val_idx] for i in range(len(train_tr))]
        print(f'Our training dataset shape is {train_x.shape}')
        print(f'Our validation dataset shape is {valid_x.shape}')
    
        gc.collect()
        shape_ = (None, train_x.shape[2]) # input is going to be the number of feature we are using (dimension 2 of 0, 1, 2)
        model = Classifier(shape_, args)
        # using our lr_schedule function
        cb_lr_schedule = LearningRateScheduler(lr_schedule)
        H = model.fit(train_x,train_y,
                  epochs = nn_epochs,
                  callbacks = [cb_lr_schedule, MacroF1(model, train_x, train_y, valid_x, valid_y)], # adding custom evaluation metric for each epoch
                  batch_size = nn_batch_size,verbose = 2,
                  validation_data = (valid_x,valid_y))
        preds_f = model.predict(valid_x)
        preds_f = preds_f[0]
        #f1_score_ = f1_score(np.argmax(valid_y, axis=2).reshape(-1),  np.argmax(preds_f, axis=2).reshape(-1), average = 'macro') # need to get the class with the biggest probability
        print('Training fold {} completed. macro f1 score : {:1.5f}'.format(n_fold+1,H.history['F1_val'][-1] ))
        preds_f = preds_f.reshape(-1, preds_f.shape[-1])
        oof_[val_orig_idx,:] += preds_f
        te_preds = model.predict(test)
        te_preds = te_preds[0]
        model.save("model-wavenet_fold{}.h5".format(n_fold+1))
        te_preds = te_preds.reshape(-1, te_preds.shape[-1])           
        preds_ += te_preds / args['Folds']
        
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
    f1_score_ = f1_score(np.argmax(train_tr[0]['a'], axis = 2).reshape(-1),  np.argmax(oof_, axis = 1), average = 'macro') # axis 2 for the 3 Dimension array and axis 1 for the 2 Domension Array (extracting the best class)
    print(f'Training completed. oof macro f1 score : {f1_score_:1.5f}')
    sample_submission['open_channels'] = np.argmax(preds_, axis = 1).astype(int)
    sample_submission.to_csv('submission_wavenet.csv', index=False, float_format='%.4f')
    
    #create the datafrane for graphing training dynamics
    Training_dynamics = pd.concat(Training_df)
    Training_dynamics.to_csv('Training_by_Epoch.csv', index=False)
    
    
    #Reducing the data footprint, compressing, and saving softmax probs 
    # of val and test data as numpy compressed files
    save_start = time()
    oof_ = oof_.astype(np.float16)
    preds_ = preds_.astype(np.float16)
    #Saving the validation predictions and test predictions for a stacknet
    print('Saving Validation Probs and test Probs to npz')
    np.savez_compressed("Train_probs.npz", train_probs=oof_)
    np.savez_compressed("Test_probs.npz", test_probs=preds_)
    print('Done Saving.  Took {} seconds'.format(time() - save_start)) 