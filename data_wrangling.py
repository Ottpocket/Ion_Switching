# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:24:09 2020

@author: andre
"""
import pandas as pd
import numpy as np
import gc
# read data
def read_data(RFC = True):
    train = pd.read_csv('/kaggle/input/data-without-drift/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
    test  = pd.read_csv('/kaggle/input/data-without-drift/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})
    sub  = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv', dtype={'time': np.float32})
    if RFC == True:
        Y_train_proba = np.load("/kaggle/input/ion-shifted-rfc-proba/Y_train_proba.npy")
        Y_test_proba = np.load("/kaggle/input/ion-shifted-rfc-proba/Y_test_proba.npy")        
        for i in range(11):
            train[f"proba_{i}"] = Y_train_proba[:, i]
            test[f"proba_{i}"] = Y_test_proba[:, i]

    return train, test, sub

# create batches of 4000 observations
def batching(df, batch_size):
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df

# normalize the data (standard scaler). We can also try other scalers for a better score!
def normalize(train, test):
    train_input_mean = train.signal.mean()
    train_input_sigma = train.signal.std()
    train['signal'] = (train.signal - train_input_mean) / train_input_sigma
    test['signal'] = (test.signal - train_input_mean) / train_input_sigma
    return train, test

# get lead and lags features
def lag_with_pct_change(df, lags, leads, diff = False):
    if lags != -1:
        for lag in lags:    
            if diff == False:
                df['signal_shift_neg_' + str(lag)] = df.groupby('group')['signal'].shift(-1 * lag).fillna(0)
            else:
                df['signal_shift_neg_' + str(lag)] = df.groupby('group')['signal'].shift(-1 * lag).fillna(0) - df.groupby('group')['signal'].shift(0).fillna(0)
    if leads != -1:
        for lead in leads:    
            if diff ==False:
                df['signal_shift_pos_' + str(lead)] = df.groupby('group')['signal'].shift(lead).fillna(0)
            else:
                df['signal_shift_pos_' + str(lead)] = df.groupby('group')['signal'].shift(lead).fillna(0) -    df.groupby('group')['signal'].shift(0).fillna(0)
    return df

def Height(df, args):
    values_list = []
    if args['Tallest'] != -1:
        for i in range(1,args['Tallest']+1):
            values_list.append(df.signal.shift(i).fillna(-999).values)
            values_list.append(df.signal.shift(-1 * i).fillna(-999).values)
        nums = np.vstack(values_list)
        mins = np.min(nums, axis=0)
        maxs = np.max(nums, axis=0)
        df['Tallest']= (df.signal > maxs).astype(int)
        df['Lowest'] = (df.signal < mins).astype(int)
        
        del([nums, mins, maxs, values_list])
    gc.collect
    return df
# main module to run feature engineering. Here you may want to try and add other features and check if your score imporves :).
def run_feat_engineering(df, args):
    # create batches
    df = batching(df, batch_size = args['GROUP_BATCH_SIZE'])
    
    # create leads and lags of 'Lag' and 'Lead' from args
    df = lag_with_pct_change(df, lags = args['Lag'], leads = args['Lead'], diff = args['Diff'])

    # create signal ** 2 (this is the new feature)
    df['signal_2'] = df['signal'] ** 2
    
    df = Height(df, args)

    return df

# fillna with the mean and select features for training
def feature_selection(train, test):
    features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time']]
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    for feature in features:
        feature_mean = pd.concat([train[feature], test[feature]], axis = 0).mean()
        train[feature] = train[feature].fillna(feature_mean)
        test[feature] = test[feature].fillna(feature_mean)
    return train, test, features
###############################################################################
#data_wrangling: function that gets the data to an appropriate form to the nn.
#INPUTS:
    #args: a dictionary containing the following keys:
        #Lag: (list of ints) lagged signals to include. the ints specify future timespots. 
        #Lead: (list of ints) lead signals to include. the ints specify future timespots
        #Diff: (Boolean) lag/lead columns subtracted by the present signal
        #Rfc: (Boolean) to use RFC preds for model
        #Group_batch_size: (int) size of the time slice to be trained on
        #Height: (int) is the signal tallerorsmaller than n lag&lead timestamps? -1 if not used
#OUTPUT:
    #train
    #test
    #features: the features used in the data for training the network
###############################################################################
def Data_Wrangling(args):
    print('Reading in Data')
    train, test, sample_submission = read_data(RFC = args['Rfc'])
    train, test = normalize(train, test)
        
    print('Creating Features')
    print('Feature Engineering Started...')
    train = run_feat_engineering(train,  args=args)
    test = run_feat_engineering(test, args=args)
    train, test, features = feature_selection(train, test)
    print('Feature Engineering Completed...')
    return (train, test, features)
        