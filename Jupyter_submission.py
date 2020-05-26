import osargs = {'Lag': [1,2,3],
             'Lead':[1,2,3],
             'Diff': True,
             'Rfc': True,
             'NN':False,
             'GROUP_BATCH_SIZE': 4000,
             'Tallest': 2,
             'Lowest': 2,
             'Rnn': False,
             'Multitask': [1,2,3],
             'Multi_Weights': .05,
             'Activation_penalty': False,
             'LR': .0015,
             'Wn':1,
             'Epochs':180,
             'Minibatch_Size': 16,
             'Seed':321,
             'Folds':5}
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

def read_data(RFC = True, NN = False):
    train = pd.read_csv('/kaggle/input/data-without-drift/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
    test  = pd.read_csv('/kaggle/input/data-without-drift/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})
    sub  = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv', dtype={'time': np.float32})
    if RFC == True:
        Y_train_proba = np.load("/kaggle/input/ion-shifted-rfc-proba/Y_train_proba.npy")
        Y_test_proba = np.load("/kaggle/input/ion-shifted-rfc-proba/Y_test_proba.npy")        
        for i in range(11):
            train[f"RF_proba_{i}"] = Y_train_proba[:, i]
            test[f"RF_proba_{i}"] = Y_test_proba[:, i]
    if NN == True:
        ytr_load = np.load('/kaggle/input/ottpocket-edits/Train_probs.npz')
        Y_train_proba = ytr_load['train_probs']
        yte_load = np.load('/kaggle/input/ottpocket-edits/Test_probs.npz')
        Y_test_proba = yte_load['test_probs']
        for i in range(11):
            train[f"NN_proba_{i}"] = Y_train_proba[:, i]
            test[f"NN_proba_{i}"] = Y_test_proba[:, i]
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
    train, test, sample_submission = read_data(RFC = args['Rfc'], NN = args['NN'])
    train, test = normalize(train, test)
        
    print('Creating Features')
    print('Feature Engineering Started...')
    train = run_feat_engineering(train,  args=args)
    test = run_feat_engineering(test, args=args)
    train, test, features = feature_selection(train, test)
    print('Feature Engineering Completed...')
    return (train, test, features, sample_submission)
        
    
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

    inp = Input(shape = shape_)
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
    
    fork = cbr(x, 32, 7, 1, 1)
    if args['Rnn']==True:
        fork = Bidirectional(LSTM(64, return_sequences=True))(fork)
        fork = Bidirectional(LSTM(64, return_sequences=True))(fork)
        fork = Bidirectional(LSTM(64, return_sequences =True))(fork)
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


def lr_schedule(epoch):
    LR = args['LR']
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
def run_cv_model_by_batch(train, test, splits, batch_col, feats, sample_submission, nn_epochs, nn_batch_size):
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
    kf = GroupKFold(n_splits=args['Folds'])
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
    
    #FileLink(r'df.npz')
    #np.savetxt('Val_Probs.gz', oof_, delimiter=',')
    #np.savetxt('Preds.gz', preds_, delimiter=',')
# this function run our entire program
def run_everything():
    start = time()
    print('Reading Data Started...')
    train, test, features, sample_submission = Data_Wrangling(args)
    print(f'Data step finished: {time() - start} seconds')

    print('Columns of Train: {}'.format(train.columns))
    print('Columns of Test: {}'.format(test.columns))
        
    
    print('Training Wavenet model with {} folds of GroupKFold Started...'.format(args['Folds']))
    run_cv_model_by_batch(train, test, args['Folds'], 'group', features, sample_submission, args['Epochs'], args['Minibatch_Size'])
    print(f'Total time: {time() - start} seconds')   
run_everything()