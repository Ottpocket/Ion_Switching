from time import time
from data_wrangling import Data_Wrangling
from cv import run_cv_model_by_batch

###############################################################################
#args: the arguments needed from the data_wrangling, model building, and 
        #crossvalidation functions to work.  Designed to enable easy hyperparameter
        #tuning.
    #Lag: (list of ints) 'signals' from n timestamps prior to current timestamp
    #Lead:(list of ints) 'signals' from n timestamps after to current timestamp
    #Diff: (Boolean) Do you subtract the present signal from lag and lead signals?
    #Rfc: (Boolean) Do you include random forest probabilites as features?
    #NN: (Boolean) Do you include neural network probabilites as features?
    #GROUP_BATCH_SIZE: (int divisible by 100,000) the size of each adjusted 
        #example for the classifier
    #Tallest: (int) Is the current signal the tallest of its n previous and 
        #posterior timestamps?
    #Rnn: use 3 rnn layers in the classifier
    #Multitask: (list of ints) predicting the outcomes of different timestamps
        #as a regularizer.  If you do not want to include, make the argument an
        #empty list.
    #Multi_weights: (float) the weight given for each multitask prediction
    #Activation_penalty: applies and activation penalty, a la Hariharan & Girshick, 2017
    #LR: (float) the learning rate
    #Epochs: (int) the number of epochs used in training
    #Minibatch_Size: (int) the size of the minibatches for the classifier
    #Seed: (int) the random seed
    #Folds: (int) the folds used in the cross validation
args = {'Lag': [1,2,3],
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
             'Epochs':180,
             'Minibatch_Size': 16,
             'Seed':321,
             'Folds':5}
def main():
    start = time()
    print('Reading Data Started...')
    train, test, features, sample_submission = Data_Wrangling(args)
    print(f'Data step finished: {time() - start} seconds')

    print('Columns of Train: {}'.format(train.columns))
    print('Columns of Test: {}'.format(test.columns))
        
    
    print('Training Wavenet model with {} folds of GroupKFold Started...'.format(args['Folds']))
    run_cv_model_by_batch(args, train, test, args['Folds'], 'group', features, sample_submission, args['Epochs'], args['Minibatch_Size'])
    print(f'Total time: {time() - start} seconds')   

main()