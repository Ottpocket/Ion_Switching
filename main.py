# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:10:30 2020

@author: andre
"""

#EPOCHS = 60 #180 
#NNBATCHSIZE = 16
#GROUP_BATCH_SIZE = 50000#4000
#SEED = 321
#LR = 0.0015
#SPLITS = 5 #
import os
os.chdir('C:/Users/andre/Documents/Github/Ion_Switching')
from data_wrangling import Data_Wrangling


def main(name = None):
    if name is None:
        print('Give a name please!')
        return
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
             'LR': .0015,
             'Wn':1,
             'Epochs':5,
             'Minibatch_Size': 16,
             'Seed':321}
    
    cv_validation(args = cv_args, model_name = name)
    pass

main()