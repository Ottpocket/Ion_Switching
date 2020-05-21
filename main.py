# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:10:30 2020

@author: andre
"""

def main(name = None):
    if name is None:
        print('Give a name please!')
        return
    data_args = {'lag': [1,2,3],
            'lead':[1,2,3],
            'diff': True,
            'RFC': True,
            'GROUP_BATCH_SIZE': 50000,
            'tallest': 2,
            'lowest': 2}
    train, test, features = data_wrangling(args = data_args)
    cv_args = {'multitask':False,
               'LSTM':0,
               'size':1,
               'activation_penalty':False,
               'LR': .0015}
    cv_validation(args = cv_args, model_name = name)
    pass

main()