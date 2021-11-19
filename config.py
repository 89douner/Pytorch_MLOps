import math
import os

CKPT_DIR = os.path.join(os.getcwd(), "checkpoints_dir")
RESULTS_DIR = os.path.join(os.getcwd(), "result_dir")

sweep_config = {
    'method': 'bayes',
    'name':'bayes-c50-disease-best_acc_v3',
    'metric' : {
        'name': 'best_acc',
        'goal': 'maximize'   
        },
    'parameters' : {
        'epochs': {
            'value' : 30},
        'batch_size': {
            'value' : 25},
        'model': {
            'value': 'resnet'}, #'values' : ['resnet', 'scratch'] 
        'optimizer': { 
            'value': 'adabelief'},#'values': ['adam', 'sgd', 'adabelief']
        'warm_up':{
            'value': 'no'}, #'values': ['yes', 'no']
        'seed':{
            'value': 0},#'values': [0, 3407]
        'learning_rate': {
            'value': 0.005},#'values': [0.001, 0.005]
        'blur':{
            'values': [1, 3, 5, 7, 9]
        },
        'brightness':{
            'distribution': 'normal',
            'mu': 0.1,
            'sigma': 0.005,
        },
        'contrast':{
            'distribution': 'normal',
            'mu': 0.2,
            'sigma': 0.01,
        },
        'noise':{
            'distribution': 'normal',
            'mu': 0.005,
            'sigma': 0.0001,
        },
        'shift':{
            'distribution': 'normal',
            'mu': 0.0625,
            'sigma': 0.001,
        },
        'rotate':{
            'distribution': 'normal',
            'mu': 10,
            'sigma': 1,
        },
        'distortion':{
            'distribution': 'normal',
            'mu': 0.05,
            'sigma': 0.001,
        },

    },
    'early_terminate':{
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27,
        },
    }