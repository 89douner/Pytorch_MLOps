import math
import os

from torch.nn.modules.loss import CrossEntropyLoss

CKPT_DIR = os.path.join(os.getcwd(), "best_dir")
RESULTS_DIR = os.path.join(os.getcwd(), "result_dir_1")

sweep_config = {
    'method': 'grid',
    'name':'grid-augmentation-test',
    'metric' : {
        'name': 'best_acc',
        'goal': 'maximize'   
        },

    ####Basic Hyper-parameters#####
    'parameters' : {
        'epochs': {
            'value' : 50},
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
            'values': 0.005},
        'loss':{
            'values': ['CrossEntropy'] # 'values': ['focal',  'CrossEntropy', 'LovaszHinge']},
        },
            
        #####Data Augmentation Hyper-parameters##########
        'blur':{
            'values': [0, 1, 2, 3] #3
        },
        'brightness':{
            'values': [0.0, 0.1, 0.2] #0.1
        },
        'contrast':{
            'values': [0.0, 0.1, 0.2] #0.2
        },
        'noise':{
            'values': [0.0, 0.005, 0.01] #0.005
        },
        'shift':{
            'values': [0.0, 0.0625, 0.1] #0.0625
        },
        'rotate':{
            'values': [0, 5, 10] #10
        },
        'distortion':{
            'values': [0.0, 0.05, 0.1] #0.05
        },
    },
    'early_terminate':{
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27,
        },
    }