import math
import os

from torch.nn.modules.loss import CrossEntropyLoss

CKPT_DIR = os.path.join(os.getcwd(), "best_dir")
RESULTS_DIR = os.path.join(os.getcwd(), "result_dir_1")


sweep_config = {
    'method': 'bayes',
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
            'value' : 4},
        'model': {
            'value': 'resnet'}, 
            #'values' : ['resnet', 'scratch','efficient']}, 
        'optimizer': { 
            'value': 'adabelief'},
            #'values': ['adam', 'sgd', 'adabelief']},
        'warm_up':{
            #'value': 'no'}, 
            'values': ['yes', 'no']},
        'seed':{
            #'value': 0},
            'values': [0, 3407]},
        'learning_rate': {
            'value': 0.005},
        'loss':{
            'value': 'focal'}, 
            #'values': ['focal',  'CrossEntropy']},
            #'values': ['focal',  'CrossEntropy', 'LovaszHinge']},
        },
        
    'early_terminate':{
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27,
        },
    }