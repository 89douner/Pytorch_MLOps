import math

sweep_config = {
    'method': 'bayes',
    'name':'bayes-c32-albu-best_acc',
    'metric' : {
        'name': 'best_acc',
        'goal': 'maximize'   
        },
    'parameters' : {
        'epochs': {
            'value' : 30},
        'batch_size': {
            'value' : 20},
        'model': {
            'value': 'resnet'}, #'values' : ['resnet', 'scratch'] 
        'optimizer': { 
            'value': 'adabelief'},#'values': ['adam', 'sgd', 'adabelief']
        'warm_up':{
            'value': 'yes'}, #'values': ['yes', 'no']
        'seed':{
            'value': 3407},#'values': [0, 3407]
        'learning_rate': {
            'value': 0.001},#'values': [0.001, 0.005]
        'shift':{
            'values': ['yes', 'no']},
        'rotate':{
            'values': ['yes', 'no']},
        'contrast':{
            'values': ['yes', 'no']},
        'distortion':{
            'values': ['yes', 'no']},
        'noise':{
            'values': ['yes', 'no']},

    },
    'early_terminate':{
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27,
        },
    }