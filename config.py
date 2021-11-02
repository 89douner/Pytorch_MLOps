import math

sweep_config = {
    'method': 'bayes',
    'name':'bayes-30-sweep',
    'metric' : {
        'name': 'best_acc',
        'goal': 'maximize'   
        },
    'parameters' : {
        'epochs': {
            'value' : 30
        },
        'batch_size': {
            'value' : 20
        },
        'model': {
            'values' : ['resnet', 'scratch']  
        },
        'optimizer': {
            'values': ['adam', 'sgd', 'adabelief']
            },
        'warm_up':{
            'values': ['yes', 'no']
        },
        'seed':{
            'values': [0, 3407]
        },
        'learning_rate': {
            'values': [0.001, 0.005]
            },
    },
    'early_terminate':{
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27,
        },
    }