import math

hyperparameter_defaults  = {
        'epochs': 30,
        'batch_size': 3,
        #'fc_layer_size': 128,
        #'weight_decay': 0.0005,
        #'learning_rate': 1e-3,
        #'activation': 'relu',
        #'optimizer': 'adam',
        #'seed': 42
    }

sweep_config = {
    'method': 'bayes',
    'project': "test", 
    'entity': 'douner89',
    'metric' : {
        'name': 'val_epoch_loss',
        'goal': 'minimize'   
        },
    'parameters' : {
        'model': {
            'values' : ['resnet', 'effnet']  #'value' : 'resnet', ['resnet', 'scratch', 'effnet'] 
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
        ##여러 줄 주석 추가: ctrl+K+C 동시에 누르기
        ##여러줄 주석 해제: ctrl+K+U 동시에 누르기
        # 'dropout': {
        #     'values': [0.3, 0.4, 0.5]
        #     },
        'learning_rate': {
            'values': [0.001, 0.005]
            },
        #'batch_size': {
        #    'distribution': 'q_log_uniform',
        #    'q': 1,
        #    'min': math.log(16),
        #    'max': math.log(32),
        #    },
        # 'data_augmentation1': {
        #     'values': ['brightness', 'no_aug']
        # },
        # 'data_augmentation2': {
        #     'values': ['contrast', 'no_aug']
        # },
    },
    'early_terminate':{
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27,
        },
    }