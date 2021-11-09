import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import model
import sweep_train

from dataload_with_alb  import DiseaseDataset
from dataload_with_alb_gpu  import GpuDataset

from torch.utils.data import DataLoader

from warmup_scheduler import GradualWarmupScheduler
from adabelief_pytorch import AdaBelief

import wandb
import config



def wandb_setting(sweep_config=None):
    wandb.init(config=sweep_config)
    w_config = wandb.config
    name_str = 's:' +  str(w_config.shift) + ' -r:' + str(w_config.rotate) + ' -c:' + str(w_config.contrast) + ' -d:' + str(w_config.distortion) + ' -n:' + str(w_config.noise)
    wandb.run.name = name_str

    #########Random seed 고정해주기###########
    random_seed = w_config.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    ###########################################

    batch_size= w_config.batch_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ##########################################데이터 로드 하기#################################################
    data_dir = os.path.join(os.getcwd(), "RSNA_COVID_png_512") #train, val 폴더가 들어있는 경로
    classes_name = os.listdir(os.path.join(data_dir, 'train')) #폴더에 들어있는 클래스명
    num_classes =  len(os.listdir(os.path.join(data_dir, 'train'))) #train 폴더 안에 클래스 개수 만큼의 폴더가 있음

    datasets = {x: GpuDataset(data_dir=os.path.join(data_dir, x), img_size=512, bit=8, 
                num_classes=num_classes, classes_name=classes_name, data_type='img', mode= x, w_config=w_config) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    num_iteration = {x: np.ceil(dataset_sizes[x] / batch_size) for x in ['train', 'val']}
    #############################################################################################################################

    if w_config.model == 'resnet':
        net = model.Pre_Resnet50(img_channel=1, num_classes=num_classes) # pretrained Resnet101 모델 사용
    elif w_config.model == 'scratch':
        net = model.ResNet50(img_channel=1, num_classes=num_classes) #gray scale = 1, color scale =3
    elif w_config.model == 'effnet':
        net = model.Efficient(img_channel=1, num_classes=num_classes) # pretrained Efficient 모델 사용

    net = net.to(device) #딥러닝 모델 GPU 업로드

    criterion = nn.CrossEntropyLoss() #loss 형태 정해주기

    if w_config.optimizer == 'sgd':
        optimizer_ft = optim.SGD(net.parameters(), lr=w_config.learning_rate, momentum=0.9)# optimizer 종류 정해주기
    elif w_config.optimizer == 'adam':
        optimizer_ft = optim.Adam(net.parameters(), lr=w_config.learning_rate)
    elif w_config.optimizer == 'adabelief':
        optimizer_ft = AdaBelief(net.parameters(), lr=1e-3, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = True)


    ############Learning rate scheduler: Warm-up with ReduceLROnPlateau#################
    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_ft, mode='min', factor=0.5, patience=10)

    if w_config.warm_up == 'yes':
        scheduler_lr = GradualWarmupScheduler(optimizer_ft, multiplier=1, total_epoch=5, after_scheduler=scheduler_lr)

    ########################################################################################
    patience = 6
    wandb.watch(net, log='all') #wandb에 남길 log 기록하기
    sweep_train.train_model(dataloaders, dataset_sizes, num_iteration, net, criterion, optimizer_ft, scheduler_lr,  \
        device, w_config, classes_name, wandb, patience= patience,num_epoch=w_config.epochs)

    #model_ft = sweep_train.train_model(dataloaders, dataset_sizes, num_iteration, net, criterion, optimizer_ft, scheduler_warmup,  device, wandb, num_epoch=30)

#sweep_id = wandb.sweep(config.sweep_config, project="test", entity="douner89")
#sweep_id = wandb.sweep(config.sweep_config, project="rsna_covid", entity="89douner")

project_name = '' # 프로젝트 이름을 설정해주세요.
entity_name  = '' # 사용자의 이름을 설정해주세요.
sweep_id = wandb.sweep(config.sweep_config, project=project_name, entity=entity_name)

wandb.agent(sweep_id, wandb_setting, count=32)





