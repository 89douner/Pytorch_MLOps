import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import model
import sweep_train

from dataload_with_alb  import DiseaseDataset
from dataload_with_origin import DiseaseDatasetOrig

from torch.utils.data import DataLoader

from warmup_scheduler import GradualWarmupScheduler
from adabelief_pytorch import AdaBelief
from losses import FocalLoss, LovaszHingeLoss

import wandb
import config


import torch.distributed as dist
#from apex.apex.parallel import DistributedDataParallel as DDP

from imbalanced_dataset_sampler.torchsampler import ImbalancedDatasetSampler


def wandb_setting(sweep_config=None):
    wandb.init(config=sweep_config)
    w_config = wandb.config
    #name_str = 'loss: ' +  str(w_config.loss) + ' | l: ' +  str(w_config.learning_rate) + ' | o: ' + str(w_config.optimizer)
    name_str = 'bl:' +  str(round(w_config.blur, 3)) + ' -br:' +  str(round(w_config.brightness, 3)) + ' -c:' + str(round(w_config.contrast, 3)) + ' -n:' + str(round(w_config.noise, 3)) \
          + ' -s:' + str(round(w_config.shift, 3)) + ' -r:' + str(round(w_config.rotate,3)) + ' -d:' + str(round(w_config.distortion, 3)) 
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
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##########################################데이터 로드 하기#################################################
    data_path = os.path.join(os.getcwd(), "data") #train, val 폴더가 들어있는 경로
    train_dir = 'over_train'
    classes_name = os.listdir(os.path.join(data_path, train_dir)) #폴더에 들어있는 클래스명
    num_classes =  len(os.listdir(os.path.join(data_path, train_dir))) #train 폴더 안에 클래스 개수 만큼의 폴더가 있음
    
    datasets = {x: DiseaseDatasetOrig(data_dir=os.path.join(data_path, train_dir), img_size=512, bit=8, 
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

    #net = net.to(device) #딥러닝 모델 GPU 업로드
    _net = net.cuda()
    net = nn.DataParallel(_net).to(device)


    ###Focal loss Code####
    weights = torch.tensor([0.08, 0.17, 0.28, 0.47], dtype=torch.float32)
    weights = weights / weights.sum()
    weights = 1.0 / weights
    weights = weights / weights.sum()
    #################

     # Loss Function
    if w_config.loss == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss().to(device)
    elif w_config.loss == 'focal':
        criterion = FocalLoss(weight=weights, gamma=2).to(device)
    elif w_config.loss == 'LovaszHinge':
        criterion = LovaszHingeLoss().to(device)

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
    CKPT_DIR = os.path.join(os.getcwd(), "checkpoints_dir")
    ckpt_dir = os.path.join(CKPT_DIR, "checkpoints_" + name_str)

    patience = 10
    wandb.watch(net, log='all') #wandb에 남길 log 기록하기
    sweep_train.train_model(dataloaders, dataset_sizes, num_iteration, net, criterion, optimizer_ft, scheduler_lr,  \
        device, w_config, classes_name, wandb, patience= patience, ckpt_dir=ckpt_dir)

#sweep_id = wandb.sweep(config.sweep_config, project="test", entity="douner89")
#sweep_id = wandb.sweep(config.sweep_config, project="rsna_covid", entity="89douner")

project_name = 'data_augmentation_grid_acc' # 프로젝트 이름을 설정해주세요.
entity_name  = 'rsna_covid_down' # 사용자의 이름을 설정해주세요.
sweep_id = wandb.sweep(config.sweep_config, project=project_name, entity=entity_name)

wandb.agent(sweep_id, wandb_setting, count=2187)