import torch
from collections import OrderedDict
import torchsummary

import model
import os

from dataloader_alb import DiseaseDataset
from torch.utils.data import DataLoader



data_path = '/workspace/dataset/' #train, val 폴더가 들어있는 경로
    
test_dir = 'test'
classes_name = os.listdir(os.path.join(data_path, test_dir)) #폴더에 들어있는 클래스명
num_classes =  len(os.listdir(os.path.join(data_path, test_dir))) #train 폴더 안에 클래스 개수 만큼의 폴더가 있음
batch_size =10

datasets = {x: DiseaseDataset(data_dir=os.path.join(data_path, x), img_size=224, bit=8, 
                num_classes=num_classes, classes_name=classes_name, data_type='numpy', mode= x, w_config=None) for x in [test_dir]}
dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in [test_dir]}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = model.Pre_Resnet50(img_channel=1, num_classes=3)
check_point = torch.load("/workspace/code/Pytorch_MLOps/resnet50_PA_2.pth")


new_state_dict = OrderedDict()
for k, v in check_point['net'].items():
    name = k[7:]
    #print(name)
    new_state_dict[name] = v

net.load_state_dict(new_state_dict)

net = net.cuda()
net.eval()

for iteration_th, (inputs, labels) in enumerate(dataloaders[test_dir]):
    inputs = inputs.to(device) #image 데이터 GPU에 업로드
    labels = labels.to(device)

    outputs = net(inputs)
    _, preds = torch.max(outputs, 1)
    print(outputs)
    print(preds)
    print("check")


