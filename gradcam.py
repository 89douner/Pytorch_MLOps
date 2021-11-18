from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import model 

from utils import load_net
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
import torch 
import cv2
from PIL import Image
from dataload_with_origin import DiseaseDatasetOrig
from config import CKPT_DIR, RESULTS_DIR
from adabelief_pytorch import AdaBelief


def eval_model(val_loader, net, criterion, optim, data_dir, imgs):
    if os.listdir(CKPT_DIR):
        net, optim, _ = load_net(ckpt_dir=CKPT_DIR, net=net, optim=optim)

    target_layers1 = [net.backbone.layer1[-1]]
    target_layers2 = [net.backbone.layer2[-1]]
    target_layers3 = [net.backbone.layer3[-1]]
    target_layers4 = [net.backbone.layer4[-1]]

    target_layers = [target_layers1, target_layers2, target_layers3, target_layers4]

    for layer_num in range(1, 5):
        for batch_idx,  (input_tensor, _) in enumerate(val_loader):
            # Construct the CAM object once, and then re-use it on many images:
            cam = GradCAM(model=net, target_layers=target_layers[layer_num-1], use_cuda='cuda:0')

            # You can also use it within a with statement, to make sure it is freed,
            # In case you need to re-create it inside an outer loop:
            # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
            #   ...

            # If target_category is None, the highest scoring category
            # will be used for every image in the batch.
            # target_category can also be an integer, or a list of different integers
            # for every image in the batch.
            target_category = 1

            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]

            img_path = os.path.join(data_dir, imgs[batch_idx][0], imgs[batch_idx][1])

            input_rgb = Image.open(img_path).convert('RGB')
            input_rgb_1 = input_rgb.copy()
            
            input_rgb = np.float32(input_rgb) / 255
            visualization = show_cam_on_image(input_rgb, grayscale_cam, use_rgb=True)

            concat_img = np.hstack((input_rgb_1, visualization))

            LAYER_DIR = os.path.join(RESULTS_DIR,  f'layer{layer_num}', str(imgs[batch_idx][0]))

            if not os.path.exists(LAYER_DIR):
                print('maked: ', LAYER_DIR)
                os.makedirs(LAYER_DIR, exist_ok=True)

            plt.imsave(os.path.join(LAYER_DIR ,f'img_{batch_idx:04}.png'), concat_img.squeeze(), cmap='gray')
        print(LAYER_DIR, ' 저장 완료!')


if __name__=='__main__':
    data_dir = os.path.join(os.getcwd(), "RSNA_COVID_png_512") #train, val 폴더가 들어있는 경로
    num_classes =  len(os.listdir(os.path.join(data_dir, 'val')))
    classes_name = os.listdir(os.path.join(data_dir, 'val'))


    val_data_dir = os.path.join(os.getcwd(), "RSNA_COVID_png_512", "val")
    val_dataset = DiseaseDatasetOrig(val_data_dir, 512, 8, num_classes, classes_name, 'img', 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0)

    imgs = []
    lst_data = os.listdir(val_data_dir)
    a = len(lst_data)

    for i in range(0, a):
        lst_data_file_name = list(sorted(os.listdir(os.path.join(val_data_dir, lst_data[i]))))
        imgs.extend([(lst_data[i], j) for j in lst_data_file_name])

    net = model.Pre_Resnet50(img_channel=1, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optim = AdaBelief(net.parameters(), lr=1e-3, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = True)

    eval_model(val_loader, net, criterion, optim, val_data_dir, imgs)