import argparse
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
import model 

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}


    data_dir = os.path.join(os.getcwd(), "RSNA_COVID_png_512") #train, val 폴더가 들어있는 경로
    num_classes =  len(os.listdir(os.path.join(data_dir, 'test')))
    classes_name = os.listdir(os.path.join(data_dir, 'test'))
    test_data_dir = os.path.join(os.getcwd(), "RSNA_COVID_png_512", "test")

    imgs = []
    lst_data = os.listdir(test_data_dir)
    a = len(lst_data)

    for i in range(0, a):
        lst_data_file_name = list(sorted(os.listdir(os.path.join(test_data_dir, lst_data[i]))))
        imgs.extend([(lst_data[i], j) for j in lst_data_file_name])

    net = model.Pre_Resnet50(img_channel=1, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optim = AdaBelief(net.parameters(), lr=1e-3, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = True)

    if os.listdir(CKPT_DIR):
        net, optim, _ = load_net(ckpt_dir=CKPT_DIR, net=net, optim=optim)

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    target_layers1 = [net.backbone.layer1[-1]]
    target_layers2 = [net.backbone.layer2[-1]]
    target_layers3 = [net.backbone.layer3[-1]]
    target_layers4 = [net.backbone.layer4[-1]]

    target_layers = [target_layers1, target_layers2, target_layers3, target_layers4]
    methods = [ScoreCAM, GradCAMPlusPlus, LayerCAM]
    method_name = ["scorecam", "gradcam++", "layercam"]

    for sel_met in range(len(methods)):
        for layer_num in range(1, len(target_layers)+1):
            for img_idx in range(len(imgs)):
                img_path = os.path.join(test_data_dir, imgs[img_idx][0], imgs[img_idx][1])

                rgb_img = Image.open(img_path).convert('RGB')
                rgb_img_cp = rgb_img.copy()

                gray_img = rgb_img.convert('L')

                rgb_img = np.float32(rgb_img) / 255
                gray_img = np.float32(gray_img) / 255

                gray_input_tensor = preprocess_image(gray_img,
                                                mean=[0.6254],
                                                std=[0.2712])
                # If None, returns the map for the highest scoring category.
                # Otherwise, targets the requested category.
                target_category = 1

                # Using the with statement ensures the context is freed, and you can
                # recreate different CAM objects in a loop.
                cam_algorithm = methods[sel_met]

                with cam_algorithm(model=net,
                                    target_layers=target_layers[layer_num-1],
                                    use_cuda='cuda:0') as cam:

                    grayscale_cam = cam(input_tensor=gray_input_tensor,
                                        target_category=target_category,)

                    # Here grayscale_cam has only one image in the batch
                    grayscale_cam = grayscale_cam[0, :]

                    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                gb_model = GuidedBackpropReLUModel(model=net, use_cuda='cuda:0')
                gb = gb_model(gray_input_tensor, target_category=target_category)

                cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
                cam_gb = deprocess_image(cam_mask * gb)
                gb = deprocess_image(gb)

                concat_img = np.hstack((rgb_img_cp, cam_image))
                LAYER_DIR = os.path.join(RESULTS_DIR, f'{method_name[sel_met]}', f'layer{layer_num}', str(imgs[img_idx][0]))

                if not os.path.exists(LAYER_DIR):
                    print('maked: ', LAYER_DIR)
                    os.makedirs(LAYER_DIR, exist_ok=True)

                plt.imsave(os.path.join(LAYER_DIR ,f'img_{img_idx:04}.png'), concat_img.squeeze(), cmap='gray')
                print(f'[{method_name[sel_met].upper()}] layer-{layer_num} | {img_idx:04} / {len(imgs):04}')

            print(LAYER_DIR, ' 저장 완료!')