import torch
from torch._C import device
from torchvision import transforms

import time
import numpy as np
from PIL import Image
import os
from glob import glob

import albumentations as A
from albumentations.pytorch import transforms  
from albumentations.augmentations.geometric.rotate import Rotate

class GpuDataset(object):
    def __init__(self, data_dir, img_size, bit, num_classes, classes_name, data_type=None, mode=None, w_config=None):
        self.data_dir = data_dir
        self.image_size = img_size
        self.type = data_type
        self.mode = mode
        self.bit = bit #bit(color) depth
        self.numclasses = num_classes
        self.imgs = []
        self.label = []
        
        s_p, c_p, r_p, d_p, n_p = 0, 0, 0, 0, 0

        s_p = float(w_config.shift == 'yes')
        c_p = float(w_config.contrast == 'yes')
        r_p = float(w_config.rotate == 'yes')
        d_p = float(w_config.distortion == 'yes')
        n_p = float(w_config.noise == 'yes')

        ########################## 전처리 코드 ##########################
        if self.mode == 'train':
            self.transforms = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=0, p=s_p),
            A.RandomContrast(p=c_p),
            Rotate(limit=15, p=r_p),
            A.OneOf([
                A.OpticalDistortion(p=1.0),
                ], p=d_p),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.MultiplicativeNoise(p=1.0),
                ], p=n_p),

            A.Normalize(mean=0.658, std=0.221),
            transforms.ToTensorV2(),
            ])
        
        elif self.mode == 'val':
            self.transforms = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=0.658, std=0.221),
                transforms.ToTensorV2()
            ])
        ##########################전처리 코드 끝############################

        ###data_dir -> train or val 폴더
        self.lst_data = os.listdir(self.data_dir) #train or val 폴더에 들어있는 하위 폴더 -> 클래스 명(알파벳순으로 인덱스 부여) -> ex) normal, pneumonia → lst_data[0]='normal'
        len_lst_data = len(self.lst_data)

        ### data_dir -> train or val // train or val 데이터셋 파일 명을 불러오고, label을 label의 index로 저장.
        for i in range(0, len_lst_data):
            lst_data_file_name = sorted(glob(os.path.join(self.data_dir, self.lst_data[i], '*.png')))
            for img in lst_data_file_name:
                self.imgs.extend([self.pre_transforms(Image.open(img).convert('L'))])
            self.label.extend([i]*len(self.imgs))
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        #train or val dataset path + 폴더명(=클래스명) + 파일명
        if self.type == 'img':     
            #label = 0 if self.imgs[idx][0] == 'NORMAL' else 1
            img = self.transforms(image=self.imgs[idx])['image']
            label = self.label[idx]

            if img.ndim == 2:
                img = img[:, :, np.newaxis]

        elif type == 'numpy':
            img = self.imgs[idx]
            label = self.label[idx]

            if img.ndim == 2:
                img = img[:, :, np.newaxis]

            if img.ndim == 3:
                print("check")
        
        return img, label
            
            #gray scale인 경우 np_img의 차원이 width, height 2차원으로만 구성 → dimension 차원이 생략됨
            #딥러닝 모델의 모든 입력 값은 dimension 차원을 포함한 3차원으로 구성되어야 함           

    def pre_transforms(self, img_data):
        self.img_data = np.array(img_data)

        if self.img_data.ndim == 2:
                self.img_data = self.img_data[:, :, np.newaxis]

        return  self.img_data

if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), "RSNA_COVID_png_512") #train, val 폴더가 들어있는 경로
    num_classes =  len(os.listdir(os.path.join(data_dir, 'train')))
    classes_name = os.listdir(os.path.join(data_dir, 'train'))

    train_data_dir = os.path.join(os.getcwd(), "RSNA_COVID_png_512", "train")
    train_dataset = GpuDataset(train_data_dir, 512, 8, num_classes, classes_name, 'img', 'train')
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=0)

    #For shape test
    for imgs, labels in iter(dataloader):
        print(imgs.shape, labels.shape)

    # For time test
    # device = 'cuda'
    # iteration = 10
    # for _ in range(iteration):
    #     since = time.time()
    #     for imgs, labels in iter(dataloader):
    #         torch.nn.Conv2d(1, 3, (3, 3), device=device)(imgs.to(device))
    
    # time_elapsed = time.time() - since
    # print(time_elapsed)