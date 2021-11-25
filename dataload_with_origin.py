import torch
from torchvision import transforms

import time
import numpy as np
from PIL import Image
import os

import albumentations as A
from albumentations.pytorch import transforms
from albumentations.augmentations.geometric.rotate import Rotate


class DiseaseDatasetOrig(object):
    def __init__(self, data_dir, img_size, bit, num_classes, classes_name, data_type=None, mode=None, w_config=None):
        self.data_dir = data_dir
        self.image_size = img_size
        self.type = data_type
        self.mode = mode
        self.bit = bit #bit(color) depth
        self.numclasses = num_classes
        self.imgs = []

        ########################## 전처리 코드 ##########################
        if self.mode == 'train':
            self.transforms = A.Compose([
                 A.OneOf([
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.MotionBlur(p=0.2),
                    A.Sharpen(alpha=(0.01, 0.2), lightness=(0.5, 1.0), always_apply=False, p=0.2),
                    ], p=0.2),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), 
                contrast_limit=(-0.2, 0.1), p=0.6),
                A.OneOf([
                    A.GaussNoise(var_limit = 0.005, p=0.2),
                    A.MultiplicativeNoise(p=0.2),
                    ], p=0.2),
                A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, 
                                    val_shift_limit=0.1, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.0625, 
                                    scale_limit=0.2, 
                                    rotate_limit=10, p=0.2),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    ], p=0.2),
                
                A.Normalize(mean=(0.6254), std=(0.2712)),
                transforms.ToTensorV2(),
            ])
            #  A.OneOf([
            #     A.MedianBlur(blur_limit=w_config.blur, p=0.1),
            #     A.MotionBlur(p=0.2),
            #     A.Sharpen(alpha=(0.01, 0.2), lightness=(0.5, 1.0), always_apply=False, p=0.2),
            #     ], p=0.2),

            # A.RandomBrightnessContrast(brightness_limit=w_config.brightness, contrast_limit=w_config.contrast, p=0.6),

            # A.OneOf([
            #     A.GaussNoise(var_limit = w_config.noise, p=0.2),
            #     A.MultiplicativeNoise(p=0.2),
            #     ], p=0.2),

            # A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, 
            #                     val_shift_limit=0.1, p=0.3),

            # A.ShiftScaleRotate(shift_limit=w_config.shift, 
            #                     scale_limit=0.2, 
            #                     rotate_limit=w_config.rotate, p=0.2),
            # A.OneOf([
            #     A.OpticalDistortion(distort_limit= w_config.distortion, p=0.3),
            #     ], p=0.2),

        
        elif self.mode == 'val':
            self.transforms = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=0.6254, std=0.2712),
                transforms.ToTensorV2()
            ])
        elif self.mode == 'test':
            self.transforms = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=0.6254, std=0.2712),
                transforms.ToTensorV2()
            ])
        ##########################전처리 코드 끝############################

        ###data_dir -> train or val 폴더
        self.lst_data = os.listdir(self.data_dir) #train or val 폴더에 들어있는 하위 폴더 -> 클래스 명(알파벳순으로 인덱스 부여) -> ex) normal, pneumonia → lst_data[0]='normal'
        a = len(self.lst_data)

        ###data_dir -> train or val // train or val 데이터셋 파일 명과 레이블을 tuple 형태로 구성
        for i in range(0, a):
            lst_data_file_name = list(sorted(os.listdir(os.path.join(self.data_dir, self.lst_data[i]))))
            self.imgs.extend([(self.lst_data[i], j) for j in lst_data_file_name])
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        #train or val dataset path + 폴더명(=클래스명) + 파일명
        img_path = os.path.join(self.data_dir, self.imgs[idx][0], self.imgs[idx][1])  
        
        if self.type == 'img':
            img_data = Image.open(img_path)

            #######train or val 데이터 중에 RGB 타입이 숨어있을 경우########
            if img_data.mode == 'RGB':
                 img_data = img_data.convert('L')
            
            # 이미 Albumentation에 ToTensorV2가 있기 때문에 255로 나눌 필요가 없음
            np_img = np.array(img_data)
            
            #gray scale인 경우 np_img의 차원이 width, height 2차원으로만 구성 → dimension 차원이 생략됨
            #딥러닝 모델의 모든 입력 값은 dimension 차원을 포함한 3차원으로 구성되어야 함           
            if np_img.ndim == 2:
                np_img = np_img[:, :, np.newaxis]
            #↑↑↑그래서 np.newaxis로 차원 하나를 더 만듦  (width, height) → (width, height, dimension)

            for i in range(0, self.numclasses):
                if self.imgs[idx][0] == self.lst_data[i]:
                    label = i 
            
            #label = 0 if self.imgs[idx][0] == 'NORMAL' else 1
            img = self.transforms(image=np_img)["image"]
            
        
        elif type == 'numpy':
            numpy_data = np.load(os.path.join(self.data_dir, self.imgs[idx][0], self.imgs[idx][1]))
            
            if numpy_data.ndim == 2:
                np_img = numpy_data[:, :, np.newaxis]

            if numpy_data.ndim == 3:
                print("check")
        
        return img, label


if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), "RSNA_COVID_png_512") #train, val 폴더가 들어있는 경로
    num_classes =  len(os.listdir(os.path.join(data_dir, 'train')))
    classes_name = os.listdir(os.path.join(data_dir, 'train'))

    train_data_dir = os.path.join(os.getcwd(), "RSNA_COVID_png_512", "train")
    train_dataset = DiseaseDataset(train_data_dir, 512, 8, num_classes, classes_name, 'img', 'train')
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=0)

    #For shape test
    for imgs, labels in iter(dataloader):
        print(imgs.shape, labels.shape)