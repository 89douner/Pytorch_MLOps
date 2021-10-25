import torch
from torchvision import transforms


import numpy as np
from PIL import Image
import os

import albumentations as A
from albumentations.pytorch import transforms


class DiseaseDataset(object):
    def __init__(self, data_dir, img_size, bit, num_classes, classes_name, data_type=None, mode=None):
        self.data_dir = data_dir
        self.image_size = img_size
        self.type = data_type
        self.mode = mode
        self.bit = bit #bit(color) depth
        self.numclasses = num_classes
        
        if self.mode == 'train':
            self.transforms = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.2),
                A.OneOf([
                        A.OpticalDistortion(p=0.3),
                        ], p=0.2),
                A.OneOf([
                        A.GaussNoise(p=0.2),
                        A.MultiplicativeNoise(p=0.2),
                        ], p=0.2),
                A.Normalize(mean=(0.485), std=(0.229)),
                transforms.ToTensorV2()
            ])
        
        elif self.mode == 'val':
            self.transforms = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=0.5, std=0.5),
                transforms.ToTensorV2()
            ])

        ###data_dir -> train or val 폴더
        self.lst_data = os.listdir(self.data_dir) #train or val 폴더에 들어있는 하위 폴더 -> 클래스 명(알파벳순으로 인덱스 부여) -> ex) normal, pneumonia → lst_data[0]='normal'
        a = len(self.lst_data)

        ###data_dir -> train or val // train or val 데이터셋 파일 명과 레이블을 tuple 형태로 구성
        for i in range(0,a):
            if i == 0:
                self.imgs = self.imgs = [(self.lst_data[i], j) for j in list(sorted(os.listdir(os.path.join(self.data_dir, self.lst_data[i]))))]
            else:
                self.imgs += [(self.lst_data[i], j) for j  in list(sorted(os.listdir(os.path.join(self.data_dir, self.lst_data[i]))))]

        
    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, idx):

        img_path = os.path.join(self.data_dir, self.imgs[idx][0], self.imgs[idx][1]) #← train or val dataset path + 폴더명(=클래스명) + 파일명 
        
        if self.type == 'img':
            img_data = Image.open(img_path)

            #######train or val 데이터 중에 RGB 타입이 숨어있을 경우########
            if img_data.mode == 'RGB':
                 img_data = img_data.convert('L')
            
            ##########################전처리 코드 시작##########################
            np_img = np.array(img_data)/255.0 #pixel 값을 0~1 사이의 값으로 표준화
            #만약 dicom bit depth가 0~255가 아니면 255 외 다른 값으로 설정해주는 코드를 추가해야함
            ##########################전처리 코드 끝############################

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

# For test
if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), "RSNA_COVID_png_512") #train, val 폴더가 들어있는 경로
    num_classes =  len(os.listdir(os.path.join(data_dir, 'train')))

    train_data_dir = os.path.join(os.getcwd(), "RSNA_COVID_png_512", "train")
    train_dataset = DiseaseDataset(train_data_dir, 512, 8, num_classes, 'img', 'train')
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=0)

    for imgs, labels in iter(dataloader):
        print(imgs.shape, labels.shape)