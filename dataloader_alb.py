import torch
from torchvision import transforms

import time
import numpy as np
from PIL import Image
import os

import albumentations as A
from albumentations.pytorch import transforms


class DiseaseDataset(object):
    def __init__(self, data_dir, img_size, bit, num_classes, classes_name, data_type=None, mode=None, w_config=None):
        self.data_dir = data_dir
        self.image_size = img_size
        self.type = data_type
        self.mode = mode
        self.bit = bit #bit(color) depth
        self.numclasses = num_classes
        self.imgs = []

        self.data_dir_name = os.path.basename(os.path.normpath(self.data_dir))

        ########################## 전처리 코드 ##########################
        if self.mode == 'train':
            self.transforms = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Flip(p=0.5),
                A.HorizontalFlip(p=0.5),
                #A.Normalize(mean=0.0, std=1.0),
                A.Normalize(mean=0.5, std=0.5), #img = (img - mean * max_pixel_value) / (std * max_pixel_value) <-  max_pixel_value=255.0 로 세팅되어 있음 (github 참고)
                transforms.ToTensorV2()
            ])
        elif self.mode == 'val':
            self.transforms = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=0.5, std=0.5), #img = (img - mean * max_pixel_value) / (std * max_pixel_value) <-  max_pixel_value=255.0 로 세팅되어 있음 (github 참고)
                #A.Normalize(mean=0.0, std=1),
                transforms.ToTensorV2()
            ])
        elif self.mode == 'test':
            self.transforms = A.Compose([
                A.Resize(self.image_size, self.image_size),
                #A.Normalize(mean=0.0, std=1.0),
                A.Normalize(mean=0.5, std=0.5), #img = (img - mean * max_pixel_value) / (std * max_pixel_value) <-  max_pixel_value=255.0 로 세팅되어 있음 (github 참고)
                transforms.ToTensorV2()
            ])
        ##########################전처리 코드 끝############################

        ###data_dir -> train or val 폴더
        self.lst_data = os.listdir(self.data_dir) #train or val 폴더에 들어있는 하위 폴더 -> 클래스 명(알파벳순으로 인덱스 부여) -> ex) normal, pneumonia → lst_data[0]='normal'
        a = len(self.lst_data)

        ###data_dir -> train or val // train or val 데이터셋 파일 명과 레이블을 tuple 형태로 구성 -> ex) imgs=(class 명, 파일 이름)
        for i in range(0, a):
            self.lst_data_file_name = list(sorted(os.listdir(os.path.join(self.data_dir, self.lst_data[i]))))
            self.imgs.extend([(self.lst_data[i], j) for j in self.lst_data_file_name])
            
        print("check")

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

            return img, label
            
        
        elif self.type == 'numpy':
            numpy_data = np.load(os.path.join(self.data_dir, self.imgs[idx][0], self.imgs[idx][1]))
            min_numpy_data = np.min(numpy_data)
            max_numpy_data = np.max(numpy_data)

            norm_check = np.max(numpy_data) #numpy data가 normalization 되어 있는건지 체크, albamentation에서 normalization 들어가기 때문에 사전에 np 값이 normalization 되면 안 됨 (그래서 체크)
            if norm_check < 1.1:
                original_numpy = numpy_data*65535 #최초의 데이터 범위 (0~65535)
                numpy_data = original_numpy/255 #albumentation의 A.Normalize의 max_pixel_value가 255로 세팅되었기 때문에, 255로 세팅
                #위의 두 줄을 그냥 애초에 numpy_data=numpy*255로 하면되는데, 가독성 때문에 위와 같이 구현
                max_check = np.max(numpy_data)
                min_check = np.min(numpy_data)
            
            #gray scale인 경우 np_img의 차원이 width, height 2차원으로만 구성 → dimension 차원이 생략됨
            #딥러닝 모델의 모든 입력 값은 dimension 차원을 포함한 3차원으로 구성되어야 함     
            if numpy_data.ndim == 2:
                np_img = numpy_data[:, :, np.newaxis]
            #↑↑↑그래서 np.newaxis로 차원 하나를 더 만듦  (width, height) → (width, height, dimension)

            if numpy_data.ndim == 3:
                print("check")

            for i in range(0, self.numclasses):
                if self.imgs[idx][0] == self.lst_data[i]:
                    label = i 
            
            np_img = self.transforms(image=np_img)["image"]
            np_img_max = torch.max(np_img)
            np_img_min = torch.min(np_img)

            return np_img, label


if __name__ == '__main__':
    data_dir = '/workspace/dataset/' #train, val 폴더가 들어있는 경로
    num_classes =  3
    classes_name = 'class0'

    train_data_dir = os.path.join(data_dir, "val")
    train_dataset = DiseaseDataset(train_data_dir, 500, 8, num_classes, classes_name, 'numpy', 'val')
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=0)

    #For shape test
    for imgs, labels in iter(dataloader):
        print(imgs.shape, labels.shape)