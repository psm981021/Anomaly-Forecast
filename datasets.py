import glob

import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import pandas as pd
import random


class Radar(Dataset):
    def __init__(self, csv_path, flag= None, augmentations= None, ):
        super(Radar, self).__init__()
        self.path = 'data/radar_test'
        self.image_csv_dir = csv_path 
        self.transform = None
        self.num_imgs = len(glob.glob(self.path+'/*.png'))
        self.img_list = glob.glob(self.path + '/*.png')
        self.flag=flag
        self.idx = np.array([i for i in range(self.__len__())], dtype=int)
        self.augmentations = augmentations
    
    def Image_Transform(self, flag):
        # 이미지 변환용
        if self.flag =="Train":
            # Train 
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((250,250))
            ])
        elif self.flag == "Valid":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((250,250))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((250,250))
            ])
        pass

    def __len__(self):
        return self.num_imgs
    
    def apply_augmentation(self, img):

        # Horizontal flip
        if random.random() >0.5:
            img = TF.hflip(img)

        # Random Rotation (clockwise and counterclockwise)
        if random.random() > 0.5:
            degrees = 10
            if random.random() > 0.5:
                degrees *= -1
            img = TF.rotate(img, degrees)

        # Brighten or darken image (only applied to input image)
        if random.random() > 0.5:
            brightness = 1.2
            if random.random() > 0.5:
                brightness -= 0.4
            img = TF.adjust_brightness(img, brightness)


    def __getitem__(self, idx):
        assert self.flag in {"Train", "Valid", "Test"}

        data=pd.read_csv(self.image_csv_dir)

        idx = idx

        img = Image.open(self.img_list[idx])
        img = self.transform(img)
        label=data['Rain_Intensity'][idx]

        #label mapping
        if self.flag == "Train":
            img = self.transform(img, self.flag)
            
            #augmentation if needed
            if self.augmentations:
                img = self.apply_augmentation(img)

        elif self.flag == "Valid":
            img = self.transform(img, self.flag)
        else:
            img = self.transform(img, self.flag)

            

        return img, label


if __name__ == "__main__":
    Radar()
