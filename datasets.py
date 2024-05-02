import glob

import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import pandas as pd
import random

from tqdm import tqdm

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
        if flag =="Train":
            # Train
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((250,250))
            ])
        elif flag == "Valid":
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
        # img = Image.open(self.img_list[idx])
        # img = self.transform(img)
        # label=data['Rain_Intensity'][idx]
        #label mapping
        if self.flag == "Train":
            # img = self.transform(img, self.flag)
            
            #augmentation if needed
            # if self.augmentations:
            #     img = self.apply_augmentation(img)

            self.Image_Transform(flag=self.flag)
            train_index=np.array(data[data['Set']=="Train"].index, dtype=int)
            # train_img_list=list(data['Image_Path'][train_index])
            train_img=[]
            train_label=[]
            train_img_tmp=[]
            for i in tqdm(train_index):
                img = Image.open(self.path + '/' + data['Image_Path'][i])
                img = self.transform(img)
                train_img_tmp.append(img)
                if (i+1)%6==0:
                    train_img.append(train_img_tmp)
                    train_label.append(data['Rain_Intensity'][i])
                    train_img_tmp.clear()

            # train_img=img[train_index]
            # train_label=label[train_index]

            return train_img, train_label

        elif self.flag == "Valid":
            valid=data[data['Set']=="Valid"]
            pass
        else:
            test=data[data['Set']=='Test']
            pass


        # return img, label


if __name__ == "__main__":
    Radar()
