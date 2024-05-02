import glob

import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import pandas as pd
import random
import os
from tqdm import tqdm

class Radar(Dataset):
    def __init__(self, csv_path, flag= None, augmentations= None, ):
        super(Radar, self).__init__()
        self.path = 'data/radar_test'
        self.image_csv_dir = csv_path 
        self.flag=flag
        self.idx = np.array([i for i in range(self.__len__())], dtype=int)
        self.augmentations = augmentations
    
    def Image_Transform(self, flag):
        # 이미지 변환용
        if flag =="Train":
<<<<<<< HEAD
            # Train 
=======
            # Train
>>>>>>> jw
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
        return 1#self.num_imgs
    
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

<<<<<<< HEAD
    def image_set(self, flag=None ): 
        data =data=pd.read_csv(self.image_csv_dir)
        
        if flag =="Train":
            train_index=np.array(data[data['Set']=="Train"].index, dtype=int)
            self.Image_Transform(flag)
            

            train_img = []
            train_label = []
            train_img_tmp =[]

            for i in tqdm(train_index):
                img = Image.open(os.path.join(self.path, data['Image_Path'][i]))
                
=======

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
>>>>>>> jw
                img = self.transform(img)
                train_img_tmp.append(img)

                if (i+1)%6==0:
                    train_img.append(train_img_tmp)
                    train_label.append(data['Rain_Intensity'][i])
                    train_img_tmp.clear()

        return train_img, train_label

    def __getitem__(self, idx):
        assert self.flag in {"Train", "Valid", "Test"}

        data=pd.read_csv(self.image_csv_dir)
        #label mapping
        print(self.flag)
        if self.flag == "Train":
            
            train_img, train_label = self.image_set("Train")
            
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
