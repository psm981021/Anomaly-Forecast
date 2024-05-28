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
import matplotlib.pyplot as plt

class Radar(Dataset):
    def __init__(self, args, csv_path,flag= None, augmentations= None ):
        super(Radar, self).__init__()
        self.path = 'data\\radar_test'
        self.csv_path = csv_path
        self.flag=flag # train/valid/test
        self.augmentations = augmentations
        self.args = args

        if args.location == "seoul": 
            # 이미지 자를 좌표 - 서울 : 150x150
            self.left = 240  
            self.top = 120   
            self.right = 390
            self.bottom = 270
        else:
            # 이미지 자를 좌표 - 강원도 150x150
            self.left = 300
            self.top = 110  
            self.right = 450
            self.bottom = 260
        
        self.transform = transforms.Compose([
            #transforms.CenterCrop((250, 250)) # centercrop 말고 
            transforms.Lambda(lambda x: x.crop((self.left, self.top, self.right, self.bottom))),  # 이미지 crop
            transforms.ToTensor() # totensor는 나중에
        ])
        
        self.idx = np.array([i for i in range(self.__len__())], dtype=int)
        self.image, self.label, self.gap, self.date = self.get_input(csv_path, self.flag)
        # self.image, self.label, self.gap, self.date, self.class_label = self.get_input(csv_path, self.flag)

    def __len__(self):
        try: 
            self.data = pd.read_csv(self.csv_path)
            self.data=self.data[self.data['Set']==self.flag].reset_index(drop=True)
            # self.data = self.data[self.data['Set'] == self.flag]
            return len(self.data)
        except:
        # self.data contains all data including train/valid/test
            print("dataset len debug");import IPython; IPython.embed(colors='Linux');exit(1);
    
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
        return self.image[idx], self.label[idx],self.gap[idx], self.date[idx] # , self.class_label[idx]

    
    def get_input(self, csv_path, flag):
        data=pd.read_csv(csv_path)
        data=data[data['Set']==flag].reset_index(drop=True)
        idx = np.array([i for i in range(len(data))], dtype=int)

        dataset_images = []
        labels=data['Label'].values
        gaps=data['Label Gap'].values
        dataset_date = data['Timestamp'].values
        # class_label = data['Class_Label'].values

        # import IPython; IPython.embed(colors='Linux'); exit(1)
        for i in tqdm(idx):
            tmp = data.loc[i]
            batch_images = []

            # Collect images and associated data
            for j in range(1, 8):
                img_path = os.path.join(self.args.data_dir, tmp[j])
                if self.args.grey_scale:
                    image = Image.open(img_path).convert('L')
                else:
                    image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)

                
                batch_images.append(image)
            # Convert batches to tensors and append to dataset lists
            batch_tensor = torch.stack(batch_images, dim=0)
            dataset_images.append(batch_tensor)

        # Combine all batches into a single dataset
        # Each element of dataset now corresponds to batched images, labels, or gaps respectively

        return (dataset_images,
                torch.Tensor(labels).type(torch.float),
                torch.Tensor(gaps).type(torch.float),
                dataset_date,
                # torch.Tensor(class_label).type(torch.long)
                )


if __name__ == "__main__":
    Radar()
