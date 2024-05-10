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
    def __init__(self, args, csv_path,flag= None, augmentations= None ):
        super(Radar, self).__init__()
        self.path = 'data\\radar_test'
        self.csv_path = csv_path
        self.flag=flag
        self.augmentations = augmentations
        self.args = args

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((250, 250))
        ])
        
        self.idx = np.array([i for i in range(self.__len__())], dtype=int)
        self.image, self.label, self.gap = self.get_input(csv_path, self.flag)


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
        return self.image[idx], self.label[idx],self.gap[idx]
    
    def get_input(self, csv_path, flag):
        
        data=pd.read_csv(csv_path)
        data=data[data['Set']==flag].reset_index(drop=True)
        idx = np.array([i for i in range(len(data))], dtype=int)

        dataset_images = []
        labels=data['Label'].values
        gaps=data['Label Gap'].values
        # import IPython; IPython.embed(colors='Linux'); exit(1)
        for i in tqdm(idx):
            tmp = data.loc[i]
            batch_images = []

            # Collect images and associated data
            for j in range(1, 8):
                img_path = os.path.join(self.args.data_dir, tmp[j])
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                batch_images.append(image)

            # Convert batches to tensors and append to dataset lists
            dataset_images.append(batch_images)

        # Combine all batches into a single dataset
        # Each element of dataset now corresponds to batched images, labels, or gaps respectively
        # import IPython; IPython.embed(colors='Linux');exit(1);
        return (dataset_images,
                torch.Tensor(labels).type(torch.float),
                torch.Tensor(gaps).type(torch.float))


        # original code from 5/8
        # def image_set(self, flag=None ): 
        #     data =data=pd.read_csv(self.image_csv_dir)
            
        #     if flag =="Train":
        #         train_index=np.array(data[data['Set']=="Train"].index, dtype=int)
        #         self.Image_Transform(flag)
                

        #         train_img = []
        #         train_label = []
        #         train_img_tmp =[]

        #         for i in tqdm(train_index):
        #             img = Image.open(os.path.join(self.path, data['Image_Path'][i]))
                    
        #             img = self.transform(img)
        #             train_img_tmp.append(img)

        #             if (i+1)%6==0:
        #                 train_img.append(train_img_tmp)
        #                 train_label.append(data['Rain_Intensity'][i])
        #                 train_img_tmp.clear()

        #     return train_img, train_label

        # original code in 5/8

        # assert self.flag in {"Train", "Valid", "Test"}
        # data=pd.read_csv(self.image_csv_dir)
        # #label mapping
        # print(self.flag)
        # if self.flag == "Train":
            
        #     train_img, train_label = self.image_set("Train")
            
        #     return train_img, train_label

        # elif self.flag == "Valid":
        #     valid=data[data['Set']=="Valid"]
        #     pass
        # else:
        #     test=data[data['Set']=='Test']
        #     pass


        # return img, label


if __name__ == "__main__":
    Radar()
