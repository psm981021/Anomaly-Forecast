import glob

import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd


class Radar(Dataset):
    def __init__(self, flag=None):
        super(Radar, self).__init__()
        self.path = 'data/radar_test'
        self.transform = None
        self.num_imgs = len(glob.glob(self.path+'/*.png'))
        self.img_list = glob.glob(self.path + '/*.png')
        self.flag=flag
        self.idx = np.array([i for i in range(self.__len__())], dtype=int)
        self.Image_Transform()
    
    def Image_Transform(self):
        # 이미지 변환용
        if self.flag =="Train":
            # Train 
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((60, 60)),
                transforms.Normalize(0, 1),
                transforms.RandomCrop(60),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((60, 60)),
            ])
        pass

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        assert self.flag in {"Train", "Valid", "Test"}
        data=pd.read_csv("loader_test.csv")
        idx = idx
        # img = Image.open(self.img_list[idx])
        # img = self.transform(img)
        # label=data['Rain_Intensity'][idx]
        #label mapping
        if self.flag == "Train":
            train_index=np.array(data[data['Set']=="Train"].index, dtype=int)
            train_img = Image.open(self.img_list[train_index])
            train_img = self.transform(train_img)
            train_label=data['Rain_Intensity'][train_index]
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



dataset=Radar(flag="Train")
train_loader = torch.utils.data.DataLoader(dataset,batch_size=8)
import IPython; IPython.embed(colors='Linux'); exit(1)
for i, (data) in enumerate(train_loader):
    print(data)
    
print("end")
print(dataset)
if __name__ == "__main__":
    Radar()
