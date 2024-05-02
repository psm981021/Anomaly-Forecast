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

    def __getitem__(self, idx):
        assert self.flag in {"Train", "Valid", "Test"}
        data=pd.read_csv("loader_test.csv")
        idx = idx
        img = Image.open(self.img_list[idx])
        img = self.transform(img)
        label=data['Rain_Intensity'][idx]
        #label mapping
        if self.flag == "Train":
            pass
        elif self.flag == "Valid":
            pass
        else:
            pass


        return img, label



dataset=Radar(train=True)
train_loader = torch.utils.data.DataLoader(dataset,batch_size=4)
import IPython; IPython.embed(colors='Linux'); exit(1)
for i, (data) in enumerate(train_loader):
    print(data)
    
print("end")
print(dataset)
if __name__ == "__main__":
    Radar()
