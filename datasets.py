import glob

import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class Radar(Dataset):
    def __init__(self, train =False):
        super(Radar, self).__init__()
        self.path = '/home/seongbeom/paper/anomaly/data/radar_test'
        self.transform = None
        self.num_imgs = len(glob.glob(self.path+'/*.jpg'))
        self.img_list = glob.glob(self.path + '/*.jpg')
        self.img_list = train
        self.idx = np.array([i for i in range(self.__len__())], dtype=int)
        self.Image_Transform()
    
    def Image_Transform(self):
        # 이미지 변환용
        if self.is_train:
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
        idx = idx
        img = Image.open(self.img_list[idx])
        img = self.transform(img)

        #label mapping
        return img #label, age

