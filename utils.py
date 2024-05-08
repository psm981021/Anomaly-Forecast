import numpy as np
import random
import os
import torch
import zipfile
from tqdm import tqdm
import cv2 
import pandas as pd
from PIL import Image
from torchvision import transforms 

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")

def z_to_r(z):
    """
    Convert radar reflectivity (Z in dBZ) to rainfall rate (R in mm/hr) using the equation Z = 200 * R^1.6.
    """
    return np.power(z / 200, 1 / 1.6)


def make_image_csv(path,file_name=None):

    image_files = sorted(os.listdir(path))

    rain_amounts = []
    # calculate rain intensity for every image
    # V1 does not account for a specific reigon, further modification is needed
    
    rain_image = tqdm(image_files)
    rain_threshold = 0.2
    print(f"Start tqdm with length of: {len(rain_image)}")
    for image in rain_image:
        image_path = os.path.join(path, image)
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        transform_ = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((250,250))
        ])
        transform_input = transform_(image_np)
        
        #z to r transformation
        z_values = np.mean(transform_input.numpy(), axis=(0, 1, 2))
        intensity = z_to_r(z_values)
        
        rain_amounts.append(round(intensity, 2))

            
    # image path
    df =pd.DataFrame({"Image_Path":image_files, "Rain_Intensity": rain_amounts})
    df["Set"] = "Train"

    images_per_day =6 * 24
    for i in range(0,len(df), images_per_day * 8):
        train_end_index = i + (images_per_day * 4)  # Train for 4 days
        valid_end_index = train_end_index + images_per_day*2  # Validation for 2 day
        test_end_index = valid_end_index + images_per_day*2  # Test for 2 day

        df.loc[i:train_end_index - 1, "Set"] = "Train"  # Exclude train_end_ index
        df.loc[train_end_index:valid_end_index - 1, "Set"] = "Valid"  # Exclude valid_end_index
        df.loc[valid_end_index:test_end_index - 1, "Set"] = "Test"  # Exclude test_en d_index
        
    if file_name:
        df.to_csv(file_name, index=False)
    return df

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")
        

if __name__ == "__main__":
    make_image_csv('data/radar_test','data/image_loader.csv')