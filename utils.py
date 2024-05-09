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
from datetime import datetime, timedelta
import glob

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


#All csvs containing average precipitation for each region must exist in data_path
#truncate_start format example: '2022-07-01 00:00'
#truncate_end format example: '2022-09-30 23:50'

def make_model_input(data_path, truncate_start = None, truncate_end = None, threshold = None):
    file_names = glob.glob(data_path + "*.csv") #Loads a list of all csv files in a folder
    
    data = pd.DataFrame() #Create an empty data frame

    for file_name in file_names:
        temp = pd.read_csv(str(file_name), encoding='utf-8') #Open the csv files one by one and create a temporary data frame
        data = pd.concat([data, temp], axis = 1) #Add to entire data frame
    
    #Remove overlapping columns
    cols = list(data.columns)
    cols[0] = 'Timestamp'
    data.columns = cols
    data.drop('일시', axis=1, inplace=True) 
    
    # Assign average value to 'Label' column
    tmp_data = data.drop('Timestamp', axis=1)
    data['Label'] = tmp_data.mean(axis=1)
    
    # Create ‘Label Gap’ column
    data['Label Gap'] = data['Label'].diff()
    
    # Drop local data columns.
    region = data.columns[1:4]
    data.drop(region, axis=1, inplace=True)
    
    # Create columns to add image paths for each timestamp
    insert_cols = ['t-60', 't-50', 't-40', 't-30', 't-20', 't-10', 't']
    for i in range(1, len(insert_cols) + 1):
        data.insert(i, insert_cols[i - 1], np.nan)
    
    # Set start and end dates
    start_date = datetime(2021, 1, 1, 0, 0)
    end_date = datetime(2023, 12, 31, 23, 00)

    # Create a date_list at 10 minute intervals
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y%m%d%H%M'))
        current_date += timedelta(minutes=10)
    
    # Fill values in dataframe with image path
    idx = 0
    for i in range(1, len(data)):
        for j in range(1, 8):
            data.iloc[i, j] = date_list[idx] + '.png'
            if j == 7:
                pass
            else:
                idx += 1
    
    # Cut rows with data from a specific timestamp
    if truncate_start:
        data = data[truncate_start <= data['Timestamp']].reset_index(drop=True)
    if truncate_end:
        data = data[data['Timestamp'] <= truncate_end].reset_index(drop=True)
    
    # Leave only rows with precipitation above threshold
    if threshold:
        data = data[data['Label'] >= threshold].reset_index(drop=True)
    
    return data


if __name__ == "__main__":
    make_image_csv('data/radar_test','data/image_loader.csv')