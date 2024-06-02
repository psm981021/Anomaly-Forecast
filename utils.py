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




class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self,log_file,checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.log_file = log_file
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        if score < self.best_score + self.delta:
            return False
        return True

    def __call__(self, score, model):
        # score

        if self.best_score is None:
            self.best_score = score
            self.score_min = 0
            self.save_checkpoint(score, model)

        elif self.compare(score):
            self.counter += 1

            with open(self.log_file, "a") as f:
                f.write(f"EarlyStopping counter: {self.counter} out of {self.patience}\n")
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f}) 
            print(f"Validation score increased.  Saving model ...")

        torch.save({
            'model_state_dict': model.state_dict(),
            }, self.checkpoint_path)

        self.score_min = score






def unify_data_by_timestamp(data_path, save_path):
    # Set start and end dates
    start_date = datetime(2021, 1, 1, 0, 0)
    end_date = datetime(2023, 12, 31, 23, 00)

    # Create a list of dates at 60-minute intervals
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y-%m-%d %H:%M'))
        current_date += timedelta(minutes=60)
    
    file_names = glob.glob(data_path + "*.csv") #Loads a list of all csv files in a folder
    
    data = pd.DataFrame() #Create an empty data frame

    for file_name in file_names:
        temp = pd.read_csv(str(file_name), encoding='cp949') #Open the csv files one by one and create a temporary data frame
        data = pd.concat([data, temp], axis = 0) #Add to entire data frame
        
    data = data[['일시', '강수량(mm)']].reset_index(drop=True)
    
    # Fill in blank dates
    full_dates_df = pd.DataFrame(date_list, columns=['일시'])

    # Merge full_dates_df and df based on 'date', and set data without '강수량(mm)' to NaN
    result_df = pd.merge(full_dates_df, data, on='일시', how='left')
    
    # Fill NaN values
    result_df['강수량(mm)'].fillna(0, inplace=True)
    
    result_df.to_csv(save_path + '연도 통합.csv', index=False, encoding='utf-8-sig')

#Annotation for make_model_input function
#Set 'data_path' variable to 'save_path' of unify_data_by_timestamp function
#truncate_start format example: '2022-07-01 00:00'
#truncate_end format example: '2022-09-30 23:50'

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
    cols[1] = 'Label'
    data.columns = cols
    if '일시' in data.columns:
        data.drop('일시', axis=1, inplace=True)
    
    # Create ‘Label Gap’ column
    data['Label Gap'] = data['Label'].diff()
    
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
    
    #make test data
    data['Set'] = np.nan
    
    # Extract rows corresponding to the top 10% of rainy days from the 'Label' column
    rained = data[data['Label'] >= threshold]
    top_10_percent = rained['Label'].quantile(0.9)
    top_10_list = list(data[data['Label'] >= top_10_percent].index)
    
    # Set top 10% as test data: prediction for outliers
    data.loc[top_10_list, 'Set'] = 'Test'
    
    for i in range(len(data)):
        # Set 2 hours before and 1 hour after the index of the outlier as test data.
        if i >= 2:
            if data.loc[i]['Set'] == 'Test':
                for j in range(i - 2, i + 2):
                    if j != i:
                        data.loc[j, 'Set'] = 'Test2'
    
    for i in range(len(data)):
        if data.loc[i]['Set'] == 'Test2':
            data.loc[i, 'Set'] = 'Test'
    
    # Cut rows with data from a specific timestamp
    if truncate_start:
        data = data[truncate_start <= data['Timestamp']].reset_index(drop=True)
    if truncate_end:
        data = data[data['Timestamp'] <= truncate_end].reset_index(drop=True)
    
    # Leave only rows with precipitation above threshold and 'Set' = 'Test
    if threshold:
        data = data[(data['Label'] >= threshold) | (~data['Set'].isna())].reset_index(drop=True)
        
    #data split 
    train_cnt = 0
    valid_cnt = 0

    for i in range(len(data)):
        if pd.isna(data.loc[i]['Set']):
            if train_cnt < 5:
                data.loc[i, 'Set'] = 'Train'
                train_cnt += 1
        
            if train_cnt >= 5 and valid_cnt < 2:
                data.loc[i, 'Set'] = 'Valid'
                valid_cnt += 1
            
            if train_cnt >= 5 and valid_cnt >= 2:
                valid_cnt = 0
                train_cnt = 0
    
    return data

def extract_image_file(csv_path,raw_data_path, extract_path):
    dataframe = pd.read_csv(csv_path)

    data_list = []
    path_columns = ['t-60','t-50','t-40','t-30','t-20','t-10','t']

    for index, row in dataframe.iterrows():
        for column in path_columns:
            full_path = os.path.join(raw_data_path, row[column])
            data_list.append(full_path)
    
    
    # make a directory if not exists
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    for file_path in data_list:
        if os.path.isfile(file_path):  # Check if the file exists
            # Copy the file to the new directory
            shutil.copy(file_path, extract_path)
        else:
            print(f"File not found: {file_path}")  




if __name__ == "__main__":
    make_image_csv('data/radar_test','data/image_loader.csv')