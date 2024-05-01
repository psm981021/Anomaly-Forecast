import numpy as np
import random
import os
import torch
import zipfile
from tqdm import tqdm
import cv2 
import pandas as pd

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
        image = cv2.imread(image_path)

        crop_image = image[120:420,170:470]
        img_size = crop_image.shape[1]
        num_pixels = img_size * img_size
        
        import IPython; IPython.embed(colors='Linux');exit(1);

        # rain pixels accumulate
        rain_pixels = np.sum(crop_image > 0)
        total_pixels = np.prod(crop_image.shape[:2])
        rain_amount = rain_pixels / total_pixels
        rain_amounts.append(round(rain_amount,2))
            
    # image path
    df =pd.DataFrame({"Image_Path":image_files, "Rain_Intensity": rain_amounts})
    df["Set"] = "Train"

    images_per_day =6 * 24
    for i in range(0,len(df), images_per_day * 8):
        train_end_index = i + (images_per_day * 4)  # Train for 2 days
        valid_end_index = train_end_index + images_per_day*2  # Validation for 1 day
        test_end_index = valid_end_index + images_per_day*2  # Test for 1 day

        df.loc[i:train_end_index - 1, "Set"] = "Train"  # Exclude train_end_ index
        df.loc[train_end_index:valid_end_index - 1, "Set"] = "Valid"  # Exclude valid_end_index
        df.loc[valid_end_index:test_end_index - 1, "Set"] = "Test"  # Exclude test_end_index
        
    if file_name:
        df.to_csv(file_name, index=False)
    return df


if __name__ == "__main__":
    make_image_csv('/Users/sb/Desktop/anomaly_forecast/data/radar_sample_images','loader_test.csv')