import zipfile
import pandas as pd
import os
import shutil

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

    path_to_zipfile = 'data_gangwon_only.zip'
    directory_to_unzip = ''

    with zipfile.ZipFile(path_to_zipfile, 'r') as zip_ref:
        zip_ref.extractall(directory_to_unzip)

