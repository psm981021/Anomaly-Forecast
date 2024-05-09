import zipfile

if __name__ == "__main__":

    path_to_zipfile = '/home/seongbeom/paper/anomaly/data/레이더_22.7_22.9.zip'
    directory_to_unzip = 'radar_test'

    with zipfile.ZipFile(path_to_zipfile, 'r') as zip_ref:
        zip_ref.extractall(directory_to_unzip)
    