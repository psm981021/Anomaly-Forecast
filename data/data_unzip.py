import zipfile

if __name__ == "__main__":

    path_to_zipfile = 'data/레이더 전체 데이터.zip'
    directory_to_unzip = 'data/radar_full'

    with zipfile.ZipFile(path_to_zipfile, 'r') as zip_ref:
        zip_ref.extractall(directory_to_unzip)