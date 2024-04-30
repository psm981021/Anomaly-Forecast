import zipfile

if __name__ == "__main__":

    path_to_zipfile = '/Users/sb/Desktop/anomaly_forecast/data/radar_sample.zip'
    directory_to_unzip = '/Users/sb/Desktop/anomaly_forecast/data/radar_sample_images'

    with zipfile.ZipFile(path_to_zipfile, 'r') as zip_ref:
        zip_ref.extractall(directory_to_unzip)
    