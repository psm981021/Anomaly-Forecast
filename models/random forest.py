import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
from sklearn.metrics import *
from sklearn.model_selection import train_test_split


def dataset(filepath):
    try:
        dataframe = pd.read_csv(filepath)
        
        #change to datetime
        dataframe['일시'] = pd.to_datetime(dataframe['일시'])

        #change column names
        new_column_names = {
            '일시': 'date',
            '풍속(m/s)': 'Wind',
            '강수량(mm)': 'Precipitation',
            '풍향(deg)': 'Wind_Direction',
            '기온(°C)': 'Temperature'
        }
        dataframe = dataframe.rename(columns=new_column_names)

    except Exception as e:
        # raise exception for 
        print("Error", e)

    return dataframe

def random_forest(dataframe, n_estimators, random_state,x_variables, y_variables, split):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    x,y = dataframe[x_variables],dataframe[y_variables]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=split, random_state=random_state)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return [mean_squared_error(y_test, y_pred),mean_absolute_error(y_test, y_pred),root_mean_squared_error(y_test, y_pred)]


    






if __name__ == "__main__":
    path = '/Users/sb/Desktop/anomaly_forecast/data/test_data.csv'
    dataframe = dataset(path)
    mse,mae,rmse = random_forest(dataframe, 100, 42,['Wind','Wind_Direction','Temperature'],'Precipitation',0.2)
    
    print("mse:", mse)
    print("mae:", mae)
    print("rmse:", rmse)
    pass

