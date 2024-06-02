# dataset.py
import pandas as pd
import numpy as np

# 주어진 DataFrame에 푸리에 변환을 수행하여 새로운 특성 생성
def fourier(df, name, total):
    """    
    Parameters:
        df (DataFrame): 변환을 수행할 데이터프레임
        name (str): 변환할 열의 이름
        total (float): 변환에 사용할 주기(총 주기)
    
    Returns:
        DataFrame: 변환된 데이터프레임
    """
    # total은 float형으로 입력할 것 
    feature1 = 'sin_' + name + '_1'
    feature2 = 'con_' + name + '_1'
    df[feature1] = np.sin(2*np.pi*df[name]/total)
    df[feature2] = np.cos(2*np.pi*df[name]/total)
    df = df.drop(name, axis=1)
    return df

# 주어진 DataFrame에 지연 특성 생성
def lag_maker(df, lag_num, name):
    """   
    Parameters:
        df (DataFrame): 변환을 수행할 데이터프레임
        lag_num (int): 생성할 지연 특성의 개수
        name (str): 지연할 열의 이름
    
    Returns:
        DataFrame: 변환된 데이터프레임
    """
    for i in range(lag_num):
        feature = 'Lag_' + str(i+1)
        df[feature] = df[name].shift(i+1)
    return df

