import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def fourier(df,name,total):
    # total은 float형으로 입력할 것 
    feature1='sin_'+name+'_1'
    feature2='con_'+name+'_1'
    df[feature1]=np.sin(2*np.pi*df[name]/total)
    df[feature2]=np.cos(2*np.pi*df[name]/total)
    df=df.drop(name, axis=1)

    return df


def lag_maker(df,lag_num,name):

    for i in range(lag_num):
        feature='Lag_'+str(i+1)
        df[feature]=df[name].shift(i+1)
    
    return df
