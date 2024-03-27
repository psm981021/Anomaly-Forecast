"""

머신러닝 계열 모델들을 여기에 다 모으면 좋을 것 같음. 

"""
import dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import random_forest_regession

"""---data preprocessing---"""

data=pd.read_csv("광진구_aws_data.csv")
data=data.set_index('일시')
data=dataset.fourier(data,'풍향(deg)',360.0)
data=dataset.lag_maker(data,3,'강수량(mm)')
data=data.fillna(0)
print("data preprocessing done")
# lag 만들면서 생긴 결측치는 일단 0으로 처리함. 

"""---train/test 나누기"""
# 나중에는 valid도 나눠줘야 함 

train=data[:-8755] # 그 앞에 9년 동안의 데이터 
test=data[-8755:] # 2023년도를 test 데이터로 넣음

train_y=train['강수량(mm)']
train_x=train.drop("강수량(mm)",axis=1)

test_y=test['강수량(mm)']
test_x=test.drop('강수량(mm)',axis=1)

print("train/test split done")

"""---model RUN---"""
# random forest 용 파라미터 
rf_params = {
    'n_estimators' : (100,200),
    'max_depth' : (5,8),
    'min_samples_leaf' : (8,18),
    'min_samples_split' :(8,16)
}
# 랜덤포레스트 실행 
random_forest_regession(params=rf_params, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, cv=2)

print("random forest done")




