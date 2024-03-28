# model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *
import math
from sklearn.model_selection import GridSearchCV

# 랜덤 포레스트 회귀 모델
def random_forest_regression(params, train_x, train_y, test_x, test_y, cv):
    """
    Parameters:
        params (dict): 랜덤 포레스트 모델의 하이퍼파라미터
        train_x (DataFrame): 학습 데이터의 독립 변수
        train_y (Series): 학습 데이터의 종속 변수
        test_x (DataFrame): 테스트 데이터의 독립 변수
        test_y (Series): 테스트 데이터의 종속 변수
        cv (int): 교차 검증의 폴드 수
    """
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_cv = GridSearchCV(rf, param_grid=params, cv=cv, n_jobs=-1)
    grid_cv.fit(train_x, train_y)
    print("GridSearch best Params : ", grid_cv.best_params_)
    
    train_predict = grid_cv.best_estimator_.predict(train_x)
    test_predict = grid_cv.best_estimator_.predict(test_x)
    print("train_predict RMSE:{}".format(math.sqrt(mean_squared_error(train_predict, train_y))))
    print("test_predict RMSE:{}".format(math.sqrt(mean_squared_error(test_predict, test_y))))
    
    # 그래프 
    x_label = pd.to_datetime(test_y.index)
    
    plt.figure(figsize=(10, 8))
    plt.plot(x_label, test_y, label='actual')
    plt.plot(x_label, test_predict, label='predicted')
    plt.title('Actual vs Predicted Precipitation of Random Forest Regression Model')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig("result_image/graph1(mse)_randomforest.png")

    # important feature
    plt.figure(figsize=(10, 6)) 
    # 한글 폰트 맞줘주는 코드 (필요시 사용)
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    rf_importances_values = grid_cv.best_estimator_.feature_importances_
    rf_importances = pd.Series(rf_importances_values, index=train_x.columns)
    rf_top = rf_importances.sort_values(ascending=False)
    
    sns.barplot(x=rf_top, y=rf_top.index)
    plt.title('Importance Features of Random Forest Regression Model')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.savefig("result_image/graph2(importance_feature)_randomforest.png")

# 선형 회귀 모델 
def linear_regression(train_x, train_y, test_x, test_y, cv, params={}):
    """
    Parameters:
        train_x (DataFrame): 학습 데이터의 독립 변수
        train_y (Series): 학습 데이터의 종속 변수
        test_x (DataFrame): 테스트 데이터의 독립 변수
        test_y (Series): 테스트 데이터의 종속 변수
        cv (int): 교차 검증의 폴드 수
        params (dict): 선형 회귀 모델의 하이퍼파라미터, 기본값은 빈 딕셔너리
    """
    lr = LinearRegression()
    lr.fit(train_x, train_y)
    train_predict = lr.predict(train_x)
    test_predict = lr.predict(test_x)
    
    print("test_predict RMSE:{}".format(math.sqrt(mean_squared_error(test_predict, test_y))))
    print("test predict MSE:{}".format(mean_squared_error(test_predict, test_y)))
    print("test predict MAE:{}".format(mean_absolute_error(test_predict, test_y)))
    
    # coef 계산
    coef_values = lr.coef_
    coef = pd.Series(coef_values, index=train_x.columns)
    coef_top = coef.sort_values(ascending=False)
    
    # 그래프 - 계수(coef_) 시각화
    plt.figure(figsize=(10, 6))
    sns.barplot(x=coef_top, y=coef_top.index)
    plt.title('Coefficient of Linear Regression Model')
    plt.xlabel('Coefficient')
    plt.ylabel('Features')
    plt.savefig("result_image/graph2(importance_feature)_linear_regression.png")
    
    # 그래프 
    x_label = pd.to_datetime(test_y.index)
    
    plt.figure(figsize=(10, 8))
    plt.plot(x_label, test_y, label='actual')
    plt.plot(x_label, test_predict, label='predicted')
    plt.title('Actual vs Predicted Precipitation of Linear Regression Model')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()


    plt.savefig("result_image/graph1(mse)_linear_regression.png")
