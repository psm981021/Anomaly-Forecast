import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math
from sklearn.model_selection import GridSearchCV

# random_forest
def random_forest_regession(params, train_x, train_y, test_x, test_y, cv):
    # train_x, train_y,test_x, test_y는 dataframe 모양으로 
    rf=RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_cv=GridSearchCV(rf, param_grid=params,cv=cv,n_jobs=-1)
    grid_cv.fit(train_x, train_y)
    print("GridSearch best Params : ",grid_cv.best_params_)
    train_predict = grid_cv.best_estimator_.predict(train_x)
    test_predict = grid_cv.best_estimator_.predict(test_x)
    print("train_predict RMS:{}".format(math.sqrt(mean_squared_error(train_predict,train_y))))
    print("test_predict RMS:{}".format(math.sqrt(mean_squared_error(test_predict,test_y))))

    # 그래프 - mse 관점에서 
    plt.plot(test_y.reset_index(drop=True))
    plt.plot(test_predict)
    plt.savefig("graph1(mse)_randomforest.png")
    # plt.show()

    # important feature
    # 한글 폰트 맞줘주는 코드 (필요시 사용)
    plt.rcParams['font.family'] ='Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] =False

    rf_importances_values=grid_cv.best_estimator_.feature_importances_
    rf_importances=pd.Series(rf_importances_values, index=train_x.columns)
    rf_top=rf_importances.sort_values(ascending=False)

    plt.figure(figsize=(8,8))
    sns.barplot(x=rf_top, y=rf_top.index)
    plt.savefig("graph2(importance_feature)_randomforest.png")
    # plt.show()

