#  <img src = "https://github.com/psm981021/Anomaly-Forecast/blob/main/images/Capstone_Logo_1.jpg" width = "15%" > [Anomaly-Forecast]
### [Team]: Fourcaster 
### 💡 [Background]
#### 24-1학기 캡스톤 프로젝트

##### 훈련 샘플이 적은 이상치 예측 방안 개발

+ 일반적인 인공지능 모델은 빅데이터 학습을 통해 규칙을 찾아내고 성능을 확보한다.
+ 특이한 사례(전례없는 집중호우, 기계의 고장 등)는 많은 피해를 발생시키고 문제가 될 수 있으나, 샘플이 적어 일반적인 인공지능 모델로 다루기 어렵다.
+ 이상치를 예측하거나 다룰 수 있는 방법론을 공부하고 실제 데이터에 적용하여 분석모델의 성능 향상을 증명하거나 활용성을 제시한다.

##### 목표
모델이 실제 데이터를 통해 Peak값 과의 차이를 최소화 할 수 있도록 학습할 수 있도록 한다.

### 👫 [Team]

+ 박성범 
+ 이찬빈
+ 김소정
+ 박지원

### 📊 [Data]

#### 사용한 데이터
+ 종관기상관측 (ASOS)
    + 전국 103개 지점의 2014 ~ 2023년 간 시간별 측정 데이터
+ 방재기상관측 (AWS)
    + 전국 554개 지점의 2014 ~ 2023년 간 시간별 측정 데이터
+ 레이더 데이터셋 
    + 기상 API 허브에서 2021 ~ 2023년 10분 간격으로 측정한 레이더 이미지 데이터셋

#### 지점 선정 및 이상치 선정
<p align="center"> <img src = "https://github.com/psm981021/Anomaly-Forecast/blob/main/images/Data_Map.png" width = "40%" ></p>

##### 🇰🇷 지점 선정 기준 

+ 전국 지점에 대해 예측을 진행하기 앞서 날씨 예보에 사용되는 특정 지점들을 임의로 선정하였다.
+ 제주도, 백령도, 울릉도 등은 섬이라는 특성이 내륙과 많은 부분이 상이하여 **지점 선정에서 제외**하였다.

##### ☔ 이상치 선정 기준
+ 같은 양의 비가 내리더라도 어떤 지역에서는 평소와 같은 양일 수 있지만, 어떤 지역에서는 굉장히 많이 오는 양일 수 있다.
+ 비가 자주 오는 지역에서는 상대적으로 강수 현상에 대비가 되어 있지만, 비가 자주 오지 않는 지역에서는 그렇지 않다.
+ 따라서 절대적으로 보지 않고, **상대적으로** 보기로 결정하였다.


### 📝 [Features]

#### FrameWork 1

<p align="center"> <img src = "https://github.com/psm981021/Anomaly-Forecast/blob/main/images/Framework_1.png" width = "80%" ></p>

+ (Pre-train)
    + 전체적인 분포에 대한 학습, 원본 + 증강(데이터)
+ (Fine-tuning)
    + 실제 예측, 미리 선정한 지점을 사용 (원본 데이터)

##### 실험 1

+ 습도, 기온, 풍향, 풍속, 강수량, 강수량 Lag(1,2,3)을 사용하였다.
+ 랜덤 포레스트, 선형 회귀 모델을 사용하였다. 
+ 광진구 데이터에 대해 실험을 진행하였다.

##### 실험 1 결과

<p align="center"> <img src = "https://github.com/psm981021/Anomaly-Forecast/blob/main/images/Experiment_1.png" width = "60%" ></p>

+ Autoregressive한 성향이 지배적임을 알 수 있다.
    + 그래프를 확대해보았을 때, 현재의 강수량을 이전 값과 유사하게 예측해버리는 성향이 지배적임을 확인할 수 있었다.
    + Lag 변수를 제거하고, DL 모델을 사용하여 강수량 예측을 위한 실험을 추가적으로 진행해본다.

    + 비가 오지 않는 날이 많아 평균 강수량 값이 0에 근접한다.
    + 비가 오지 않은 날을 데이터에서 제외하고 실험을 추가적으로 진행해본다.

##### 실험 2

+ Transformer, Informer, Autoformer, Linear, DLinear, NLinear 등의 모델을 사용하였다.
+ 실험 1에 Month, Season등의 변수를 추가해 주었다.

##### 실험 2 결과

<p align="center"> <img src = "https://github.com/psm981021/Anomaly-Forecast/blob/main/images/Experiment_2.png" width = "80%" ></p>

+ 성능 지표는 개선되었으나, 여전히 강수량 예측 결과가 평균값 근처에 머무르고 있으므로 유의미하지 않다고 판단하였다.
+ 예측 단계에서 모델이 시계열 데이터 내 변수들의 값을 변수들의 특성에 맞게 고려하지 못 하는 것으로 판단하였다.
+ **시계열 데이터만 사용하는 경우, 강수량 예측에 결정적인 영향을 주는 변수를 찾기 어렵다고 판단하였다.**


##### FrameWork 1 결과

+ 간단한 실험을 통해 초기 프레임워크를 사용하기에는 **명확한 한계**가 존재한다고 판단하였다. 
+ 고품질 데이터셋을 만드는 것에 중점을 두어야 한다.
    + 지형조건, 수증기, 기압, 구름의 분포및 변화과정등 강수 현상에 영향력을 미치는 변수들을 추가해줘야 한다.
    + 시계열 
+ 기존 MSE 평가 지표는 강수량 예측에 있어서 설명력이 부족하다.
    + 평가시, 이상치에 해당하는 값에 대해서만 MSE를 사용하는 것으로 조정한다.


#### FrameWork 2

<p align="center"> <img src = "https://github.com/psm981021/Anomaly-Forecast/blob/main/images/Framework_2.png" width = "40%" ></p>

+ Autoregressive와 시계열 데이터의 한계를 극복하기 위한 새로운 전략 수립

### 📚 [Stack]

+ Python 
+ Pytorch 
+ Numpy 
+ Pandas
