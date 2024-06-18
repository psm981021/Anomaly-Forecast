#  <img src = "https://github.com/psm981021/Anomaly-Forecast/blob/main/images/Capstone_Logo.jpg" width = "15%" > [Anomaly-Forecast]

### 💡 [Background]
#### 24-1학기 캡스톤 프로젝트

##### 훈련 샘플이 적은 이상치 예측 방안 개발

+ 일반적인 인공지능 모델은 빅데이터 학습을 통해 규칙을 찾아내고 성능을 확보한다.
+ 특이한 사례(전례없는 집중호우, 기계의 고장 등)는 많은 피해를 발생시키고 문제가 될 수 있으나, 샘플이 적어 일반적인 인공지능 모델로 다루기 어렵다.
+ 이상치를 예측하거나 다룰 수 있는 방법론을 공부하고 실제 데이터에 적용하여 분석모델의 성능 향상을 증명하거나 활용성을 제시한다.

### ⛈ [목표]
특정 지역에서 **집중호우**발생 시의 강수량을 예측하여 실제값 과의 차이를 최소화 할 수 있도록 모델을 설계한다.

### 👫 [Team]

+ [박성범](https://github.com/psm981021)
+ [이찬빈](https://github.com/coldbeeen)
+ [김소정](https://github.com/Soojeoong)
+ [박지원](https://github.com/woney23)

### 📊 [Data]

#### 데이터 수집
+ 방재기상관측 (AWS)
    + 전국 554개 지점의 2021 ~ 2023년 간 **시간별** 측정 데이터
+ 레이더 강수량 데이터
    + 기상 API 허브에서 2021 ~ 2023년 **10분** 간격으로 측정한 레이더 이미지 데이터셋

#### 데이터 전처리
<!-- <p align="center"> <img src = "https://github.com/psm981021/Anomaly-Forecast/blob/main/images/Data_Map.png" width = "40%" ></p> -->

##### 🇰🇷 지점 선정 기준 

+ 전국 지점에 대해 예측을 진행하기 앞서 날씨 예보에 사용되는 특정 지점들을 임의로 선정하였다.
+ 제주도, 백령도, 울릉도 등은 섬이라는 특성이 내륙과 많은 부분이 상이하여 **지점 선정에서 제외**하였다.
+ **서울 관악** 지점을 통해 실험을 진행하고, 추가로 모델의 일반화를 보이기 위해 **강원도 철원** 지점에 대해 실험을 진행하였다.


##### ☔ 이상치 선정 및 라벨링
+ 같은 양의 비가 내리더라도 어떤 지역에서는 평소와 같은 양일 수 있지만, 어떤 지역에서는 굉장히 많이 오는 양일 수 있다.
+ 비가 자주 오는 지역에서는 상대적으로 강수 현상에 대비가 되어 있지만, 비가 자주 오지 않는 지역에서는 그렇지 않다.
+ 따라서 절대적으로 보지 않고, **상대적으로** 보기로 결정하였다.
+ 비의 강도에 따라 **Light** (상위 50% 이하), **Normal** (상위 10%~50%), **Heavy** (상위 10% 이상)으로 할당해주었다.

##### ⛏ 데이터셋 구축

<table>
  <tr>
    <td>
      <img src="https://github.com/psm981021/Anomaly-Forecast/blob/main/images/Build_Dataset.png" width="200">
    </td>
    <td>
      <p>
        1. Timestamp는 1시간의 주기를 가진다.<br>
        2. 각 Timestamp t에 대해 t-1과의 rain값의 차이를 gap에 저장한다.<br>
        3. 각 rain에 대하여 비의 강도에 따라 Light, Normal, Heavy로 label을 할당해준다.<br>
        4. 강수량이 0.1mm 미만인 Timestamp는 비가 오지 않은 것으로 간주, 제외해준다.<br>
        5. 데이터셋을 Train/Valid/Test 로 나눠준다.<br>
          &nbsp;&nbsp;&nbsp;&nbsp;   Train : 4개의 연속된 Timestamp<br>
          &nbsp;&nbsp;&nbsp;&nbsp;   Valid : Train 뒤에 이어지는 2개의 연속된 Timestamp (2시간)<br>
          &nbsp;&nbsp;&nbsp;&nbsp;   Test : Heavy Rain에 속하는 Timestamp 중 랜덤하게 20개 선정<br>
      </p>
    </td>
  </tr>
</table>



### 📝 [Features]

#### FrameWork 1

<p align="center"> <img src = "https://github.com/psm981021/Anomaly-Forecast/blob/main/images/Framework_1.png" width = "80%" ></p>

+ (Pre-train)
    + 전체적인 분포에 대한 학습, 원본 + 증강(데이터)
+ (Fine-tuning)
    + 실제 예측, 미리 선정한 지점을 사용 (원본 데이터)

##### 실험 (Baseline)

+ 습도, 기온, 풍향, 풍속, 강수량, 강수량 Lag(1,2,3)을 사용하였다.
+ 랜덤 포레스트, 선형 회귀 모델, DL 모델을 사용하였다. 

##### 실험 (Baseline) 결과

<p align="center"> <img src = "https://github.com/psm981021/Anomaly-Forecast/blob/main/images/Baseline_Result.png" width = "80%" ></p>

+ Autoregressive한 성향이 지배적임을 알 수 있다.
+ 평균에 근접하게 예측하는 현상을 보인다.
    


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



#### FrameWork 2 (Fourcaster Overview)

<p align="center"> <img src = "https://github.com/psm981021/Anomaly-Forecast/blob/main/images/Fourcaster_Overview.png" width = "100%" ></p>

+ Autoregressive와 예측 결과가 평균값 근처에 머무는 한계를 극복하기 위한 새로운 FrameWork

<p align="center"> <img src = "https://github.com/psm981021/Anomaly-Forecast/blob/main/images/Fourcaster_Overview2.png" width = "100%" ></p>

##### Rainfall Movement Generator

+ 10분 점의 레이더 이미지를 입력으로 받아 해당 이미지의 10분 후를 예측한 이미지를 생성한다.
+ 10분 뒤 시점을 생성한 이미지와 실제 이미지를 비교하여 모델을 업데이트 한다.

##### Heavy Rain Estimator

+ 생성된 이미지들 사이의 Gap을 계산한 뒤, Intensity Classifier를 사용하여 비의 강도를 분류하고 지정된 Expert가 훈련되도록 한다.
+ 각각의 Experts는 Light Rain, Normal Rain, Heavy Rain만 보도록 설계되었다.
+ MoE(Mixture of Experts) 구조는 강수량 예측 분야의 한계점 중 하나인 **Underprediction** 현상을 극복하고자 고안되었다.

#### 실험 1 (Generation Loss)

<p align="center"> <img src = "https://github.com/psm981021/Anomaly-Forecast/blob/main/images/Generation_Loss_Experiment.png" width = "100%" ></p>

+ 파란색이 Ground Truth
+ 본 프로젝트에서 설계한 Balancing Loss가 가장 좋은 성능을 보였다.

#### 실험 2 (Regression with MoE)

<p align="center"> <img src = "https://github.com/psm981021/Anomaly-Forecast/blob/main/images/MoE_Experiment.png" width = "80%" ></p>

+ Single Linear 보다 MoE를 사용한 경우 성능이 가장 높음을 확인할 수 있다.
+ MoE 구조를 통해 **Underprediction** 현상을 해결할 수 있음을 보인다.

#### 실험 3 (Forecasting Different Region)

<p align="center"> <img src = "https://github.com/psm981021/Anomaly-Forecast/blob/main/images/Experiment_3.png" width = "100%" ></p>

+ 강원도 철원 지점에 대한 실험을 통해 모델의 일반화를 보인다.
+ 이전 실험에서의 결과를 바탕으로 Balancing Loss, MoE 구조를 사용하여 실험한다.

### 🏃 [Conclusion & Insights]

<p align="center"> <img src = "https://github.com/psm981021/Anomaly-Forecast/blob/main/images/Conclusion.png" width = "80%" ></p>


### 📚 [Stack]

+ Python 
+ Pytorch 
+ Numpy 
+ Pandas
