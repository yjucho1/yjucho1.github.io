---
title: "시계열 분석 - part3"
categories: 
  - Spatio-Temporal Data
comments: true
mathjax : true
published: true

---

Part1에서는 stationary를 가정으로, 시계열 데이터의 기본 모델인 AR과 MA에 대해서 알아보고 모델의 파라미터를 추정하기 위해서 Yule-Walker Equations을 알아보았습니다.  또한 모델의 order를 결정하기 위해 ACF, PACF를 이용하는 방법을 살펴보았습니다. 실제 분석에서 모델의 파라미터를 추정하는 것은 소프트웨어를 이용해 자동적으로 계산되기 때문에 하나 하나를 기억할 필요는 없지만, 그 원리에 대해서는 이해하고 있는 것이 좋습니다. Part2에서는 실제 데이터를 이용해 statonary 가정을 검증하고, 적합한 모델을 찾고 모델의 order를 결정하는 방법, 추정한 모델이 적합한지 검증하는 방법들을 알아보았습니다. AIC, BIC, HQIC와 같은 지표를 사용하여 여러가지 모델 중 더 나은 성능의 모델을 선택할 수 있고, 모델의 residual을 이용해 모델을 진단하는 것을 직접 수행해보았습니다. 또한 최종 선택한 모델을 이용해 미래 값을 예측(forecast)한 결과를 시각화하고 MSE를 이용해 평가해보기도 하였습니다.

이번 Part3에서는 non-stationary 시게열 데이터를 모델링하는 방법들에 대해서 이야기해보도록 하겠습니다. 앞서 포스팅에서 살펴보았듯이 주어진 데이터 $${x_1, …, x_n}$$에 대한 시계열 차트를 그렸을때, (a) stationarity와 비교하여 명확한 편차 변화가 보이지 않고 (b) ACF가 점차 감소하는 모습을 보이면 (평균을 0로 맞춘 후) ARMA를 사용하면 됩니다. 그렇지 않은 경우, 주어진 데이터를 변환시켜 (a)와 (b) 특정을 갖는 새로운 시계열 데이터를 생성하는 방법을 사용해야합니다. 이 때 주로 사용하는 방법은 “differencing(차분)”으로, differencing한 새로운 시리즈 $${y_1, … y_n}$$가 ARMA를 따를 때, ARIMA 프로세스를 따른다고 표현합니다. 

A generalization of this class, which incorporates a wide range of nonstationary series, is provided by the ARIMA processes, i.e., processes that reduce to ARMA processes when differenced finitely many times

<b>Definition</b>

If d is a nonnegative integer, then $${X_t}$$ is an ARIMA(p, d, q) process if $$Y_t := (1-B)^dX_t$$ is a causal ARMA(p, q) process.

 
참고로 B는 backward shift operator를 의미합니다. $$BX_t = X_{t-1}$$로 $$\nabla X_t = X_t - X_{t-1} = (1-B) X_t$$로 표기합니다.

이해를 돕기 위해서 ARIMA(1, 1, 0) process를 따르는 데이터를 생성한후, sample ACF, sample PACF를 살펴보도록 하겠습니다. statsmodels의 arma_generate_sample를 이용해 ARMA(1,0) 프로세스를 따르는 데이터를 생성한 후($$\phi$$ = 0.8), cumsum()을 이용하면 ARIMA(1,1,0) 프로세스를 따르는 데이터를 생성할 수 있습니다. 

```python
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
n = 200

alphas = np.array([0.8])
ar = np.r_[1, -alphas]
ma = np.r_[1]

arma10 = sm.tsa.arma_generate_sample(ar=ar, ma=ma, nsample=n)
arima110 = arma10.cumsum()
```

이 인공 데이터의 sample ACF와 sample PACF를 나타내면 아래와 같습니다. 
```
plt.plot(arima110)
plt.show()
sm.tsa.graphics.plot_acf(arima110, lags=50)
sm.tsa.graphics.plot_pacf(arima110, lags=50)
plt.show()
```

<img src = "/assets/img/2018-12-31/output_0_0.png">
<img src = "/assets/img/2018-12-31/output_1_0.png">
<img src = "/assets/img/2018-12-31/output_1_1.png">

위 그림에서도 알 수 있듯이 ARIMA 모델의 적합성을 나타내는 가장 큰 증거는 양의 sample ACF가 천천히 감쇠하는 모습을 띄는 것입니다. 시계열 데이터가 주어지고, 주어진 데이터에 적합한 모델을 찾기 위해서 가장 먼저 할일은 데이터가 ARMA 프로세스와 대응되도록(sample ACF가 급격히 감소하는 모습이 보일때까지)  $$\nabla = 1-B$$ 오퍼레이터를 적용하하는 것입니다. 실제로 위의 인공데이터에서 $$\nabla$$ 오퍼레이터를 한번 적용한 후 다시 시계열 차트와 sample ACF, sample PACF를 그려본 결과는 아래와 같습니다. (앞의 것에 비해서 ACF가 더 급격히 감소하는 것을 볼 수 있습니다)

```
plt.plot(arma10)
plt.show()
sm.tsa.graphics.plot_acf(arma10, lags=50)
sm.tsa.graphics.plot_pacf(arma10, lags=50)
plt.show()
```
<img src = "/assets/img/2018-12-31/output_2_0.png">
<img src = "/assets/img/2018-12-31/output_2_1.png">
<img src = "/assets/img/2018-12-31/output_2_2.png">

 
```python
arima_mod110 = sm.tsa.ARIMA(arima110, (1,1,0)).fit(trend='nc')
print(arima_mod110.summary())
print('variance :', arima_mod110.resid.var())
```

                                 ARIMA Model Results                              
    ==============================================================================
    Dep. Variable:                    D.y   No. Observations:                  199
    Model:                 ARIMA(1, 1, 0)   Log Likelihood                -270.813
    Method:                       css-mle   S.D. of innovations              0.941
    Date:                Mon, 31 Dec 2018   AIC                            545.626
    Time:                        06:49:49   BIC                            552.212
    Sample:                             1   HQIC                           548.291
                                                                                  
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1.D.y      0.8085      0.041     19.576      0.000       0.728       0.889
                                        Roots                                    
    =============================================================================
                     Real           Imaginary           Modulus         Frequency
    -----------------------------------------------------------------------------
    AR.1            1.2369           +0.0000j            1.2369            0.0000
    -----------------------------------------------------------------------------
    variance : 0.8895674306182718

* Estimated parameters from data : 
$$(1-0.8085)(1-B)X_t = Z_t, \ \ \ \ \ \ \ \{Z_t\} \sim WN(0, 0.8798)$$
* True generating Process : 
$$(1-0.8)(1-B)X_t = Z_t, \ \ \ \ \ \ \ \ \ \ \ \ \{Z_t\} \sim WN(0, 1)$$

위의 두 결과가 거의 비슷한 것을 볼 수 있습니다. 지금까지 예시에서 본 것처럼 ARIMA 모델을 추정하는 방법은 differencing을 적용하는 것을 제외하고는 ARMA와 비슷한 맥략을 유지합니다. 


### Identification Techniques

<b> (a) Preliminary Transformations </b> 
ARMA 모델을 사용하기 전에 주어진 데이터가 stationarity 가정에 더 부합한 새로운 시리즈로 변환할 필요성이 있는지 검토해야합니다. 예를 들어 스케일에 대한 의존도가 있는 경우, 로그변환을 취해서 스케일에 대한 의존성을 제거해야합니다. 또한 시계열 차트에서 전체적으로 우상향/우하향하는 추세(trend)나 주기적으로 반복되는 패턴(seasonality)이 발견될 수 있습니다. 이런 경우 2가지 방법으로 처리할 수 있습니다.

1) Classical decomposition - trend component, seasonal component, random residual component

2) Differencing - Wine sale의 예시에서는 12개월 주기로 반복되는 패턴이 존재하므로, $${(1-B^{12})V_t}$$를 적용

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.api as sm

df = pd.read_csv('monthly-australian-wine-sales-th.csv', skiprows=1, header=None, encoding='euc-kr')
u_t = df[1]
v_t = np.log(u_t)
v_t_st = v_t.diff(12)

fig = plt.figure(figsize=(12,10))
fig.add_subplot(311)
plt.plot(df[1], marker='o')
plt.legend(['$U_t$'])
fig.add_subplot(312)
plt.plot(v_t, c='orange', marker='o')
plt.legend(['$V_t = lnU_t$'])
fig.add_subplot(313)
plt.plot(v_t_st, c='green', marker='o')
plt.legend(['$V_t^* = (1-B^{12})V_t$'])
plt.suptitle('Monthly Australian Wine Sales')
plt.show()
```

<img src = "/assets/img/2018-12-31/output_6_0.png">

<b> (b) Identification and Estimation </b> 전처리 변환을 통해서 stationarity 가정에 부합하는 시리즈를 얻을 후에는 적합한 ARMA(p, q)를 찾습니다. p와 q를 먼저 결정해야하는데, 보통 p와 q를 크게 설정할 수록 추정된 모델의 성능이 더 좋은 것으로 나타납니다. 추정해야하는 파라미터 갯수가 많아질수록 패널티가 증가하도록 AIC 등과 같은 지표를 사용하는 것이 바람직합니다. 또한 모델을 선택한 후에는 residual이 white noise가 맞는지 검증하는 작업을 수행해야합니다. 이에 대한 내용은 part2에서 이미 살펴보았으니 이번에는 자세히 다루지 않겠습니다.


```python
stationary_series = np.array(v_t_st.dropna())
result = sm.tsa.stattools.arma_order_select_ic(stationary_series, ic=['aic', 'bic'], trend='nc')
## result.aic_min_order = (2,2)
## result.bic_min_order = (1,1)

wine_model = sm.tsa.ARMA(np.array(v_t_st.dropna()), result.aic_min_order).fit(trend='nc')
print(wine_model.summary())
print('variance :', wine_model.resid.var())
```

                                ARMA Model Results                              
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                  175
    Model:                     ARMA(2, 2)   Log Likelihood                  95.322
    Method:                       css-mle   S.D. of innovations              0.139
    Date:                Mon, 31 Dec 2018   AIC                           -180.643
    Time:                        20:50:06   BIC                           -164.819
    Sample:                             0   HQIC                          -174.225
                                                                                
    ==============================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1.y        0.0304      0.055      0.556      0.579      -0.077       0.138
    ar.L2.y        0.8740      0.047     18.621      0.000       0.782       0.966
    ma.L1.y        0.2052      0.086      2.398      0.018       0.038       0.373
    ma.L2.y       -0.7948      0.085     -9.367      0.000      -0.961      -0.628
                                        Roots                                    
    =============================================================================
                    Real          Imaginary           Modulus         Frequency
    -----------------------------------------------------------------------------
    AR.1            1.0524           +0.0000j            1.0524            0.0000
    AR.2           -1.0872           +0.0000j            1.0872            0.5000
    MA.1           -1.0000           +0.0000j            1.0000            0.5000
    MA.2            1.2582           +0.0000j            1.2582            0.0000
    -----------------------------------------------------------------------------
    variance : 0.019416898661637372
