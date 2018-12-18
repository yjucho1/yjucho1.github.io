---
title: "시계열 분석 - part1"
categories: 
  - time series
comments: true
mathjax : true
published: true

---

시계열 데이터는 일정 시간동안 수집된 일련의 데이터로, 시간에 따라서 샘플링되었기 때문에 인접한 시간에 수집된 데이터는 서로 상관관계가 존재하게 됩니다. 따라서 시계열 데이터를 추론하는데, 전통적인 통계적 추론이 적합하지 않을수 있습니다. 현실 세계에서는 날씨, 주가, 마케팅 데이터 등등 다양한 시계열 데이터가 존재하고, 시계열 분석을 통해 이러한 데이터가 생성되는 메카니즘을 이해하여 설명가능한 모델로서 데이터를 표현하거나, 미래 값을 예측하는 것, 의미있는 신호값만만 분리하는 일에 활용할 수 있습니다.

일반적인 시계열분석의 과정은 아래와 같습니다.

* Plot the series and examine the main features of the graph, checking in particular whether there is
  * A trend
  * A seasonal component
  * Any apparent sharp changes in behavior
  * Any outlying observations
* Remove the trend and seasonal components to get stationary residuals 
  * To achieve goal, you may need to apply a preliminary transformation like taking logarithms
  * Estimate the trend and seasonal components using classical decomposition model
  * Or Eliminate the trend and seasonal components using differencing
  * Anyway, the goal is to get stationary residuals
* Choose a model to fit the residuals
* Forecasting the residuals and then inverting the transformations for original series
* Alternative approach is to express the series in terms of its Fourier components. But it will not be discussed in here.

시계열 분석은 시간의 경과에 따라 변하지 않는 어떤 특성을 가진 프로세스를 분석하는 것입니다. 만약 시계열 데이터를 예측하고자 한다면, 우리는 시간에 따라 변하지 않는 무언가가 있다는 것을 반드시 가정해야합니다. 예를 들면 평균이 변하지 않는것(no trend), 분산이 변하지 않고, 주기적인 패턴이 없는 데이터를 가정합니다. 이러한 속성을 "stationary"하다고 합니다. 어떤 시계열 데이터가 stationary하다는 가정을 하면, 우리는 다양한 기법들을 활용할수 있습니다. 앞으로의 포스팅에서는 이러한 기법들이 무엇인지 소개하려고 합니다. 

### Stationary, Auto-Covariance Function and Auto-Correlation Function

시계열 $$\{X_t, t=0, \pm 1, ...\}$$ 가 stationary하다는 것은 h lag만큼 time-shifted 된 시계열 $$\{X_{t+h}, t=0, \pm 1, ...\}$$와 통계적인 속성이 유사하다는 것을 의미합니다. 이 때 통계적인 속성을 first and second order moments(mean and covariance)로만 제한하면, 아래와 같이 수식적 정의를 할 수 있습니다.


<b>Definition<b>

Let $${X_t}$$ be a time series with $$E(X_t^2) < \infty$$. 

The mean function of $${X_t}$$ is $$\mu_X(t) = E(X_t)$$. 

The covariance function of $${X_t}$$ is $$\rho_X(r, s) = Cov(X_r, X_s) = E[(X_r - \mu_X(r))(X_s - \mu_X(s))]$$ for all integers r and s. 

<b>Definition</b>

$${X_t}$$ is (weakly) stationary if 

1) $$\mu_X(t)$$ is independent of t.

2) $$\gamma_X(t+h, t)$$ is independent of t for each h. 

Strictly stationary is also

3) $$(X_1, ..., Xn)$$ and $$(X_{1+h}, ..., X_{n+h})$$ have the same joint distributions for all h and n >0

2)에서의 정의를 이용해 stationary한 타임시리즈의 $$\gamma_X(t+h, t)$$는 t에 대해서 무관하기 때문에 $$\gamma(\cdot)$$는 "autocovariance function"으로 부르며, $$\gamma_X(h)$$는 h lag에서의 값을 지칭하는 것으로 하겠습니다.

<b>Definition</b>

Let $${X_t}$$ be a stationary time series. 

The autocovariance function (ACVF) of $${X_t}$$ at lag h is $$\gamma_X(h) = Cov(X_{t+h}, X_t)$$. 

The autocorrelation function (ACF) of $${X_t}$$ at lag h is $$\rho_X(h) \equiv \frac{\gamma_X(h)}{\gamma_X(0)} = Cor(X_{t+h}, X_t)$$


주어진 시계열 데이터의 mean과 covariance를 추정하기 위해 sample mean과 sample covariance function, sample autocorrelation function를 사용합니다.  

<b>Definition</b>

Let $$x_1, ..., x_n$$ be observations of a time series. 

The sample mean of $$x_1, ..., x_n$$ is
$$\bar{x} = \frac{1}{n} \sum_{t=1}^n x_t$$

The sample autocovariance function is
$$\hat{\gamma} := n^{-1} \sum_{t=1}^{n-|h|}(x_{t+|h|} - \bar{x})(x_{t} - \bar{x}), -n < h < n$$

The sample autocorrelation function is
$$\hat{\rho} := \frac{\hat{\gamma}(h)}{\hat{\gamma}(0)}, -n < h < n$$

 
### 실제 데이터 - 초미세먼지 농도

* 데이터 : 서울 용산구 한강대로 405(서울역 앞)의 초미세먼지 농도 
* 기간 : 	2018-11-18 19:00 ~ 2018-12-18 17:00 (1시간 단위)

<img src="/assets/img/2018-12-18/df.png" width="300">

<img src = "/assets/img/2018-12-18/pm25.png" width="500">

statmodels 패키지를 이용해 ACF를 바로 구할수 있습니다.

```python
from statsmodels.graphics.tsaplots import plot_acf
fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(df.pm25Value, ax=ax)
plt.show()
```
<img src = "/assets/img/2018-12-18/acf.png" width="500">

acf 그래프를 보면 lag h(가로축)가 증가할수록 값이 점차 감소하며, 일정시간 양의 상관관계를 보이다가 음의 상관관계로 변화하는 패턴이 반복되는 것이 나타납니다(periodic). 이는 시계열 데이터가 stationary하지 않고 시간에 의존되는 성질을 가지고 있다는 것을 의미합니다. 

시계열 데이터에서 trend와 seasonality 성분을 추정하는 것을 decompose라고 합니다. 이론적인 내용은 다음 포스팅에서 다루도록하고, 여기서는 결과값만 확인하도록 하겠습니다. statsmodels라는 패키지에서 seasonal_decompose를 사용할수 있습니다.  

```python
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df.pm25Value)
result.plot()
```
<img src = "/assets/img/2018-12-18/decompose.png" width="500">

trend와 seasonality를 제거하고 남은 값(residual)만 가지고 다시 acf를 그려보면 아래와 같습니다.

```python
plot_acf(result.resid.dropna()) 
plot_acf(result.resid.dropna(), lags=50) 
```
<img src = "/assets/img/2018-12-18/residual_acf_all.png" width="500">

original data의 ACF에서 나타났던 주기적인 변화는 거의 없어진 것을 볼수 있습니다. 

<img src = "/assets/img/2018-12-18/residual_acf.png" width="500">

하지만 lags=50 이전 부분을 살펴보면 여전히 lags=3까지 강한 상관계수를 갖고 반복적인 변화패턴을 보이는 것을 알수 있습니다. 이는 residual에 남아있는 패턴이 여전히 미래 값을 예측하는데 도움이 되는 것을 의미합니다. 

### Moving Average and Auto Regressive process 

The MA(q) process:<br>
$$\{X_t\}$$ is a moving-average process of order q if
$$ X_t = Z_t + \theta_1 Z_{t-1} + ... + \theta_q Z_{t-q} $$ <br>
where $$\{Z_t\} ~ White Noise(0, \theta^2)$$ and $$\theta_1, ..., \theta_q$$ are constants.

The AR(q) process:<br>
$$\{X_t\}$$ is a auto-regressive process of order q if
$$ X_t = Z_t + \theta_1 X_{t-1} + ... + \theta_q X_{t-q} $$ <br>
where $$\{Z_t\} ~ White Noise(0, \theta^2)$$ and $$\theta_1, ..., \theta_q$$ are constants.

I can do it!

### ARMA

### Particial ACF

### ARIMA

### 다시 미세먼지

<b>reference</b>

[1] [Introduction to Time Series and Forecasting, Peter J. Brockwell, Richard A. Davis,](https://www.springer.com/us/book/9781475777505)

[2] [Statsmodel's Documentation](https://www.statsmodels.org/dev/index.html)
