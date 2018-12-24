---
title: "시계열 분석 - 실제 데이터"
categories: 
  - Spatio-Temporal Data
comments: true
mathjax : true
published: false

---

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

시계열 데이터에서 trend와 seasonality 성분을 추정하는 것을 decompose라고 합니다. 이론적인 내용은 다음 포스팅에서 다루도록하고, 여기서는 결과값만 확인하도록 하겠습니다. statsmodels라는 패키지에서 seasonal_decompose를 사용할수 있습니다. 참고로 seasonal_decompose의 trend estimation은 moving window 방식을 이용합니다. 

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
