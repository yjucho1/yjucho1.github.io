---
title: "시계열 분석 - part2"
categories: 
  - Spatio-Temporal Data
comments: true
mathjax : true
published: true

---

지난 [포스팅]({% post_url 2018-12-18-time-series-part1 %})에서는 시계열 데이터를 모델링하기 위한 모델과 모델의 파라미터를 추정하기 위한 이론적 배경을 살펴보았습니다. 

stationary하고 causal하다는 가정 하에서 Auto-Regressive 또는 Moving average, 또는 두가지가 섞인 ARMA 모델을 사용할 수 있고, 모델의 order를 결정하기 위해서 ACF와 PACF를 사용하는 방법을 학습하였습니다. 

실제 분석 과정에서는 적합한 모델을 선택하기 위해서는 ACF, PACF 이외에 여러가지 방법을 복합적으로 사용합니다. 예를 들어 시계열 모델을 추정한 후 추정값과 실제값의 차이로 residual을 계산합니다. residual이 정규분포를 따르는 white noise라는 가정을 검증함으로써 모델이 적합한지 확인할 수 있습니다.  이를 위해 q-q plot 등을 사용합니다. 또한 이론적인 예측력을 확인하기 위해서 AIC(Akaike Information Criterion), BIC(Bayesian information criterion)와 같은 지표를 살펴봅니다. 데이터에 기반해 실험적인 예측력을 위해 cross validation기법 등을 사용할 수도 있습니다. 

정리하면, 실제 분석 과정은 이와 같은 검증 기법들을 사용하여 적합한 모델을 규명하고, unknown 파라미터를 추정하는 방법을 반복하여 점점 더 좋은 성능의 모델을 찾아내는 방식으로 진행됩니다.  

> Model Building <br>
> 1) identify model <br>
> 2) estimate unknowns <br>
> 3) diagonstic checking <br>
> 4) prediction <br>

for 1), 3) use :
* ACF, PACF
* for large - n cases, Box-Ljung test, Sign test, Rank test, q-q plot .. => for the residuals after fitting the model
* theoretical predictive power : AIC, BIC 
* empirical predictive power : Cross validation

이번 포스팅에서는 AIC와 BIC에 대해서 보다 자세히 알아보도록 하겠습니다. 


