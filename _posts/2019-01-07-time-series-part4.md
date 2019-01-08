---
title: "시계열 분석 - part4"
categories: 
  - Spatio-Temporal Data
comments: true
mathjax : true
published: true

---

지금까지는 시계열 데이터가 univariate일 경우를 모델링하는 방법을 알아보았습니다. 이번 포스팅에서는 여러개의 시계열 데이터가 존재할 경우, 즉 multivariate 모델인 Vector AutoRegression (VAR) Model을 알아보도록 하겠습니다.

Bivariate time series에 대해서 알아보도록 하겠습니다. Bivariate time series는 2차원의 벡터 시리즈로 표현됩니다. $$(X_{t1}, X_{t2})’$$ at time t ( t= 1, 2, 3, …) 와 같이 표현하도록 하겠습니다. 두 컴포넌트 $$\{X_{t1}\}$$와 $$\{X_{t2}\}$$를 각 각 독립적인 univariate time series로 분석할 수 있지만, 만약 두 컴포넌트사이에 dependency가 존재할 경우 이를 고려해서 모델링하는 것이 더 바람직합니다. 

랜덤 벡터 $$\mathbf{X_t} =(X_{t1}, X_{t2})$$’에 대해서 평균벡터와 공분산 메트릭스는 $$\mu_t $$와 $$\Gamma(t+h, t)$$와 같이 정의됩니다. 

$$
\mu_t := EX_t = \begin{bmatrix} EX_{t1} \\ EX_{t2} \end{bmatrix}
$$

$$
\Gamma(t+h, t) := Cov(\mathbf{X_{t+h}, X_t}) = \begin{bmatrix} Cov(X_{t+h,1}, X_{t1}) & Cov(X_{t+h,1}, X_{t2}) \\ Cov(X_{t+h,2}, X_{t1}) & Cov(X_{t+h,2}, X_{t2}) \end{bmatrix} 
$$

또한 univariate와 마찬가지로 $$\mu_t $$와 $$\Gamma(t+h, t)$$가 모두 t와 독립적일 때, (weakly) stationary하다고 정의합니다. 이 때의 공분산 메트릭스는 $$\Gamma(h)$$로 표기하고, 메트릭스의 diagonal은 각 각 univariate series인 $$\{X_{t1}\}$$와 $$\{X_{t2}\}$$의 Auto-Covariance Function와 같습니다. 반면 off-diagonal elements는 $$X_{t+h, i}$$와 $$X_{t, i}$$, $$i \ne j$$의 covariances를 나타냅니다. 이 때, $$\gamma_{12}(h) = \gamma_{21}(-h)$$	의 특성이 있습니다. 

$$
\Gamma(h) := Cov(\mathbf{X_{t+h}, X_t}) = \begin{bmatrix} \gamma_{11}(h) & \gamma_{12}(h) \\ \gamma_{21}(h) & \gamma_{22}(h) \end{bmatrix} \\
$$

$$\mu$$ vector와 $$\Gamma(h)$$, $$\rho_{ij}(h)$$의 estimator는 다음과 같습니다. 

$$
\bar{\mathbf{X_n}} = \frac{1}{n} \sum_{t=1}^n\mathbf{X_t}
$$

$$
\begin{align}
\hat\Gamma(h) & = n^{-1} \sum_{t=1}^{n-h} \left( \mathbf{X_{t+h}} - \bar{\mathbf{X_n}} \right) \left( \mathbf{X_{t}} - \bar{\mathbf{X_n}} \right) \ \ \ \ \ \ \ & for \ 0 \le h \le n-1 \\
& = \hat\Gamma(-h)'\ \ \ \ \ \ \  & for \ -n+1 \le h \le 0 \\
\end{align}
$$

correlation between $$X_{t+h,i}$$ and $$X_{t,j}$$

$$
\hat{\rho_{ij}}(h) = \hat{\gamma_{ij}}(h) (\hat{\gamma_{ii}}(0)\hat{\gamma_{jj}}(0))^{-1/2}
$$


이를 m개의 다변량 시계열 데이터로 확대하면 다음과 같습니다.

$$
\mathbf{X_t} :=\begin{bmatrix} X_{t1} \\ \vdots \\ X_{tm} \end{bmatrix}
$$

$$
\mu_t := EX_t = \begin{bmatrix} \mu_{t1} \\ \vdots \\ \mu_{tm} \end{bmatrix}
$$

$$
\Gamma(t+h, t) := \begin{bmatrix} \gamma_{11}(t+h, t) & \cdots & \gamma_{1m}(t+h, t) \\ 
\vdots & \vdots & \vdots \\
\gamma_{m1}(t+h, t) & \cdots & \gamma_{mm}(t+h, t) \end{bmatrix} \\
where \ \gamma_{ij}(t+h, t):=Cov(X_{t+h, i}, X_{t, j})
$$

<b>Definition</b>\\
The m-variate series $$\{\mathbf{X_t}\}$$ is (weakly) stationary if \\
(1) $$\mu_X(t)$$ is independent of t, and \\
(2) $$\Gamma_X(t+h, t)$$ is independent of t for each h. 

이 때 $$\gamma_{ij}(\cdot)$$, $$i \ne j$$ 는 서로 다른 두 시리즈 $$\{X_{ti}\}$$와 $$\{X_{tj}\}$$ 의 cross-covariance라고 부릅니다. 일반적으로 $$\gamma_{ij}(\cdot)$$는 $$\gamma_{ji}(\cdot)$$와 같지 않기 때문에 주의하셔야합니다. 

<b>Basic Properties of $$\Gamma(\cdot)$$:</b>\\
(1) $$\Gamma(h) = \Gamma'(h)$$ \   \\
(2) $$\left\vert\gamma_{ij}(h)\right\vert \le [\gamma_{ii}(0)\gamma_{jj}(0)]^{1/2}$$, i,j=1,...,m \\
(3) $$\gamma_{ii}(\cdot)$$ is an autocovariance function, i = 1,...,m \\
(4) $$\rho_{ii}(0) = 1$$ for all i

Univariate에서 살펴본 white noise 역시 아래와 같이 다변량 정규분포를 따르는 벡터로 정의됩니다. 

<b>Definition</b> \\
The m-variate series $$\{Z_t\}$$ is called white noise with mean 0 and covariance matrix $$\Sigma$$, written
$$\{Z_t\} \sim WN(\mathbf{0}, \Sigma)$$, \\
if $$\{Z_t\}$$ is stationary with mean vector $$\mathbf{0}$$ and covariance matrix function \\
$$\begin{align} \Gamma(h) & = \Sigma, \ \ if \ h = 0 \\ & = 0, \ \ \ \ otherwise. \end{align}$$

White noise의 선형조합으로 표현되는 m-variate series $$\{X_t\}$$를 linear process로 부르며, MA($$\infty$$) 프로세스는 j가 0보다 작은 경우에는 $$C_j=0$$인 linear process에 해당합니다. 

<b>Definition</b>\\
The m-variate series $$\{X_t\}$$ is a linear process if it has the representation
$$\mathbf{X_t} = \sum_{j=-\infty}^\infty C_j \mathbf{Z_{t-j}}, \ \ \ \ \ \{\mathbf{Z_{t-j}}\} \sim WN(\mathbf{0}, \mathbf{\Sigma})$$,\\
where $$\{C_j\}$$ is a sequence of m X m matrics whose components are absolutely summable.

또한 causality를 만족하는 모든 ARMA(p, q) 프로세스는 MA($$\infty$$) 프로세스로 변환할 수 있으며, invertibiliy를 만족하는 모든 ARMA(p, q) 프로세스는 AR($$\infty$$) 프로세스로 변환할 수 있습니다. causality 조건을 만족하면 항상 stationary하고, stationary ARMA(p, q) process는 항상 causality를 만족합니다. 

<b>Causality</b> \\
An ARMA(p, q) process $$\{X_t\}$$ is causal, or a causal function of $$\{Z_t\}$$, if there exist matrices $$\{\Psi_t\}$$ with absolutely summable components such that \\
$$X_t = \sum_{j=0}^{\infty}\Psi_jZ_{t-j}$$ for all t.\\
Causality is equivalent to the condition\\
$$det\Phi(z) \ne 0$$ for all $$z \in \mathbb{C}$$ such that $$\vert z \vert \le 1$$

Causality 조건을 다시 보면 1보다 작은 모든 z에 대해서 $$\Phi(z)$$의 determinant는 0이 아니여야합니다.  이는 <b>$$\Phi(z)$$의 모든 eigenvalue들이 1보다 작아야한다</b> 것과 동일합니다. 

<b> Yule-walker equation</b> \\
Yule-walker equation을 이용해 모델의 파라미터를 추정하는 방법에 대해서 살펴보도록 하겠습니다. 실제로는 소프트웨어를 통해 계산되기 때문에 복잡한 계산을 수행할 필요는 없지만, 원리를 이해하는 것을 목적으로 합니다. 

$$
X_t = \Phi X_{t-1} + Z_t 
$$

양변에 $$X_{t-h}'$$를 곱해줍니다. 

$$
X_t X_{t-h}' = \Phi X_{t-1}X_{t-h}' + Z_tX_{t-h}'
$$

이후 Expectation을 취해줍니다.

$$
E[X_t X_{t-h}'] = \Phi E[ X_{t-1}X_{t-h}' ] + E[Z_tX_{t-h}' ]
$$

$$X_t$$의 평균이 0인 시리즈라고 가정하고, h가 1보다 크거나 같을 때와 h가 0일때로 나눠서 생각해볼수 있습니다.
$$
\begin{align}
\Gamma(h) & = \Phi \Gamma(h-1) + \mathbf{0} \ \ \ \ \ \ for \ h \ge 1 \\
\Gamma(0) & = \Phi \Gamma(-1) + \Sigma_Z \\
& = \Phi \Gamma(1)' + \Sigma_Z \ \ \ \ \ \ for \ h =0
\end{align}
$$

h가 1일때와 h가 0일때의 두 식을 연립하여 풀면 $$\Phi$$를 구할수 있습니다. 
$$
\begin{align}
\Gamma(1) & = \Phi \Gamma(0) + \mathbf{0} \ \ \ \ \ \ for \ h = 1 \\
\Gamma(0) & = \Phi \Gamma(1) + \Sigma_Z \ \ \ \ \ \ for \ h =0 
\end{align}
$$

<b>Cointegration</b>

## 실제 데이터를 이용한 VAR Forecasting

앞에서 계속 사용한 초미세먼지 데이터(PM2.5)에 미세먼지(PM10) 농도를 추가하여 Bivariate 시계열 예측을 수행해보도록 하겠습니다.

* 데이터 : 경남 창원시 의창구 원이대로 450(시설관리공단 실내수영장 앞)에서 측정된 초미세먼지(PM2.5)와 미세먼지(PM10)
* 기간 : 	2018-9-18 19:00 ~ 2018-12-18 17:00 (3개월, 1시간 단위)

<img src = "/assets/img/2019-01-07/output_0_0.png">
