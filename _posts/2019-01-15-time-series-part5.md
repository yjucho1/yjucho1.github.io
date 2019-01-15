---
title: "시계열 분석 - part5"
categories: 
  - Spatio-Temporal Data
comments: true
mathjax : true
published: true

---

지금까지는 linear 형태의 시계열 모형에 대해서 살펴보았습니다. 이번 포스팅에서는 non-linear 모형의 대표적인 예인 ARCH, GARCH에 대해서 공부해보도록 하겠습니다. 

## ARCH

ARCH(autoregressive conditional heteroskedasticity) 모델은 다음과 같이 정의됩니다. 

<b>Definition</b>

$$
\begin{align}
X_t & = \sigma_t * Z_t \\
\sigma_t^2 & = \alpha_0 + \sum_{I=1}^p \alpha_i x_{t-i}^2
\end{align}
$$

ARCH(p)를 이해하기 위해서 평균과 분산을 살펴보도록 하죠.

$$
\begin{align}
E[X_t \vert  X_{t-1}, X_{t-2}, …] & = E[\sigma_t * Z_t \vert  X_{t-1}, X_{t-2}, …] \\
& = \sigma_t E[ Z_t \vert  X_{t-1}, X_{t-2}, …]  \\
& = 0 \\ 
\\
E[X_t] & = E_{X_{t-1}, X_{t-2}, …}[E_{X_t \vert  X_{t-1}, X_{t-2}, …} [X_t] ] \\
&= 0\\
\\
Var[X_t \vert  X_{t-1}, X_{t-2}, …] & = E[X_t^2 \vert  X_{t-1}, X_{t-2}, …] \\
& = E[ \sigma_t^2Z_t^2 \vert  X_{t-1}, X_{t-2}, …] \\ 
& = \sigma_t^2 E[ Z_t^2 \vert  X_{t-1}, X_{t-2}, …] \\
& = \sigma_t^2 \\ 
\\
Cov[X_{t+h}, X_t] & = E[X_{t+h} X_t] \\
& = E_{X_{t+h-1}, X_{t+h-2}, …}[X_t E_{X_{t+h} \vert  X_{t+h-1}, X_{t+h-2}, …} [X_{t+h}] ]\\
& =0
\end{align}
$$

$$\{X_t\}$$의 평균은 0이고, lag=h인 관측값간의 공분산이 0이라는 것은 $$\{X_t\}$$가 white noise라는 것을 의미합니다. 또한 $$Var[X_t]=\sigma_t^2$$이기때문에, $$\sigma_t^2$$를 volatility 라고 합니다. 

<b>example</b>

ARCH(1) : 
$$
\left\{
\begin{align}
X_t & = \sigma_t * Z_t \\
\sigma_t^2 & = \alpha_0 + \alpha_1 X_{t-1}^2 \\
\end{align}
\right.
$$


첫번째 식을 제곱한 후, 두 식을 빼면 다음과 같습니다. 
$$
\begin{align}
X_t^2 & = \sigma_t^2 * Z_t^2 \\
\alpha_0 + \alpha_1 X_{t-1}^2  & = \sigma_t^2 \\
X_t^2 - (\alpha_0 + \alpha_1 X_{t-1}^2) & = \sigma_t^2(Z_t^2 - 1) \\
X_t^2 & = \alpha_0 + \alpha_1 X_{t-1}^2 + \sigma_t^2(Z_t^2 - 1)
\end{align}
$$

마지막 수식을 살펴보면 $$\{X_t^2\}$$ 가 auto-regressive 형태로 설명됩니다. 즉, ARCH(1) 모델은 noise가 non-Gaussian 인 AR(1) 프로세스와 동일한 것을 알 수 있습니다. 

ARCH(p) 모델에서 추정해야하는 모델 파라미터는 $$\alpha_0, \alpha_1$$으로 Maximum Likelihood Estimation(MLE)를 이용해 추정합니다. 자세한 증명은 여기서 생략하도록 하겠습니다. 

ARCH(p) 모델을 앞서 살펴본 linear 모델들과 합친 joint ARCH model도 생각해볼수 있습니다. 예를 들어, AR(1)-ARCH(1) 모델은 다음과 같습니다. 

<b>example</b>

AR(1)-ARCH(1) : $$\{X_t\}$$는 AR(1) process이고, $$\{Z_t\}$$가 ARCH인 모델

$$
\left\{
\begin{align}
X_t & = \phi X_{t-1} + Z_t \\
\sigma_t^2 & = \alpha_0 + \alpha_1 Z_{t-1}^2 \\
\end{align}
\right.
$$
