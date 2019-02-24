---
title: "RobustSTL : A Robust Seasonal-Trend Decomposition Algorithm for Long Time Series"
categories: 
 - Deep Learning paper
 - Time-series
comments: true
mathjax : true
published: true

---
<b>Qingsong Wen et al (2018, Alibaba Group)</b>

Implementation : [https://github.com/LeeDoYup/RobustSTL](https://github.com/LeeDoYup/RobustSTL)

## Abstract
* 시계열데이터를 trend, seasonality, and remainder components로 분해하는 것은 어노말리 디텍션이나 예측 모델을 만드는데 중요한 역할을 합니다. 
* 기존의 여러가지 성분분해 방식들은 
   * 1) 주기성이 변하거나 이동하는 것, 트렌드나 나머지성분의 갑작스러운 변화를 잘 처리하지 못하며(seasonality fluctuation and shift, and abrupt change in trend and reminder) 
   * 2) 어노말리 데이터에 대해서 로버스트하지 못하거나
   * 3) 주기가 긴 시계열 데이터에 대해서 적용하기 어려운 문제가 있습니다. 
* 본 논문에서는 위와 같은 문제점을 해결할 수 있는 새로운 성분 분해 방식을 제안합니다. 
   * 먼저 sparse regularization와 least absolute deviation loss를 이용해 트렌드를 뽑고
   * Non-local seasonal filter를 사용하여 seasonality 성분을 얻습니다. 
   * 이 과정을 정확한 디컴포지션을 얻을때까지 반복합니다. 
* 실험데이터와 실제 시계열데이터에 대해서 기존 방법들 대비 더 좋은 성능을 보임을 확인하였습니다. 

## Introduction
* 디컴포지션 방법으로 널리 사용되는 방법은 STL(seasonal trend decomposition using Loess), X-13-ARIMA-SEATS, X-11-ARIMA, X-12-ARIMA 등이 있습니다. 하지만 seasonality shift나 fluctuation이 존재할 경우 정확하지 않거나, 빅데이터에 존재하는 long seasonality에는 적합하지 않습니다. 
   * seasonality fluctuation and shift - 하루가 주기인 시계열 데이터가 있다고 했을때, 오늘 1시에서의 seasonality component는 어제의 12시 30분에 대응되고, 그제의 1시 30분에 대응될수 있음
   * Abrupt change of trend and remainder - local anomaly could be a spike during an idle period (busy day의 높은 값보다는 낮아서 정확히 디텍션하기 어려움
   * Long seasonality - 보통은 quarterly or monthly data임. T 주기의 시즈널리티를 찾기 위해서는 T-1개의 데이터가 필요함. 하루 주기에 1분 간격 데이터의 경우 T=1440개고 이와 같은 long seasonality는 기존 방법들로는 풀기어려움
* 이 논문에서 제안한 방법은 Long seasonality period and high noises 더라도 시즈널리티를 비교적 정확하게 디컴포지션할수 있습니다.

## Robust STL Decomposition
###  Model Overview

$$
\begin{align}
y_t & = \tau_t + s_t + r_t, & t = 1, 2, …, N \\
r_t & = a_t + n_t \\
\end{align}
$$ 

where $$a_t$$ denotes spike or dip, and $$n_t$$ denotes the white noise.

* 시계열 모델은 트렌드($$\tau_t$$), 시즈널리티($$s_t$$), 리마인더($$r_t$$)로 구성되어 있고, 리마인더는 스파크 또는 딥과 같은 어노말리($$a_t$$)와 화이트 노이즈($$n_t$$)로 이루어집니다. 
* 제안하는 알고리즘은 크게 4-steps 으로 각 성분을 분해합니다. 
   * Denoise time series by applying bilateral filtering
   * Extract trend robustly by solving a LAD regression with sparse regularization
   * Calculate the seasonality component by applying a non-local seasonal filtering to overcome seasonality fluctuation and shift
   * Adjust extracted components

### Noise Removal

$$
\begin{align}
y^\prime_t & = \sum_{j \in J} w_j^t y_t, & J = t, t \pm 1, …, t \pm H \\
w_j^t & = \frac{1}{z} e^{-\frac{\left\vert j- t \right\vert ^2}{2\delta_d^2}} e^{-\frac{\left\vert y_j - y_t \right\vert ^2}{2\delta_i^2}}
\end{align}
$$

* J는 필터의 윈도우를 의미하며, 윈도우 사이즈는 2H+1 입니다.
* 필터의 가중치는 두개의 가우시안 함수로 구성됩니다. bilateral filter는 [여기](https://en.wikipedia.org/wiki/Bilateral_filter)를 참고하세요.

* After denoising,

$$
\begin{align}
y^\prime_t & = \tau_t + s_t + r^\prime_t, & t = 1, 2, …, N \\
r^\prime_t & = a_t + (n_t - \hat{n}_t \\
\end{align}
$$

Where the $$\hat{n}_t = y_t - y^\prime_t$$ is the filtered noise.

### Trend Extraction
* 시즈널 디퍼런스 오퍼레이터는 같은 주기의 값을 차분하는 것으로 아래와 같이 정의할 수 있습니다. 

$$
\begin{align}
g_t & = \nabla_T y^\prime_t = y^\prime_t - y^\prime_{t-T} \\
& = \nabla_T \tau_t + \nabla_T s_t + \nabla_T r^\prime_t \\
& = \sum_{I=0}^{T-1} \nabla \tau_{t-i} + ( \nabla_T s_t + \nabla_T r^\prime_t )
\end{align}
$$

* 마지막 줄의 수식에서 첫번째 항이 $$g_t$$에 가장 많은 기여를 합니다. $$s_t$$ and $$r^\prime_t$$에 시즈널 디퍼런스 오퍼레이터를 적용하면 값이 매우 작아진다고 가정하기 때문입니다. 
* $$g_t$$에서 트렌드의 first order differece($$\nabla \tau_t$$)를 구하기 위해서 다음과 같은 최적화 식을 사용합니다. 

$$
Minimize \ \sum_{t=T+1}^N \left\vert g_t - \sum_{I=0}^{T-1} \nabla \tau_{t-i} \right\vert + \lambda_1 \sum_{t=2}^N \left\vert \nabla \tau_t \right\vert + \lambda_2 \sum_{t=3}^N \left\vert \nabla^2 \tau_t \right\vert 
$$

* 첫번째 항은 LAD를 사용한 emprical error를 의미합니다. sum-of-squares 보다 아웃라이어에 대해서 더 로버스트하기 때문에 LAD를 사용하였습니다. 
* 두번째와 세번째 항은 각 각 트렌드에 대한 first-order 와 second-order difference operator 입니다.
* 두번째 항은 트렌드 디퍼런스 $$\nabla \tau_t $$ 가 천천히 변화하지만 종종 갑작스러운 레벨 쉬프트(abrupt level shift)가 있다는 것을 의미합니다. 
* 세번째 항은 트렌드가 smooth하고 piecewise linear such that $$\nabla^2 x_t = \nabla(\nabla x_t)) = x_t -2 x_{t-1} + x_{t-2} $$ are sparse

* 이를 매트릭스 형태로 표현하면 다음과 같습니다. 

$$
\Vert P \nabla \tau - q \Vert _1
$$

where the matrix P and vector q are

$$
P = \begin{bmatrix}
M_{(N-T) \times (N-1)} \\
\lambda_1 I_{(N-1) \times (N-1)} \\
\lambda_2 D_{(N-2) \times (N-1)} \\
\end{bmatrix}, 
q = \begin{bmatrix}
g_{(N-T) \times 1} \\
0_{(2N-3) \times 1} \\
\end{bmatrix}
$$

M and D are Toeplitz matrix (refer to the paper for details)

* 위의 최적화식을 통해서 $$\tau_1$$에 대한 상대적인 트렌드(relative trend, $$\tilde{\tau}_t^r$$)를 구할수 있습니다.

$$
\tilde{\tau}_t^r = \tilde{\tau}_t - \tau_1 = 
\begin{cases}
0, & t=1 \\
\sum_{I=2}^t \nabla \tilde{\tau}_i, & t \ge 2
\end{cases}
$$

* 그리고 나서, 디컴포지션 모델은 아래와 같이 업데이트 됩니다. 
$$
y_t'' = y_t' - \tilde{\tau}_t^r = s_t + \tau_1 + r_t'' \\
r_t’’ = a_t + (n_t - \hat{n}_t) +  (\tau_t - \tilde{\tau}_t) 
$$

### Seasonality Extraction
* relative trend component를 분리한 후에는, $$y’’_t$$는 시즈널리티로 오염되어 있다고 생각할 수 있습니다. 
* 기존의 시즈널리티 분해 방법들은 주기가 $$T$$인 $$s_t$$’를 구하기 위해서는 K개의 연속적인 값 $$y_{t-KT}, y_{t-(K-1)T}, …, y_{t-T}$$ 만 고려하였습니다. 하지만, 이 방식은 시즈널리 쉬프트 현상을 설명할수 없다는 단점이 있습니다. 

* 여기서는  $$y’’_{t-KT}$$를 중심으로 인접한 값들을 고려합니다.  $$y’’_{t-KT}$$를 계산할때는 그 값을 중심으로 2H+1개의 인접값들 $$y’’_{t-KT-H}, y’’_{t-KT-H+1}, …, y’’_{t-KT}, y’’_{t-KT+1}…, y’’_{t-KT+H}$$를 사용합니다. 
* 시즈널 컴포넌트 $$s_t$$ 는 아래와 같이 of $$y’’_t$$의 가중합으로 표현됩니다. 

$$
\tilde{s}_t = \sum_{(t’, j) \in \Omega} w^t_{(t’,j’)}y’’_j
$$

Where the $$w^t_{(t’,j’)}$$ and $$\Omega$$ are defined as 

$$
w^t_{(t’,j’)} = \frac{1}{z}e^{-\frac{\left\vert j- t \right\vert ^2}{2\delta_d^2}} e^{-\frac{\left\vert y’’_j - y’’_{t’} \right\vert ^2}{2\delta_i^2}} \\
\Omega = \{(t’,j) \vert (t’=t-k \times T, j= t’ \pm h )\} \\
k=1, 2, …, K; \ h=0, 1, …, H
$$

* 시즈널티리를 분리한 후에는, 리마이더 시그널은 아래와 같이 표현됩니다. 
$$
r’’’_t = y’’_t - \tilde{s}_t = a_t + (n_t - \hat{n}_t) + (\tau_t - \tilde{\tau}_t) + (s_t - \tilde{s}_t)
$$

### Final Adjustment
* 시즈널리티 컴포넌트의 합계는 0으로 조정되어야합니다. 

$$\sum_{I=j}^{I=j+T-1}s_i = 0 $$

* 따라서 평균값(트렌드 $$\tau_1$$에 대응되는 값)을 빼줌으로서 시즈널리트를 조정합니다. 

$$
\hat{\tau}_1 = \frac{1}{T\lfloor N/T \rfloor} \sum_{t=1}^{T\lfloor N/T \rfloor} \tilde{s}_t \\
\hat{s}_t = \tilde{s}_t - \hat{\tau}_1 \\
\hat{\tau}_t = \tilde{\tau}^r_t + \hat{\tau}_1 \\
\hat{r}_t = y_t - \hat{s}_t + \hat{\tau}_t 
$$

* 리마인더 시그널 $$\hat{r}_t $$ 가 수렴할 때까지 위 과정을 반복합니다. 

<img src = "/assets/img/2019-02-24/algorithm1.png" width='550'><br>

## Experiments

<img src = "/assets/img/2019-02-24/fig3.png" width='550'><br>

<img src = "/assets/img/2019-02-24/fig4.png" width='550'><br>

<img src = "/assets/img/2019-02-24/table2.png" width='450'><br>
