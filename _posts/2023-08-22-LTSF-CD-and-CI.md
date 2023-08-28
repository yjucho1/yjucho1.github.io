---
title: "The Capacity and Robustness Trade-off: Revisiting the Channel Independent Strategy for Multivariate Time Series Forecasting"
categories: 
 - Time-series
comments: true
mathjax : true
published: true

---
<b>Han, Lu, Han-Jia Ye, and De-Chuan Zhan (2023)</b>

우왕 2023년ㅎㅎㅎ 

## Abstract

## Analysis 
* CD전략과 CI전략에서의 yule-walker equation 분석과 train과 test데이터셋의 각 채널별 ACF와 전체 채널의 ACF합 비교해 봄
* 학습데이터로부터 파라미터 $$W$$를 추정한 모델 ($$R(\hat{W})$$)의 리스크분석과 CD 및 CI 전락으로 학습된 모델에 대해 각 데이터셋뱔 통계값(train error, test error, diff W, generalization error)를 확인함 

## Practical Guide
* 이전 섹션을 통해서 CD 학습전략이 capacity는 높지만, robustness 측면에서는 떨어지는 것을 확인하였다. 반대로 CI 학습전략은 robustness 측면에서는 강건함을 보였다. 
* 이번 섹션은 위와 같은 특성을 고려하여 CD 전략을 개선함으로써 CI 전략의 성능(capacity)보다 더 좋은 성능을 낼수있음을 실험적으로 확인하고자 한다. 

### Predict Residuals with Regularization 
<img src = "/assets/img/2023-08-22/fig7.png"><br>
위에 Fig 7은 실제 값과, CD와 CI 전략간의 예측값을 비교하여 살펴본 결과, CD 학습에 의한 모델이 sharp하며, 강견하지 않은(non-robust) 예측값을 생성하는 것을 예시로 보여주고 있다. 이러한 현상을 개선하기 위해 이 연구에서는 regularization를 이용하며 Residual을 예측하는 objective를 제안하였다. 

$$
min_{f} \frac{1}{N} \sum_{i=1}^{N} \ell (f(X^{(i)}-N^{(i)})+N^{(i)}, Y^{(i)}) + \lambda\Omega(f)
\\ where \ N^{(i)}=X_{:,L}^{(i)}$$

$$N^{(i)}$$는 입력값 X의 마지막 값입니다. reqularization term $$\Omega$$는 pytorch의 L2를 사용하였다고 한다. 

<img src = "/assets/img/2023-08-22/table4.png"><br>
Table4는 실험 결과를 보여준다. Linear 모델과 Transformer 모델에 제안한 방법(PRReg)를 목적식으로 사용하였을때 결과로, 적절한 $$\lambda$$를 사용할경우 CI전략보다 더 우수한 성능을 보인다. PRReg는 기본적으로 CD전략이고 regularization term의 $$\lambda$$가 너무 크면 언더피팅되고, 너무 작으면 CD와 마찬가지로 오버피팅되어 robustness가 떨어지는 것을 확인할수 있었다.  

<img src = "/assets/img/2023-08-22/table5.png"><br>
PatchTST와 비교 (patchtst가 가장 우수하거나, transfomer+PRReg모델모다 더 나은 성능)
