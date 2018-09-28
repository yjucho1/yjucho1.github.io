---
title: "[ML] 클러스터링을 평가하는 척도 - Normalized Mutual Information"
categories: 
  - Machine Learning
  - Unsupervised Learning
  - Evaluation
last_modified_at: 2018-09-28
---

클러스터링은 주어진 데이터에 대한 정보가 많이 많을 때 유용하게 쓸수있는 머신러닝 기법 중 하나입니다. 마케팅에서 유저 정보를 이용해 군집화하여 맞춤 전략을 도출한다던지, 상품(요즘엔 동영상, 음원까지도) 속성을 기반으로 카테고리화하는 것들 등등에서 활용됩니다. 

알고리즘 측면에서는 전통적으로 [Hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering), [K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) 등 다양한 클러스터링 알고리즘이 존재하고, 최근에는 [딥러닝 기반의 클러스터링](https://arxiv.org/abs/1801.07648) 알고리즘도 다양하게 제안되고 있습니다. 

여러가지 논문이나 자료들을 찾아보면 클러스터링 후 결과를 평가하는 방법이 잘 와닿지 않는 경우가 많습니다. 기회가 될 때 한번 정리해보자는 생각으로 시작합니다.

클러스터링 결과를 평가하는 방식은 크게 2가지 형태가 있습니다.
* supervised, which uses a ground truth class values for each sample.
* unsupervised, which does not and measures the ‘quality’ of the model itself.

여기서는 unsupervised 방식의 지표들을 소개하고자 합니다. 

그 첫번째 놈으로 Mutual Information에 대해서 정리합니다.


http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
http://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation

Mutual Information 
두 랜덤 변수간의 상호 의존도를 나타냄
독립변수 A를 통해서 B에 대해서 정보(shannons처럼 단위, 일반적으로는 bits)를 얼마나 얻을수 있는가
결합확률 P(X, Y)와 marginal distribution분포의 곱, P(X)*P(Y)이 얼마나 유사한지를 결정

## 정의

<img src= "/assets/img/2018-09-28/MI_definition.png" width="400">
