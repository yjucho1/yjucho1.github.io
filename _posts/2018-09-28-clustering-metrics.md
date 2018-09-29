---
title: "[ML] 클러스터링을 평가하는 척도"
categories: 
  - Machine Learning
  - Unsupervised Learning
  - Clustering Evaluation
last_modified_at: 2018-09-28
---

클러스터링은 주어진 데이터에 대한 정보가 많지 않을 때 유용하게 쓸수있는 머신러닝 기법 중 하나입니다. 마케팅에서 유저 정보를 이용해 세그먼트를 나눠 맞춤 전략을 도출한다던지, 유사한 상품(동영상, 음원까지도) 속성을 분석하여 인사이트를 도출하는 분석 등등에서 활용됩니다. 

알고리즘 측면에서는 전통적으로 [Hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering), [K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) 등 다양한 클러스터링 알고리즘이 존재하고, 최근에는 [딥러닝 기반의 클러스터링](https://arxiv.org/abs/1801.07648) 알고리즘도 다양하게 제안되고 있습니다. 

여러가지 논문이나 자료들을 찾아보면 클러스터링 후 결과를 평가하는 방법이 잘 와닿지 않는 경우가 많습니다. 기회가 될 때 한번 정리해보자는 생각이 들어 포스팅을 시작하게 되었습니다.

클러스터링 결과를 평가하는 방식은 크게 2가지 형태가 있습니다.
* supervised, which uses a ground truth class values for each sample.
  * 지도 방식으로 실제 데이터의 클래스가 존재할때입니다. 
  * 새로운 클러스터링 알고리즘의 성능을 평가하기위해 이미 알려진 벤치마크 데이터셋을 이용해 다른 알고리즘들과 Accuracy 기준으로 비교하는 방식입니다. 
* unsupervised, which does not and measures the ‘quality’ of the model itself.
  * 비지도 방식으로 모델의 좋고 나쁨을 평가하지 않는 방식입니다.

이 포스팅에서는 비지도 방식으로 클러스터링 결과를 펴가하는 지표들을 소개하고자 합니다. 



http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
http://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation

## Mutual Information 
[Mutual Information](https://en.wikipedia.org/wiki/Mutual_information)은 정보학이나 확률론에서 두 확률 변수간의 상호 의존도를 나타내는 지표입니다. 확률변수 X와 Y가 존재할때, X를 통해서 Y에 대해서 정보(shannons처럼 단위, 일반적으로는 bits)를 얼마나 얻을수 있는가를 측정하는 것으로 결합확률분포 P(X, Y)와 각 변수의 marginal distribution의 곱, P(X)*P(Y)이 얼마나 유사한가로도 해석할수 있습니다.

## 정의

<img src= "/assets/img/2018-09-28/MI_definition.png" width="400">
