---
title: "[ML] 클러스터링을 평가하는 척도 - Normalized Mutual Information"
categories: 
  - Machine Learning
  - Unsupervised Learning
  - Evaluation
last_modified_at: 2018-09-28
---

클러스터링은 주어진 데이터에 대한 정보가 많이 많을 때 유용하게 쓸수있는 머신러닝 기법 중 하나이다. 
[Hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering), [K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) 등 다양한 클러스터링 알고리즘이 존재하고, 최근에는 [딥러닝 기반의 클러스터링](https://arxiv.org/abs/1801.07648) 알고리즘도 다양하게 제안되고 있습니다. 

어쩌고 저쩌고..
The sklearn.metrics.cluster submodule contains evaluation metrics for cluster analysis results. There are two forms of evaluation:

supervised, which uses a ground truth class values for each sample.
unsupervised, which does not and measures the ‘quality’ of the model itself.

http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
http://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation

Mutual Information 
두 랜덤 변수간의 상호 의존도를 나타냄
독립변수 A를 통해서 B에 대해서 정보(shannons처럼 단위, 일반적으로는 bits)를 얼마나 얻을수 있는가
결합확률 P(X, Y)와 marginal distribution분포의 곱, P(X)*P(Y)이 얼마나 유사한지를 결정

## 정의

![Mutual Information Definition]({{ "/assets/img/2018-09-28/MI_definition.png"}})
