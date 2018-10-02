---
title: "클러스터링을 평가하는 척도 - Rand Index"
categories: 
  - Clustering Evaluation
comments: true
last_modified_at: 2018-10-01
---

[클러스터링을 평가하는 척도 - Mutual Information](/clustering%20evaluation/clustering-metrics/)와 이어집니다. 

## Rand Index

클러스터링 결과를 평가하기 위해 [Rand Index](https://en.wikipedia.org/wiki/Rand_index) 도 자주 쓰입니다. Rand Index는 주어진 N개의 데이터 중에서 2개을 선택해 이 쌍(pair)이 클러스터링 결과 U와 V에서 모두 같은 클러스터에 속하는지, 서로 다른 클러스터에 속하는지를 확인합니다. 

## 정의

n개의 원소로 이루어진 집합 S={o<sub>1</sub>, ... o<sub>n</sub>}와 S를 r개의 부분집합으로 할당한 partition U={U<sub>1</sub>, ..., U<sub>r</sub>}와 S를 s개의 부분집합으로 할당한 partition V={V<sub>1</sub>, ..., V<sub>r</sub>}에 대해서 아래와 같을 때, 
* a = S의 원소로 이루어진 쌍(pair) 중에서 U와 V에서 모두 동일한 부분집합으로 할당된 쌍의 갯수
* b = S의 원소로 이루어진 쌍(pair) 중에서 U와 V에서 모두 다른 부분집합으로 할당된 쌍의 갯수
* c = S의 원소로 이루어진 쌍(pair) 중에서 U에서는 동일한 부분집합으로 할당되었지만, V에서는 다른 부분집합으로 할당된 쌍의 갯수
* d = S의 원소로 이루어진 쌍(pair) 중에서 U에서는 다른 부분집합으로 할당되었지만, V에서는 동일한 부분집합으로 할당된 쌍의 갯수

Rand Index, RI는 다음과 같이 정의됩니다.

<img src= "/assets/img/2018-09-28/RI.gif" width="300">

직관적으로 분모는 S에 속하는 n개의 원소 중에서 2개를 뽑는 경우의 수, S의 원소로 가능한 모든 쌍(pair)의 갯수를 의미하고, 분자인 a와 b는 U와 V의 결과가 서로 일치하는 쌍의 갯수를 의미합니다. 

## 클러스터링 평가 지표로서 Rand Index

클러스터링 결과는 n개의 주어진 데이터를 r개 혹은 s개의 부분집합으로 할당하는 것과 같습니다. 즉 위의 정의에서 partition U와 partition V는 두개의 서로 다른 클러스터링 결과에 해당됩니다. 

예를 들어, {a, b, c, d, e, f} 총 6개의 데이터가 존재하고 첫번째 클러스터링 알고리즘을 적용한 결과가 U = [1, 1, 2, 2, 3, 3]와 같고, 두번째 클러스터링 알고리즘을 적용한 결과 V = [1, 1, 1, 2, 2, 2]라고 합시다. <br><br> 6개의 데이터로 가능한 pair는  {a, b}, {a, c}, {a, d}, {a, e}, {a, f}, {b, c}, {b, d}, {b, e}, {b, f}, {c, d}, {c, e}, {c, f}, {d, e}, {d, f}, {e, f}로 총 15개입니다. <br><br> 그중에서 {a, b}는 U와 v에서 모두 동일한 클러스터에 할당됩니다. (a와 b가 U에서 클러스터1, V에서도 클러스터1에 할당) 마찬가지로 {e, f}도 동일한 클러스터에 할당됩니다. (U에서는 클러스터3에 할당, V에서는 클러스터2에 할당) <br><br>  반면에 {a, d}는 U와 V에서 모두 다른 클러스터에 할당되는 쌍입니다. (U에서는 a는 클러스터1이고 d는 클러스터2, V에서는 a는 클러스터1이고 d는 클러스터2) {a, e}, {a, f}, {b, d}, {b, e}, {b, f}, {c, e}, {c, f} 도 마찬가지로 서로 다른 클러스터에 할당됩니다. <br><br> 나머지 쌍 {c, d}, {d, e} 은 U에서는 동일한 클러스터에 할당되었지만, V에서는 다른 클러스터에 할당되거나 그 반대에 해당하는 경우들입니다.  <br><br> 위의 내용을 토대로 최종 Rand Index는 (2+8) / 15 = 0.667이 됩니다.

> Rand Index는 항상 0과 1사이의 값을 갖습니다. 

> 두가지 클러스터링 결과인 U와 V가 서로 완전히 일치할 때, Rand Index는 1의 값을 갖습니다. 반대로 어떤 데이터 쌍에 대해서도 일치된 결과를 보이지 않을 경우 0의 값을 갖게 됩니다.

하지만 Rand Index를 그대로 사용하는 경우는 없습니다. 사례를 통해 살펴보겠습니다. 지난번에 살펴보았던 Fashin MNIST를 이용해 데이터를 랜덤으로 할당한 결과(y_random)와 실제 클래스(y_true)를 이용해 rand index를 구해보겠습니다.

> scikit-learn에는 rand index를 바로 계산하는 api가 없네요.


```python
from keras.datasets import fashion_mnist
import numpy as np

def rand_index(y_true, y_pred):
    n = len(y_true)
    a, b = 0, 0
    for i in range(n):
        for j in range(i+1, n):
            if (y_true[i] == y_true[j]) & (y_pred[i] == y_pred[j]):
                a +=1
            elif (y_true[i] != y_true[j]) & (y_pred[i] != y_pred[j]):
                b +=1
            else:
                pass
    RI = (a + b) / (n*(n-1)/2)
    return RI

_, (X, y_true) = fashion_mnist.load_data()
y_random = np.random.randint(0, 10, 10000)


print(rand_index(y_true, y_random))
## output : 0.8200785478547855
```

<b>랜덤하게 클래스를 할당했음에도 불구하고 RI가 0.82로 높게 나타납니다.</b> 클러스터의 수가 증가할수록 pair를 이루는 두 데이터가 서로 다른 클러스터에 속할 확률이 높아지기 때문입니다. 이로 인해 클러스터 수가 많아지면 b값이 커질 확률이 크고, rand index도 높은 값을 갖습니다. 

<img src= "/assets/img/2018-09-28/RI_num_cluster.png" width="700">
<i>Fig. 클러스터 수에 따른 Rand Index 변화</i>

> 이런건 우리가 원하는게 아니잖아요?


### Adjusted Rand Index

따라서 일반적으로 Rand Index를 확률적으로 조정한 Adjusted Rand Index를 사용합니다. 

### contingency table
contingency table은 U와 V partition을 |U<sub>i</sub> ∩ V<sub>i</sub>| = n<sub>ij</sub> 로 표기하여 요약한 테이블 입니다.

<img src= "/assets/img/2018-09-28/contingency.svg" width="300">

<img src= "/assets/img/2018-09-28/ARI.gif" width="400">










