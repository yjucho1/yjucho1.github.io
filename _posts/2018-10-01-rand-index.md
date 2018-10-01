---
title: "클러스터링을 평가하는 척도 - Rand Index"
categories: 
  - Clustering Evaluation
comments: true
last_modified_at: 2018-10-01
---

## Rand Index

[Rand Index](https://en.wikipedia.org/wiki/Rand_index) 도 자주 쓰입니다. Rand Index는 classification 문제에서 accuracy와 유사하지만, 클래스 라벨을 이용하는 것은 아니다. N개의 데이터 중에서 2개의 샘플을 선택해 이 쌍(pair)이 클러스터링 결과 U와 V에서 모두 같은 클러스터에 속하는지, 서로 다른 클러스터에 속하는지를 확인합니다. 

## 정의

n개의 원소로 이루어진 집합 S={o<sub>1</sub>, ... o<sub>n</sub>}와 S를 r개의 부분집합으로 할당한 partition X={X<sub>1</sub>, ..., X<sub>r</sub>}와 S를 s개의 부분집합으로 할당한 partition Y={Y<sub>1</sub>, ..., Y<sub>r</sub>}가 있을때, 
* a는 X와 Y에서 모두 동일한 클러스터로 할당된 


예를 들어, {a, b, c, d, e, f} 총 6개의 데이터가 존재하고 첫번째 클러스터링 알고리즘을 적용한 결과가 U = [1, 1, 2, 2, 3, 3]와 같고, 두번째 클러스터링 알고리즘을 적용한 결과 V = [1, 1, 1, 2, 2, 2]라고 합시다. 
* 6개의 데이터 중 가능한 pair는  {a, b}, {a, c}, {a, d}, {a, e}, {a, f}, {b, c}, {b, d}, {b, e}, {b, f}, {c, d}, {c, e}, {c, f}, {d, e}, {d, f}, {e, f}로 총 15개입니다. 
* 그중에서 {a, b}는 U와 v에서 모두 동일한 클러스터에 할당됩니다. (a와 b가 U에서 클러스터1, V에서도 클러스터1에 할당됨) 마찬가지로 {e, f}도 동일한 클러스터에 할당됩니다. (U에서는 클러스터3에 할당, V에서는 클러스터2에 할당)
* 반면에 {a, d}는 U와 V에서 모두 다른 클러스터에 할당되는 쌍입니다. (U에서는 a-클러스터1 & d-클러스터2, V에서는 a-클러스터1 & d-클러스터2) {a, e}, {a, f}, {b, d}, {b, e}, {b, f}, {c, e}, {c, f} 도 마찬가지로 서로 다른 클러스터에 할당됩니다.
* 나머지 쌍들은 U에서는 동일한 클러스터에 할당되었지만, V에서는 다른 클러스터에 할당되거나 그 반대에 해당하는 경우들입니다. {c, d}, {d, e}
위의 내용을 토대로 최종 Rand Index는 (2+8) / 15 = 0.667이 됩니다.



