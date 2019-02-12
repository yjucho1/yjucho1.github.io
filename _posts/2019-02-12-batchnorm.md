---
title: "How does batch normalization help optimization?"
categories: 
 - Deep Learning paper
comments: true
mathjax : true
published: false

---

<b>Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas et al. (2018, MIT) </b>

## Abstract
* BatchNorm은 딥러닝의 안정적이고 학습속도를 빠르게 하는데 도움을 주는 기법으로 널리 활용되고 있습니다 .
* 하지만 그 활용성에 비해서 왜 BatchNorm이 효과적인지에 대한 실질적인 고찰은 거의 없었으며, 대부분은 internal covariance shift를 줄이는 효과를 줄이기 때문이라고 믿고 있습니다.
* 이 논문에서는 internal covariance shift라는 것이 실제로는 BatchNorm과 거의 상관없이 없다는 것을 실험적으로 밝혔습니다. 또한 BatchNorm기법이 최적화 함수를 훨씬 smoother하게 만들어주기 때문이라는 것을 이론적, 실험적으로 확인하였으며 이 영향으로 인해 그래디언트가 더 안정적이게 하여 빠른 학습이 가능하다는 것을 주장합니다. 

## Introduction
* 지난 몇년간 딥러닝이 컴퓨터비전, 스피치 인식 등과 같은 풀기어려운 문제를 해결하는데 성공하였으며, 이러한 성공에는 BatchNorm이 큰 기여를 하고 있습니다. 
* BatchNorm의 실용성은 논란의 여기가 없지만, 그러한 효과가 왜 발생하느지에 대한 명확한 이유는 아직 밝혀지지 않았습니다. 
* BatchNorm이 처음 제안되었을 때는 internal covariance shift(ICS)를 최소화하기 위한 방안으로 설명되었지만, 이 연구에서는 ICS와의 연관성을 말해주는 상세한 증거를 찾지 못하였습니다,
* Constribution
   * BatchNorm과 ICS는 아무런 관련이 없다는 것을 설명하며
   * BatchNorm이 효과적인 이유는 최적화 함수를 더 smooth하게 만들어 learning rate가 더 크더라도 안정적인 그래디언트를 보장하여 더 빠른 학습을 가능하게 하기 때문임을 확인하였습니다. 
   * 이러한 내용을 실험적인 확인 이외에 loss함수와 그 gradient의 Lipschitzness 를 이용해 이론적으로 설명하였습니다. 
   * 이러한 근본적인 고찰을 통해서 BatchNorm과 동일한 효과를 가져오는 다른 기법들이 존재할수 있는 가능성을 제시하였습니다.

## Batch normalization and internal covariance shift
* 처음에 Ioffe and Szegedy의 BatchNorm은 모델이 학습될때는 파라미터가 바뀌기때문에 각 레이엉의 입력값들의 분포가 달라지는 현상(internal covariate shift)를 줄이기 위해 제안되었습니다. 
* BatchNorm이란 각 레이어의 액티베이션값을 평균과 분산을 각 각 0과 1로 정규화시킨 후, 모델의 설명력을 유지하기 위해 다시 scaled and shifted를 해주는 과정으로 이루집니다. 이 과정은 이전 레이어의 non-linearity전에 이루어집니다. 

### Does BatchNorm’s performance stem form controlling internal covariate shift?
* BatchNorm 논문에서는 각 레이어의 인풋 분포의 평균과 분산을 제한하는 것이 학습성능에 직접적인 영향을 준다고 주장하였습니다. 이 것을 입증하기 위해 한가지 실험을 수행하였습니다. 
* 배치놈 이후에 랜덤한 노이즈를 일부러 주입하여 네트워크를 학습시켜 보았습니다. 노이즈는 평균이 0이 아니고, 분산도 1이 아닌 분포에서 추출하여 각 레이어의 액티베이션에 더해주었습니다. 이 때, 각 스텝마다 노이즈의 분포가 달라지도록 하여 꽤 심한 covariate shift를 만들도록 하였습니다. 
* Fig2는 standard, standard + batchnorm, standard + noisy batchnorm 세가지 네트워크의 학습 성능(좌)을 나타냅니다. 또한 시간에 따른 레이어의 액티베이션의 분포들(우)을 함께 표시하였습니다. 그림에서 볼수 있듯이, 학습데이터셋을 기준으로 batchnorm과 noisy batchnorm의 성능차이는 거의 없는 것을 확인하였습니다. 두가지 모두 standard 보다 높은 성능을 보였습니다. 
* Noisy batchnorm의 액티베이션 분포들은 불안정하지만, 학습 성능은 좋다는 실험 결과는 batchnorm의 효과가 레이어의 인풋 분포를 안정시키기 때문이다는 주장을 반박하는 결과입니다.

### Is BatchNorm reducing internal covariate shift?
* 그렇다면 더 광의적인 측면의 internal covariate shift가 batchnorm의 높은 학습성능과 직접적으로 연관된 것은 아닐까? 
* 네트워크의 각각 레이어는 주어진 인풋에 대해서 리스크 최적화 문제를 푸는 것으로 생각할수 있고, 파라미터가 업데이트될 때마다 인풋을 바꾸고, 결과적으로 최적화 문제자체를 바꾸게 됩니다. Ioffe and Szegedy는 이 현상을 internal covariate shift라고 불렀고, 각 레이어의 인풋 분포 관점에서 설명하려고 했습니다. 하지만 이 관점은 batchnorm의 성공적인 성능을 설명해주지 못한다는 것을 앞서 실험을 통해 살펴보았습니다.
* 인풋 분포가 아니라 전체 최적화 관점에서 각 레이어의 그래디어트 변화를 살펴보도록 하겠습니다. 
