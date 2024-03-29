---
title: "Attention is All You Need"
categories: 
  - Attention
  - Deep Learning paper
comments: true
mathjax : true
published: true

---

<b> Ashish Vaswani et al. (Google Brain), 2017 </b>

Tensorflow implementtation : 
* [https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)
* [https://github.com/Kyubyong/transformer](https://github.com/Kyubyong/transformer)

PyTorch implementation : 
[https://github.com/jadore801120/attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
[guide annotating the paper with PyTorch implementation]
(http://nlp.seas.harvard.edu/2018/04/03/attention.html)

## Introduction

언어 모델링과 기계 번역과 같은 시퀀스 모델링에서 RNN, LSTM, GRU는 최신 기법으로 확고하게 자리잡고 있습니다. 인코더-디코더 구조를 활용하는 등 향상된 성능을 얻기 위해 많은 시도들이 있어왔습니다. 

recurrent models에서 hidden state $$h_t$$는 previus hidden state $$h_{t-1}$$과 $$t$$번째 입력값의 함수로 계선됩니다. 이러한 순차적 계산방식은 병렬처리가 어렵습니다. 입력 시퀀스가 길어질수록 이러한 제약 사항은 중요한 이슈가 됩니다. 

어텐션 메커니즘은 시퀀스 모델링에서 인풋과 아웃풋 시퀀스간 거리에 상관없이 의존성을 모델링할수 있는 방법으로 사용되었습니다. 하지만 기존 연구들은 recurrent netowork의 보완역할로만 어텐션 메커니즘을 사용하여 여전히 순차적 계산방식에 대한 제약이 남아있습니다. 

이 연구에서는 `"Transformer"`라는 새로운 구조를 제안합니다. recurrence 구조를 탈피하고, 인풋과 아웃풋간의 글로벌 의존성을 모델링하는 어텐션 메커니즘만을 사용합니다. 이로 인해 병렬처리가 가능하고, P100 GPU 8장으로 12시간동안 학습시키는 것만으로도 state of art 수준의 번역 품질을 달성할수 있었습니다. 

## Backgroud

[Extended Neural GPU](https://arxiv.org/abs/1610.08613), [ByteNet](https://arxiv.org/abs/1610.10099), [ConvS2S](https://arxiv.org/abs/1705.03122) 등은 컨볼루션 뉴럴 네트워크를 사용하여 병렬처리가 가능하도록 시퀀스 모델을 제안하였습니다. 하지만 인풋과 아웃풋 포지션의 거리가 멀어질수록 계산량도 이에 따라 증가하고, 멀리 떨어진 포지션 사이의 의존성을 학습하기가 더 어려워집니다. 

`self-attention`, 혹은 `intra-attention`은 포지션을 고려하여 시퀀스의 representation을 계산하며, 이는 독해나 요약 등에서 다양하게 사용되고 있습니다. 

[End-to-end memory network](https://arxiv.org/abs/1503.08895)는 시퀀스를 순차적으로 다루는 것이 아니라, 어텐션을 순차적으로 다루는 메커니즘으로 간단한 문답과 같은 언어 모델링에서 잘 작동하는 것으로 알려져 있습니다. 

(우리가 아는 한도에서는) Transformer는 RNN이나 CNN을 사용하지 않으면서도 인풋과 아웃풋의 표현력을 셀프-어텐션만으로 계산하는 최초의 접근법입니다. 

## Model Architecture

대부분의 시퀀스 변환 모델은 인코더-디코더 구조를 사용합니다. 인코더는 인풋 시퀀스 $$(x_1, \cdots, x_n)$$를 $$z=(z_1, \cdots, z_n)$$로 맵핑합니다. z가 주어지면, 디코더는 아웃풋 시퀀스 $$(y_1 \cdots, y_m)$$를 한번에 하나씩 생성합니다. 매스텝마다 모델은 auto-regressive하게 다음단어를 생성할때마다 입력값과 이전에 생성된 심볼을 사용합니다. 

Transformer는 여러개의 셀프-어텐션을 쌓고, point-wise하게 계산하며, fully connected layer를 사용합니다. 

### Encoder and Decoder Stacks

<img src = "/assets/img/2018-10-30/fig1.png" width="500">

<small>*그림1. The Transformer - model architecture*</small>


<b>Encoder</b> : 인코더는 6개의 동일한 레이어로 구성됩니다. 각 레이어는 2개의 서브레이어를 갖습니다. 첫번째는 `multi-head self-attention mechanism`, 두번째는 간단한 `position-wise fully connected feed-forward network`입니다. 두 서브레이어 사이에는 `residual connection`과 `layer normalization`을 사용하였습니다. 즉 서브레이어의 아웃풋은 $$LayerNorm(x+Sublayer(x))$$입니다. residual connection이 가능하도록 모델의 모든 레이어의 아웃풋 디멘전은 $$d_{model}=512$$로 설정하였습니다. 

<b>Decoder</b> : 디코더도 6개의 동일한 레이어로 구성됩니다. 마찬가지로 각 레이어는 위에서 언급한 2개의 서브레이어를 갖지만, 추가로 한가지가 더 있습니다. multi-head attention을 수정하여 생성하고자 하는 $$i$$번째 시퀀스가 후속 시퀀스에는 영향을 받지 않고, $$i$$보다 작은 위치의 아웃풋에만 의존할수 있도록 하였습니다(`Masked Multi-Head Attention`). 

### Attention

어텐션 함수는 Query, Key-Value 쌍을 아웃풋에 맵핑시키는 것입니다. query, key, valut, output은 모두 벡터입니다. 아웃풋은 value의 가중합이고, 이 때 가중치는 query와 값에 대응되는 key로 계산됩니다. 

#### Scaled Dot-Product Attention


<img src = "/assets/img/2018-10-30/fig2.png" width="500">

<small>*그림2. (왼쪽) Scaled Dot-Product Attention (오른쪽) Multi-head Attention은 병렬로 계산되는 여러개의 어텐션 레이어로 이뤄집니다.*</small>

여기서 사용한 어텐션 방식을 `"Scaled Dot-Product Attention"`이라고 하겠습니다. 인풋은 $$d_k$$차원의 query와 key, $$d_v$$차원의 value입니다. query와 key를 dot-products한 후 $$\sqrt{d_k}$$로 나누고, 소프트맥스함수를 취해 가중치를 구합니다. 

실제로는 여러개의 query 집합을 메트릭스 Q로 묶어 계산합니다. key와 value도 각 각 메트릭스 K와 V로 나타내어 메트릭스 형태의 아웃풋을 얻습니다. 

$$
Attention(Q, K, V) = softmax({QK^T\over{\sqrt{d_k}}})V
$$

널리 알려진 어텐션 함수는 `additive attention`과 `dot-product(multiplicative) attention`입니다. 여기서 사용한 알고리즘은 dot-product attention과 동일하지만, 스케일링을 위해 $$1\over{\sqrt{d_k}}$$로 나눠주는 것을 추가하었습니다. additive attention은 싱글 히든 레이어로 이루어진 피드-포워드 네트워크를 사용하는 방식입니다. 두가지 모두 이론적인 계산복잡도는 유사하지만, dot-product attention이 최적화된 메트릭스 멀티플리케이션 코드으로 구현할수 있기때문에 더 빠르고 효율적입니다. 

$$d_k$$가 작을 때는 두 메커니즘은 유사한 성능을 보이지만, $$d_k$$가 클때는 additive attention이 스케일링이 없는 dot-product attention보다 더 우수한 성능을 보입니다. 우리는 $$d_k$$가 클 때는 dot-product의 유효구간이 커지고, 이는 소프트맥스함수에서 그래디어트가 아주 작은 영역으로 가까워지게 하기 때문이라고 생각합니다.<sup>(\*)</sup> 이러한 효과를 줄이기 위해서 $$1\over{\sqrt{d_k}}$$로 나눠주었습니다. 

<small>(\*) dot-products의 유효구간이 커진다는 것을 설명하기 위해, q와 k가 평균 0이고 분산이 1인 독립적인 변수를 생각보겠습니다. q와 k의 dot-product는 $$q \cdot k = \sum_{i=1}^{d_k}q_ik_i$$이고, 이는 평균이 0이고 분산은 $$d_k$$ 가 됩니다. </small>

#### Multi-Head Attention

$$d_{model}$$ 차원의 keys, values, queries로 싱글 어텐션을 학습할수 있지만, 우리는 선형 프로젝션을 통해 h개의 어텐션을 이용하는 것이 더 효과적이라는 것을 발견하였습니다. 각 각의 프로젝션을 병렬로 계산하여, $$d_v$$-차원의 아웃풋값을 얻고, h개의 아웃풋 값들은 concatenate한 후 다시 프로젝션하여 최종 값을 계산합니다. 

`multi-head attention`은 서로 다른 represenation subspace에서의 정보를 결합하여 사용하는 것입니다. 싱글 어텐션은 평균값으로 인해 이러한 정보들이 없어져버립니다. 


$$
MultiHead(Q, K, V)=Concat(head_1, \cdots, head_h)W^O \\
where \ head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
$$

프로젝션을 위한 파라미터는 $$W_i^Q \in \mathbb{R}^{d_{model}\times d_k}$$, $$W_i^K \in \mathbb{R}^{d_{model}\times d_k}$$, $$W_i^V \in \mathbb{R}^{d_{model}\times d_v}$$, $$W_i^O \in \mathbb{R}^{hd_v\times d_{model}}$$입니다.  

실험에서 $$h = 8$$의 병렬 어텐션 레이어를 사용하였습니다. 또한 $$d_k = d_v =d_{model}/h = 64$$ 사용하였습니다. 각 헤드의 차원이 줄어들었지만, 전체적인 계산비용은 full dimensionality와 유사합니다. 

#### Application of Attention in our Model

Transformer는 multi-head attention을 3가지 방식으로 사용합니다. 

* 인코더-디코더 어텐션 레이어에서 queries는 이전 디코더 레이어로부터 입력되고, keys-valus는 인코더 아웃풋으로부터 입력됩니다. 따라서 디코더가 시퀀스의 매 포지션을 생성할때마다, 인풋 시퀀스의 모든 포지션 정보를 이용할수 있습니다. 이는 sequence-to-sequence모델에서 인코더-디코더 어텐션 메커니즘을 그대로 차용한 것입니다.

* 인코더는 셀프-어텐션 레이어를 포함합니다. 셀프 어텐션 레이어의 keys, values, queries는 이전 레이어의 아웃풋입니다. 레이어의 포지션 정보는 이전 레이어가 생성한 모든 포지션 정보를 다 이용합니다. 

* 유사하게 디코더도 셀프-어텐션 레이어를 포함하며, 디코더가 시퀀스의 매 포지션를 생성할때마다 그 위치까지의 모든 디코더 정보를 이용할수 있습니다. 다만 auto-regressive 속성을 유지하기 위해 소프트맥스의 인풋값 중 후속 포지션에 해당하는 값들은 모두 1로 마스킹합니다.

### Position-wise Feed-Forward Networks

어텐션 서브-레이어 이외에 인코더와 디코더 모두 `fully connected feed-forward network`를 서브레이어로 갖고 있습니다. 각 포지션별로 동일하게 적용되는 네트워크로 ReLU활성함수과 선형변환으로 구성됩니다. 

$$
FFN(x) = max(0, xW_1 + b_1)W_2 +b_2
$$

선형변화는 다른 포지션이더라도 동일한 선형 변환을 하지만, 레이어간에는 서로 다른 파라미터를 사용합니다. 이는 커널 사이즈가 1인 두개의 컨볼루션 오퍼레이션이라고도 생각할수 있습니다. 인풋과 아웃풋의 차원은 $$d_{model}=512$$이고, 레이어 내부(W)는 $$d_{ff} = 2048 $$입니다. 

### Embedding and Softmax

다른 시퀀스 변환 모델과 유사하게, 인풋과 아웃풋 토큰을 $$d_{model}$$차원의 벡터로 변환하는 임베딩을 학습합니다. 또한 디코더 아웃풋을 아웃풋 토큰 예측확률값으로 변환하기 위해 선형변환와 소프트맥스 함수를 사용하였습니다. 여기서는 두 임베딩 레이어와 소프트맥스앞의 선형변환에 대해서 모두 동일한 가중치 메트릭스를 사용하였습니다. 임베딩레이어에서는 가중치에 $$\sqrt{d_{model}}$$를 곱하여 사용하였습니다. 


### Positional Encoding

Transformer는 recurrence나 convolution을 사용하지 않기때문에, 시퀀스의 순서(order) 정보를 사용하기 위해서 시퀀스에서 토큰의 절대적인 위치나 상대적인 위치 정보를 강제로 입력해주어야 합니다. 따라서 포지셔널 인코딩을 인코더와 디코더의 가장 아래에 있는 인풋 임베딩 레이어에 추가하였습니다. 포지셔널 인코딩은 $$d_{model}$$과 동일한 차원으로 임베딩 벡터와 sum할수 있도록 하였습니다. 여러 종류의 포지셔널 인코딩이 있습니다만, 여기서는 서로 다른 주기의 `sine`과 `cosine`함수를 사용하였습니다. 

$$
PE_{(pos, 2i)} = sin(pos/10000^{2i/d_{model}}) \\
PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{model}})
$$

$$pos$$는 포지션이고, $$i$$는 디멘전입니다. 즉 포지셔널 인코딩의 각 차원은 sinusoid(사인모양의 파동)와 대응됩니다. 파장은 $$2i$$에서 $$10000\cdot 2i$$까지의 기하학적 진행을 의미합니다. 이 함수를 선택한 이유는 고정된 오프셋 $$k$$에 대해서 $$PE_{pos+k}$$는 $$PE_{pos}$$의 선형변환으로 쉽게 표현될수 있기때문에 모델이 상대적인 포지션 정보를 쉽게 학습할수 있을것이라 가정했기 때문입니다.

sinusoid 형태 외에 포지셔널 인코딩을 별도의 네트워크로 두고 학습하는 형태를 비교 실험해보았습니다. 그 결과, 두 버전 모두 거의 유사한 성능을 보였습니다(테이블3의 (E)). 최종적으로는 학습과정에서 마주친 것보다 더 긴 시퀀스에 대해서 extrapolate(외삽)할 수 있다는 점 때문에 sinusoid버전을 선택하였습니다. 

## Why Self-Attention

이 섹션에서는 self-attention layer를 recurrent나 convolutional layer와 비교하여 설명하겠습니다. 모두 $$(x_1, \cdots, x_n)$$의 시퀀스를 다른 시퀀스인 $$(z_1, \cdots, z_n) \ with \ x_i, z_i \in \mathbb{R^d}$$로 맵핑하는데 일반적으로 사용되는 레이어들입니다. 이 논문에서 셀프-어텐션을 사용한 이유는 세가지 관점 때문입니다. 

첫번째는 레이어 당 계산 복잡도를 고려했기 때문이고, 두번째는 병렬계산할수 있는 총 계산량으로 필요한 순차적 오퍼레이션의 최소 수를 이용해 정량화하였습니다. 


<img src = "/assets/img/2018-10-30/table1.png" width="600">

세번째는 네트워크에서 long-range dependencies간의 거리입니다. 멀리떨어진 단어들간의 의존성을 학습하는 것은 시퀀스 변환 문제에서 아주 중요한 이슈입니다. 이 이슈는 포워드와 백워드 시그널들이 네트워크에서 얼마나 이동할수 있는지에 영향을 받습니다. 인풋과 아웃풋 시퀀스들의 포지션 결합이 짧으면 짧을수록 장기 의존성은 쉽게 학습할수 있습니다. 따라서 우리는 네트워크 상에서 인풋과 아웃풋 포지션간의 최대 경로 길이를 측정하여 서로 다른 구조의 레이어를 비교 평가하였습니다. 

테이블1에서 볼수 있듯이 셀프 어텐션 레이어는 고정된 횟수만큼의 순차적인 오퍼레이션를 통해서 모든 포지션을 연결할수 있지만, recurrent layer는 $$O(n)$$만큼의 순차적 오퍼레이션이 필요합니다. 계산복잡도 측면에서 시퀀스의 길이 $$n$$이 representation 차원인 $$d$$보다 작을 때 (word-piece, byte-pair와 같은 일반적인 기계번역에서 SOTA모델이 사용하는 방식) 셀프어텐션 레이어가 recurrent layer보다 더 빠릅니다. 아주 긴 시퀀스를 다룰 때 계산복잡도를 개선하기 위해 셀프어텐션은 아웃풋 포지션 주위로 r개의 인접 포지션만으로 제약하여 고려하도록 할수 있습니다. (in future work)

커널 사이즈 $$k < n$$인 단일 컨볼루션 레이어는 인풋과 아웃풋 포지션의 모든 쌍을 연결하지 못합니다. 인풋과 아웃풋의 모든 포지션을 연결하기 위해서는 contiguous kernels을 이용하여 $$O(n/k)$$만큼의 콘볼루션 레이어를 쌓거나, dilated convlutions을 이용하여 $$O(log_k(n))$$를 쌓아야합니다. 이는 위에서 말한 경로 길이를 더 늘리는 것입니다. 컨볼루션 레이어는 일반적으로 recurrent layer보다 계산 비용이 더 큽니다. seperable convolution은 $$O(k \cdot n \cdot d + n \cdot d^2)$$만큼 복잡도를 줄일수 있습니다. $$k=n$$일때 seperable convolution는 이 논문에서 제안한 셀프-어텐션과 point-wise feed-forward layer를 결합한 것과 동일한 계산복잡도를 갖습니다. 

추가적으로 셀프-어텐션은 조금더 해석가능한 모델을 학습합니다. 어텐션 분포를 살펴보고 토의한 예제가 어펜딕스에 있습니다. 각각의 head가 명확하게 다른 작업을 수행하는 것을 학습할뿐만 아니라, 여러 head가 문장의 구조 및 의미 구조와 관련된 행동을 하는 것을 확인하였습니다. 

## Training

* WMT 2014 English-German dataset and WMT 2014 English-French dataset
* 8 NVIDIA P100 GPUs
* 100,000 steps or 12 hours (each traini step took about 0.4 seconds)
	* for big model(bottom line of table 3), step time : 1.0 sec, total 300,000 steps(3.5 days)
* Adam optimizer, $$\beta_1$$ = 0.9, $$\beta_2$$ = 0.98, $$\epsilon_1$$ = $$10^{-9}$$
	* learning rate : increasing linearly for the first warmup_steps, decreasing it thereafter proportionally the the inverse square root fo the step number 
	$$warmup\_steps$$ = 4000
	$$ lrate = d_{model}^{-0.5} \cdot min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})$$
* three types of regularization 
	* residual dropout : dropout to the output of each sub-layer, dropout to the sums of the embeddings and the positional encodings, $$P_{drop} = 0.1$$
	* Label Smoothing : label smoothing of value $$\epsilon_{ls} = 0.1$$

## Results

<img src = "/assets/img/2018-10-30/table2.png" width="600">

* WMT 2014 English-to-German : new state of the art BLEU 28.4
* WMT 2014 English-to-French : outperforming all previous single models with 1/4 traing cost of previous model
* single model obtained by averaging the last 5 checkpoints(for big modle, 20 checkpoints)
* beam searching with a beam size of 4 and length penalty $$\alpha = 0.6$$
	* beam searching? [뭐냐](https://ratsgo.github.io/deep%20learning/2017/06/26/beamsearch/)
* set the maximum output length during inferece to input length + 50 

<img src = "/assets/img/2018-10-30/table3.png" width="600">

표3은 Transformer의 여러 요소들의 중요도를 평가하기 위해서 베이스 모델을 변형하면서 실험데이터(English-to-German translation on the development set, newstest2013)에 대한 성능 변화를 확인한 내용입니다. 

(A) : 멀티헤드의 개수 - 싱글-헤드 어텐션은 최적모델 대비 0.9만큼 낮은 BLEU를 보였습니다. 너무 많은 헤드일때도 성능이 떨어집니다. <br>
(B) : key의 차원($$d_k$$)을 줄이면 성능이 떨어집니다. compatibility를 결정하는 것은 쉽지 않고, dot-product보다 더 정교한 함수가 유용할지 모른다는 것을 의미합니다. <br>
(C) & (D) : bigger models are better. dropout is very helpful in avoiding over-fitting <br>
(E) : sinusoidal positional encoding 대신에 learned positional embedding으로 변경한 결과, 거의 유사한 결과를 얻었습니다. (0.1 낮음)

## conclusion
이 논문은 어텐션만 사용하여 시퀀스 모델을 학습한 최초의 접근인 Transformer를 제안하였습니다. 향후에는 텍스트 이외에 이미지, 오디오, 비디오와 같은 인풋과 아웃풋 문제로 확장하고, 로컬로 제약된 어텐션 메커니즘을 연결할 계획입니다. 또한 덜 순차적인 방식으로 일반화시키는 것이 또 다른 목표입니다. 
