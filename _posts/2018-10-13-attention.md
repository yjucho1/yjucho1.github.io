---
title: "[번역] Attention? Attention!"
categories: 
  - Attention
comments: true
mathjax : true
published: false

---

> Attention은 최근 딥러닝 커뮤니티에서 자주 언급되는 유용한 툴입니다. 이 포스트에서는 어떻게 attention 개념이 만들어졌는지, 그리고 다양한 어텐션 매커니즘과 transformer와 SNAIL과 같은 모델들에 대해서 알아보고자 합니다.

- What’s Wrong with Seq2Seq Model?
- Born for Translation
    - Definition
- A Family of Attention Mechanisms
    - Summary
    - Self-Attention
    - Soft vs Hard Attention
    - Global vs Local Attention
- Transformer
    - Key, Value and Query
    - Multi-Head Self-Attention
    - Encoder
    - Decoder
    - Full Architecture
- SNAIL
- Self-Attention GAN
- References

Attention은 우리가 이미지에서 어떤 영역을 주목하는지, 한 문장에서 연관된 단어는 무엇인지를 찾는데서 유래하였습니다. 그림1에 있는 시바견을 살펴보세요. 

<img src = "/assets/img/2018-10-13/shiba-example-attention.png" width="500">

<small>*그림1. 사람옷을 입은 시바견. 이미지의 모든 권리는 인스타그램 [@mensweardog](https://www.instagram.com/mensweardog/?hl=en)에 있습니다.*</small>

인간은 이미지의 특정 부분을 고해상도로(노란 박스안에 뽀족한 귀) 집중하는 반면, 주변 부분들은 저해상도((눈이 쌓인 배경과 복장)로 인식하고 이후 초점영역을 조정하여 그에 따른 추론을 합니다. 이미지의 작은 패치가 가려져있을때, 나머지 영역의 픽셀들은 그 영역에 어떤 것이 들어가야 할지 알아맞추는데 힌트가 됩니다. 우리는 노란 박스 안은 뽀족한 귀가 있어야 하는 것을 알고 있습니다. 왜냐하면 우리는 개의 코, 오른쪽의 다른 귀, 시바견의 몽롱한 눈(빨란 박스안에 것들)를 봤기 때문입니다. 하지만 아래쪽에 있는 스웨터나 담요는 별 도움이 되지 못합니다. 

마찬가지로, 우리는 한 문장이나 가까운 문맥 상에서 단어들간의 관계를 설명할수 있습니다. "eating"이라는 단어를 보았을때, 음식 종류에 해당하는 단어가 가까이 위치에 있을 것을 예상할수 있습니다. 그림2에서 "green"은 eating과 더 가까이 위치해있지만 직접적으로 관련있는 단어는 아닙니다. 

<img src = "/assets/img/2018-10-13/sentence-example-attention.png" width="500">

<small>*그림2. 한 단어는 같은 문장의 단어들에 서로 다른 방식으로 주목하게 만듭니다.*</small>

간단히 말해, 딥러닝에서 어텐션은 weights의 중요도 벡터로 설명할수 있습니다. 이미지의 픽셀값이나 문장에서 단어 등 어떤 요소를 예측하거나 추정하기 위해, 다른 요소들과 얼마나 강하게 연관되어 있는지 확인하고(많은 논문들에서 읽은 것처럼) 이것들과 어텐션 백터로 가중 합산된 값의 합계를 타겟값으로 추정할 수 있습니다. 

## What’s Wrong with Seq2Seq Model?

seq2seq 모델은 언어 모델링에서 유래되었습니다. 간단히 말해서 입력 시퀀스를 새로운 시퀀스로 변형하는 것을 목적으로 하며, 이때 입력이나 결과값 모두 임의 길이를 갖습니다. 예를들면 텍스트, 오디오를 통한 기계번역, 질의응답 생성, 문장을 문법 트리로 구문 분석하는 작업 등이 있습니다.

seq2seq 모델은 보통 인코더-디코더 구조로 이루어져있습니다 :

* 인코더는 입력 시퀀스를 처리하여 고정된 길이의 컨텍스트 벡터(sentence embedding 또는 thought vector로도 알려진)로 정보를 압축합니다. 이러한 축소 표현은 소스 시퀀스의 컨텍스트적인 요약 정보로 간주할수 있습니다. 
* 디코더는 컨텍스트벡터를 다시 처리하여 변형된 결과값을 만들어 냅니다. 인코더 네트워크의 결과값만 이용해 변형을 수행합니다. 

인코더와 디코더 모두 [LSTM이나 GRU](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) 같은 Recurrent Neural Networks 구조를 사용합니다. 

<img src = "/assets/img/2018-10-13/encoder-decoder-example.png" width="500">

<small>*그림3. 인코더-디코더 모델, she is eating a green apple 이란 문장을 중국어로 변형함. 순차적인 방식으로 풀어서 시각화함*</small>

고정된 길이의 컨텍스트벡터로 디자인하는 것의 문제점은 아주 긴 문장의 경우, 모든 정보를 다 기억하지 못한다 것입니다. 일단 전체 문장을 모두 처리하고 나면 종종 앞 부분을 잊어버리곤 합니다. 어텐션 매커니즘은 이 문제점을 해결하기 위해 제안되었습니다. ([Bahdanau et al., 2015](https://arxiv.org/pdf/1409.0473.pdf))

## Born for Translation
어텐션 매카니즘은 딥러닝 기반의 기계번역([NMT](https://arxiv.org/pdf/1409.0473.pdf))에서 긴 소스 문장을 기억하기 위해서 만들어졌습니다. 인코더의 마지막 히든 스테이트의 컨텍스트 벡터뿐만아니라, 어텐션을 이용해 컨텍스트 벡터와 전체 소스 문장 사이에 지름길(shortcuts)을 만들어 사용하는 것입니다. 이 지름길의 가중치들은 각 아웃풋 요소들에 맞게 정의할 수 있습니다. 

컨텍스트벡터는 전체 입력 시퀀스에 접근할수 있고, 잊어 버릴 염려가 없습니다. 소스와 타겟 간의 정렬은 컨텍스트 벡터에 의해 학습되고 제어됩니다. 기본적으로 컨텍스트 벡터는 세가지 정보를 사용합니다. 

- 인코더 히든 스테이트
- 디코더 히든 스테이트
- 소스와 타겟 사이의 정렬

<img src = "/assets/img/2018-10-13/encoder-decoder-attention.png" width="500">

<small>*그림4. additive attention mechanism이 있는 인코더-디코더 모델 [Bahdanau et al., 2015](https://arxiv.org/pdf/1409.0473.pdf)*</small>

### Definition
조금 더 명료하게 NMT에서 사용되는 어텐션 매카니즘을 정의해보도록 하겠습니다. 길이가 $$n$$인 소스 문장 $$x$$를 이용해 길이가 $$m$$인 타겟 문장 $$y$$을 만들어보도록 하겠습니다. 

$$
\mathbf{x} = [x_1, x_2, ..., x_n] \\
\mathbf{y} = [y_1, y_2, ..., y_m]
$$

(볼드 표시된 변수는 벡터를 의미합니다. 이하의 모든 내용에 적용됩니다)

인코더는 [bidirectional RNN](https://www.coursera.org/lecture/nlp-sequence-models/bidirectional-rnn-fyXnn)(또는 다른 구조의 RNN를 갖을 수 있습니다)로 히든 스테이트 $$\mathbf{\overrightarrow{h_i}}$$ 와 반대방향 히든 스테이트 $$\mathbf{\overleftarrow{h_i}}$$를 갖습니다. 