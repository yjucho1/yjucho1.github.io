var store = [{
        "title": "from GAN to WGAN",
        "excerpt":"이 글은 lilianweng의 from GAN to WGAN 포스팅을 동의하에 번역한 글입니다. From GAN to WGAN 이 포스트는 generative adversarial netowrk (GAN) model에 사용되는 수식과 GAN이 왜 학습하기 어려운지를 설명합니다. Wasserstein GAN은 두 분포간의 거리를 측정하는데 더 향상된(smoooth한) 메트릭를 사용하여 GAN의 학습과정을 개선하였습니다. Generative adversarial network(GAN)은 이미지나 자연어, 음성과 같은 현실의...","categories": ["Generative Adversarial Network"],
        "tags": [],
        "url": "http://localhost:4000/generative%20adversarial%20network/from-GAN-to-WGAN/",
        "teaser":null},{
        "title": "클러스터링을 평가하는 척도 - Mutual Information",
        "excerpt":"클러스터링은 주어진 데이터에 대한 명시적인 정보가 많지 않을 때 유용하게 쓸수있는 머신러닝 기법 중 하나입니다. 다양한 사용자 정보를 이용해 몇가지 고객군으로 분류하여 고객군별 맞춤 전략을 도출한다던지, 유사한 상품(동영상, 음원까지도)군의 속성을 분석하여 의미있는 인사이트를 도출하는 것에 활용됩니다. 클러스터링 알고리즘 측면에서는 전통적인 Hierarchical clustering, K-means clustering 등이 비교적 쉽게 사용되고 있고, 최근에는...","categories": ["Clustering"],
        "tags": [],
        "url": "http://localhost:4000/clustering/mutual-information/",
        "teaser":null},{
        "title": "클러스터링을 평가하는 척도 - Rand Index",
        "excerpt":"클러스터링을 평가하는 척도 - Mutual Information와 이어집니다. 클러스터링 결과를 평가하기 위해 Rand Index 도 자주 쓰입니다. Rand Index는 주어진 N개의 데이터 중에서 2개을 선택해 이 쌍(pair)이 클러스터링 결과 U와 V에서 모두 같은 클러스터에 속하는지, 서로 다른 클러스터에 속하는지를 확인합니다. Rand Index n개의 원소로 이루어진 집합 S={o1, … on}와 S를 r개의...","categories": ["Clustering"],
        "tags": [],
        "url": "http://localhost:4000/clustering/rand-index/",
        "teaser":null},{
        "title": "Paper I read ",
        "excerpt":"Unsupervised Deep Embedding for Clustering Analysis, J. Xie, R. Girshick, A. Farhadi (University of Washington, Facebook AI Reaserch), 2016 Visualizing and Understanding Convolutional Networks, Matthew D. Zeiler, Rob Fergus, 2013 ChoiceNet: Robust Learning by Revealing Output Correlations, Sungjoon Choi, Sanghoon Hong, Sungbin Lim (Kakao Brain, 2018) Generative Adversarial Nets, Ian...","categories": ["Deep Learning paper"],
        "tags": [],
        "url": "http://localhost:4000/deep%20learning%20paper/old-posts/",
        "teaser":null},{
        "title": "RNN for Quick drawing ",
        "excerpt":"Tutorial : https://www.tensorflow.org/tutorials/sequences/recurrent_quickdraw Code : https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/train_model.py 발번역 주의 Recurrent Neural Networks for Drawing Classification Quick, Draw!는 플레이어가 물체를 그리고, 컴퓨터가 그림을 인식해서 어떤 물체를 그린것인지 맞출 수 있는지 확인하는 게임입니다. Quick, Draw!에서는 사용자가 그린 그림에서 x,y의 점의 시퀀스를 입력으로 받아 학습된 딥러닝 모델이 사용자가 그렸던 물체의 카테고리를 맞추는 것으로 동작합니다....","categories": ["Tensorflow"],
        "tags": [],
        "url": "http://localhost:4000/tensorflow/quick-draw/",
        "teaser":null},{
        "title": "cs231n - 이해하기",
        "excerpt":"cs231n http://cs231n.stanford.edu/이 포스팅은 딥러닝에 대한 기본 지식을 상세히 전달하기보다는 간략한 핵심과 실제 모델 개발에 유용한 팁을 위주로 정리하였습니다. activation functions 1) sigmoid saturated neurons kill the gradient sigmoid outputs are not zero-centered2) tanh zero-cented but staturated neurons kill the gradient3) relu doest not saturate computationally efficient4) leaky relu 5) exponential...","categories": ["cs231n"],
        "tags": [],
        "url": "http://localhost:4000/cs231n/cs231n/",
        "teaser":null},{
        "title": "Quick drawing - dogs and cats",
        "excerpt":"개와 고양이는 어떻게 구분되는가 quick drawing은 구글에서 공개하는 오픈소스 데이터셋입니다. 345개 종류의 5백만장의 그림으로 이루어져있습니다. 이 포스팅에서는 그 중 개와 고양이 그림을 이용해 개와 고양이 그림을 구분하는 모델을 학습하고, 모델이 그림을 어떻게 인식하는지 시각화해보았습니다.import numpy as npimport matplotlib.pyplot as plt## Quick! drawing dataset## https://quickdraw.withgoogle.com/data## https://github.com/googlecreativelab/quickdraw-dataset## download : https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmapdogs = np.load('full%2Fnumpy_bitmap%2Fdog.npy')cats...","categories": ["Keras"],
        "tags": [],
        "url": "http://localhost:4000/keras/quick-drawing-exec/",
        "teaser":null},{
        "title": "cs231n - 이해하기 2",
        "excerpt":"cs231n http://cs231n.stanford.edu/이 포스팅은 딥러닝에 대한 기본 지식을 상세히 전달하기보다는 간략한 핵심과 실제 모델 개발에 유용한 팁을 위주로 정리하였습니다. Detection and Segmentation 1) semantic segmentation : sliding window Fully convolutional : labeling class per every pixel downsampling and upsampling : how to upsampling(unpooling) nearest neighbor bed of nails max unpooling(remember which...","categories": ["cs231n"],
        "tags": [],
        "url": "http://localhost:4000/cs231n/cs231n-2/",
        "teaser":null},{
        "title": "Attention? Attention!",
        "excerpt":"이 글은 lilianweng의 Attention? Attention! 포스팅을 번역한 글입니다.Attention은 최근 딥러닝 커뮤니티에서 자주 언급되는 유용한 툴입니다. 이 포스트에서는 어떻게 어텐션 개념과 다양한 어텐션 메커니즘을 설명하고 transformer와 SNAIL과 같은 모델들에 대해서 알아보고자 합니다. What’s Wrong with Seq2Seq Model? Born for Translation Definition A Family of Attention Mechanisms Summary Self-Attention Soft vs Hard...","categories": ["Attention"],
        "tags": [],
        "url": "http://localhost:4000/attention/attention/",
        "teaser":null},{
        "title": "recommender systems",
        "excerpt":"추천시스템에 대해서 알아보자! 앤드류응의 머신러닝 강의 중 추천시스템 부분에 대해서 정리하였습니다. problem formulation 아래와 같이 4명의 유저가 5개 영화를 평가한 데이터가 있다고 하겠습니다. 추천시스템은 이와 같은 평점 데이터를 이용해, 유저가 아직 평가하지 않은 영화를 몇점으로 평가할지 예측하는 문제로 생각할 수 있습니다. Movie Alice(1) Bob(2) Carol(3) Dave(4) Love at last 5...","categories": ["recommender systems"],
        "tags": [],
        "url": "http://localhost:4000/recommender%20systems/recommendation/",
        "teaser":null},{
        "title": "Big Data Analysis with Scala and Spark ",
        "excerpt":"https://www.coursera.org/learn/scala-spark-big-data/home/welcome Shared Memory Data Parallelism (SDP)와 Distributed Data Parallelism (DDP)의 공통점과 차이점을 얘기해주세요. 공통점 : 데이터를 나눠서, 병렬로 데이터를 처리한 후 결과를 합침(data-parallel programming). Collection abstraction을 처리할 수 있음. 차이점 : SDP의 경우 한 머신 내 메모리 상에서 데이터가 나눠져 처리가 일어나지만, DDP는 여러개의 노드(머신)에서 처리가 됨. DDP는 노드간의 통신이...","categories": ["spark"],
        "tags": [],
        "url": "http://localhost:4000/spark/spark-with-scala/",
        "teaser":null},{
        "title": "recommender systems 2",
        "excerpt":"추천시스템에 대해서 알아보자! - 지난 1편에서는 앤드류 응의 강의를 통해서 추천시스템의 전반적인 내용에 대해 알아보았습니다. 이번에는 Collaboratvie Filtering에 대해서 더 자세히 알아보고자 합니다. Collaborative filtering을 이용해 상품을 추천하는 방법은 크게 2가지 접근 방식이 있습니다. neighborhood method와 latent factor models 입니다. Neighborhood method neighborhood method는 아이템간 혹은 유저간 관계를 계산하는 것에...","categories": ["recommender systems"],
        "tags": [],
        "url": "http://localhost:4000/recommender%20systems/recommendation2/",
        "teaser":null},{
        "title": "Attention is All You Need",
        "excerpt":"Ashish Vaswani et al. (Google Brain), 2017 Tensorflow implementtation : https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py https://github.com/Kyubyong/transformerPyTorch implementation : https://github.com/jadore801120/attention-is-all-you-need-pytorch[guide annotating the paper with PyTorch implementation](http://nlp.seas.harvard.edu/2018/04/03/attention.html) Introduction 언어 모델링과 기계 번역과 같은 시퀀스 모델링에서 RNN, LSTM, GRU는 최신 기법으로 확고하게 자리잡고 있습니다. 인코더-디코더 구조를 활용하는 등 향상된 성능을 얻기 위해 많은 시도들이 있어왔습니다. recurrent models에서...","categories": ["Attention","Deep Learning paper"],
        "tags": [],
        "url": "http://localhost:4000/attention/deep%20learning%20paper/attention-is-all-you-need/",
        "teaser":null},{
        "title": "Clustering and Unsupervised Anomaly Detection with l2 Normalized Deep Auto-Encoder Representations",
        "excerpt":"Caglar Aytekin, Xingyang Ni, Francesco Cricri and Emre Aksu (Nokia) 2017 Introduction Recently, there are many works on learning deep unsupervised representations for clustering analysis. Works rely on variants of auto-encoders and use encoder outputs as representation/features for cluster. In this paper, l2 normalization constraint during auto-encoder training makes the...","categories": ["Clustering","Deep Learning paper"],
        "tags": [],
        "url": "http://localhost:4000/clustering/deep%20learning%20paper/clustering-with-l2-norm/",
        "teaser":null},{
        "title": "django를 이용한 대시보드 만들기",
        "excerpt":"django는 python 기반의 웹프레임워크로 비교적 쉽고 빠르게 웹어플리케이션을 제작할수 있도록 도와줍니다. django와 여러가지 오픈소스 라이브러리를 이용해 간단한 대시보드를 제작해보았습니다. 이 포스트에서는 1차 프로토타입을 소개하고, 사용한 라이브러리를 소개하도록 하겠습니다. Missions 데이터를 통한 인사이트 서비스를 제공하고 세상이 더 효율적으로 돌아가는데 기여하자눈에 보이는 유형의 서비스로 만들자 빠르게 만들고, 피드백을 받아 수정하자 Demo :...","categories": ["django"],
        "tags": [],
        "url": "http://localhost:4000/django/django/",
        "teaser":null},{
        "title": "위경도 - 기상청 격자 맵핑",
        "excerpt":"기상청은 전국을 5km×5km 간격의 촘촘한 격자화하여 읍,면,동 단위로 상세한 날씨를 제공하는 동네예보를 제공합니다. 구역별 기상데이터를 관리하기 위해 한반도를 가로로 149개, 세로로 253개의 선을 그어 그리드형태로 관리하며, 위경도 데이터를 이 그리드 상의 좌표로 변화하는 알고리즘을 제공하고 있습니다. 위경도 정보가 포함된 다양한 데이터를 기상청의 격자와 맵핑하면 날씨 데이터를 이용한 다양한 분석을 수행할...","categories": ["Spatio-Temporal Data"],
        "tags": [],
        "url": "http://localhost:4000/spatio-temporal%20data/latlon-to-grid/",
        "teaser":null},{
        "title": "Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction",
        "excerpt":"Junbo Zhang, Yu Zheng, Dekang Qi (Microsoft Research) 2017 Keras implementation : https://github.com/lucktroy/DeepST Introduction Forecating the flow of crowds In this paper, we predict two types fo crowd flows : inflow and outflow Inflow and outflow of crowds are affected by the following Spatial dependencies Temporal dependencies External influence :...","categories": ["Spatio-Temporal Data","Deep Learning paper"],
        "tags": [],
        "url": "http://localhost:4000/spatio-temporal%20data/deep%20learning%20paper/ST-resnet/",
        "teaser":null},{
        "title": "시계열 분석 part1 - Auto-Regressive, Moving-Average ",
        "excerpt":"시계열 데이터는 일정 시간동안 수집된 일련의 데이터로, 시간에 따라서 샘플링되었기 때문에 인접한 시간에 수집된 데이터는 서로 상관관계가 존재하게 됩니다. 따라서 시계열 데이터를 추론하는데, 전통적인 통계적 추론이 적합하지 않을수 있습니다. 현실 세계에서는 날씨, 주가, 마케팅 데이터 등등 다양한 시계열 데이터가 존재하고, 시계열 분석을 통해 이러한 데이터가 생성되는 메카니즘을 이해하여 설명가능한 모델로서...","categories": ["Spatio-Temporal Data","Time-series"],
        "tags": [],
        "url": "http://localhost:4000/spatio-temporal%20data/time-series/time-series-part1/",
        "teaser":null},{
        "title": "시계열 분석 part2 - practical example",
        "excerpt":"지난 포스팅에서는 시계열 데이터를 모델링하기 위한 모델과 모델의 파라미터를 추정하기 위한 이론적 배경을 살펴보았습니다. stationary하고 causal하다는 가정 하에서 Auto-Regressive 또는 Moving average, 또는 두가지가 섞인 ARMA 모델을 사용할 수 있고, 모델의 order를 결정하기 위해서 ACF와 PACF를 사용하는 방법을 학습하였습니다. 실제 분석 과정에서는 ACF, PACF 이외에 여러가지 방법을 복합적으로 사용하여 적합한...","categories": ["Spatio-Temporal Data","Time-series"],
        "tags": [],
        "url": "http://localhost:4000/spatio-temporal%20data/time-series/time-series-part2/",
        "teaser":null},{
        "title": "Deploying Keras models using TensorFlow Serving and Flask",
        "excerpt":"이 글은 Himanshu Rawlani의 Deploying Keras models using TensorFlow Serving and Flask을 참고하여 작성한 글입니다. 원글은 tensorflow serving 설치방법을 apt-get(리눅스 환경에서만 가능)를 이용하였으나, 본 글은 docker를 이용해 설치하는 방법을 이용해 MAC OSX에서 테스트한 내용을 추가하였습니다. 원글은 api서버로 Flask를 이용하였으나, 본 글은 django를 이용한 예제을 추가하였습니다.django APP의 전체코드는 이 깃헙 레포지토리를...","categories": ["Tensorflow"],
        "tags": [],
        "url": "http://localhost:4000/tensorflow/tensorflow-serving/",
        "teaser":null},{
        "title": "시계열 분석 part3 - Non-stationary",
        "excerpt":"Part1에서는 stationarity를 가정으로, 시계열 데이터의 기본 모델인 AR과 MA에 대해서 알아보고 모델의 파라미터를 추정하기 위해서 Yule-Walker Equations을 알아보았습니다. 또한 모델의 order를 결정하기 위해 ACF, PACF를 이용하는 방법을 살펴보았습니다. 실제 분석에서 모델의 파라미터를 추정하는 것은 소프트웨어를 이용해 자동적으로 계산되기 때문에 하나 하나를 기억할 필요는 없지만, 그 원리에 대해서 이해하는 것을 목적으로...","categories": ["Spatio-Temporal Data","Time-series"],
        "tags": [],
        "url": "http://localhost:4000/spatio-temporal%20data/time-series/time-series-part3/",
        "teaser":null},{
        "title": "시계열 분석 part4 - VAR",
        "excerpt":"지금까지는 시계열 데이터가 univariate일 경우를 모델링하는 방법을 알아보았습니다. 이번 포스팅에서는 여러개의 시계열 데이터가 존재할 경우, 즉 multivariate 모델인 Vector AutoRegression (VAR) Model을 알아보도록 하겠습니다. Bivariate time series에 대해서 알아보도록 하겠습니다. Bivariate time series는 2차원의 벡터 시리즈로 표현됩니다. at time t ( t= 1, 2, 3, …) 와 같이 표현하도록 하겠습니다....","categories": ["Spatio-Temporal Data","Time-series"],
        "tags": [],
        "url": "http://localhost:4000/spatio-temporal%20data/time-series/time-series-part4/",
        "teaser":null},{
        "title": "시계열 분석 part5 - ARMAX, ARFIMA, ARCH, GARCH",
        "excerpt":"지금까지 우리는 시계열 데이터를 설명하기 위해 ARMA모델을 살펴보고, non-stationary 시그널의 경우 differecing을 통해서 stationary 시그널을 얻은 후, ARMA를 적용하는 ARIMA 모델을 공부하였습니다. 또한 여러개의 시그널을 동시에 모델링하도록Vector AR 모델도 알아보았습니다. 이번 포스팅에서는 1) ARMA 모델에 exogenous(외적 요인) 입력이 추가된 형태인 ARMAX 모델과 2) 자연수 형태였던 difference order를 유리수로 확장하여 long-term...","categories": ["Spatio-Temporal Data","Time-series"],
        "tags": [],
        "url": "http://localhost:4000/spatio-temporal%20data/time-series/time-series-part5/",
        "teaser":null},{
        "title": "django - AWS 배포하기",
        "excerpt":"django application을 Amazon Web Service(AWS)에 배포하는 과정을 요약한 포스팅입니다. 이 블로그를 주로 참고하였고, 수행 중 발생하는 문제에 대한 trouble shooting 과정을 기억하기 위해 작성하였습니다. AWS 가입 후 EC2 서버 셋팅 계정 가입 후 콘솔 로그인 서비스 검색에 IAM(Identity and Access Management) 사용자 탭에 사용자 추가 엑세스 유형 : 프로그래밍 방식...","categories": ["django"],
        "tags": [],
        "url": "http://localhost:4000/django/django-deploy-aws/",
        "teaser":null},{
        "title": "시계열 분석 part6 - Spectral analysis",
        "excerpt":"지금까지 우리는 time domain에서의 여러가지 시계열 모델을 살펴보았습니다. 이번 포스팅은 주어진 시계열 데이터를 frequency domain에서 분석하는 방법에 대해서 설명하도록 하겠습니다. 수학적으로 다소 복잡해보이지만, 실제로는 numpy 등을 통해서 쉽게 활용할수 있는 방법입니다. 여기서는 이론적인 내용을 통해서 주파수 도메인에서의 개념을 직관적으로 이해하고, 실제 데이터를 통해서 결과값을 이해하고 활용할 수 있는 것을 목표로...","categories": ["Spatio-Temporal Data","Time-series"],
        "tags": [],
        "url": "http://localhost:4000/spatio-temporal%20data/time-series/time-series-part6/",
        "teaser":null},{
        "title": "Self-Supervised Generative Adversarial Networks",
        "excerpt":"Ting Chen et al. (Google Brain, 2018) Abstract Conditional GAN은 이미지 생성에서 탁월한 성능을 보이지만, 많은 양의 라벨링 데이터를 필요로 한다는 단점이 있습니다. 이 논문은 self-supervision learning과 adversarial training 기법을 적용하여 별도의 라벨링 없이도 image representations을 학습하고 좋은 품질의 이미지를 생성할 수 있음을 보였습니다.Introduction GAN의 학습은 고차원의 파라미터 공간에서 non-convex...","categories": ["Generative Adversarial Network","Deep Learning paper"],
        "tags": [],
        "url": "http://localhost:4000/generative%20adversarial%20network/deep%20learning%20paper/self-supervised-gan/",
        "teaser":null},{
        "title": "How does batch normalization help optimization?",
        "excerpt":"Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas et al. (2018, MIT) Abstract BatchNorm은 딥러닝의 안정적이고 학습속도를 빠르게 하는데 도움을 주는 기법으로 널리 활용되고 있습니다 . 하지만 그 활용성에 비해서 왜 BatchNorm이 효과적인지에 대한 실질적인 고찰은 거의 없었으며, 대부분은 internal covariance shift를 줄이는 효과를 줄이기 때문이라고 믿고 있습니다. 이 논문에서는 internal covariance...","categories": ["Deep Learning paper"],
        "tags": [],
        "url": "http://localhost:4000/deep%20learning%20paper/batchnorm/",
        "teaser":null}]
