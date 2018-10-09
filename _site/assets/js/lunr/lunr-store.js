var store = [{
        "title": "[번역] from GAN to WGAN",
        "excerpt":"이 글은 lilianweng의 from GAN to WGAN 포스팅을 동의하에 번역한 글입니다. From GAN to WGAN 이 포스트는 generative adversarial netowrk (GAN) model에 사용되는 수식과 GAN이 왜 학습하기 어려운지를 설명합니다. Wasserstein GAN은 두 분포간의 거리를 측정하는데 더 향상된(smoooth한) 메트릭를 사용하여 GAN의 학습과정을 개선하였습니다. Generative adversarial network(GAN)은 이미지나 자연어, 음성과 같은 현실의...","categories": ["Generative Adversarial Network","Wasserstein GAN"],
        "tags": [],
        "url": "http://localhost:4000/generative%20adversarial%20network/wasserstein%20gan/from-GAN-to-WGAN/",
        "teaser":null},{
        "title": "클러스터링을 평가하는 척도 - Mutual Information",
        "excerpt":"클러스터링은 주어진 데이터에 대한 명시적인 정보가 많지 않을 때 유용하게 쓸수있는 머신러닝 기법 중 하나입니다. 다양한 사용자 정보를 이용해 몇가지 고객군으로 분류하여 고객군별 맞춤 전략을 도출한다던지, 유사한 상품(동영상, 음원까지도)군의 속성을 분석하여 의미있는 인사이트를 도출하는 것에 활용됩니다. 클러스터링 알고리즘 측면에서는 전통적인 Hierarchical clustering, K-means clustering 등이 비교적 쉽게 사용되고 있고, 최근에는...","categories": ["Clustering Evaluation"],
        "tags": [],
        "url": "http://localhost:4000/clustering%20evaluation/mutual-information/",
        "teaser":null},{
        "title": "클러스터링을 평가하는 척도 - Rand Index",
        "excerpt":"클러스터링을 평가하는 척도 - Mutual Information와 이어집니다. 클러스터링 결과를 평가하기 위해 Rand Index 도 자주 쓰입니다. Rand Index는 주어진 N개의 데이터 중에서 2개을 선택해 이 쌍(pair)이 클러스터링 결과 U와 V에서 모두 같은 클러스터에 속하는지, 서로 다른 클러스터에 속하는지를 확인합니다. Rand Index n개의 원소로 이루어진 집합 S={o1, … on}와 S를 r개의...","categories": ["Clustering Evaluation"],
        "tags": [],
        "url": "http://localhost:4000/clustering%20evaluation/rand-index/",
        "teaser":null},{
        "title": "[old] Paper I read ",
        "excerpt":"Unsupervised Deep Embedding for Clustering Analysis, J. Xie, R. Girshick, A. Farhadi (University of Washington, Facebook AI Reaserch), 2016 Visualizing and Understanding Convolutional Networks, Matthew D. Zeiler, Rob Fergus, 2013 ChoiceNet: Robust Learning by Revealing Output Correlations, Sungjoon Choi, Sanghoon Hong, Sungbin Lim (Kakao Brain, 2018) Generative Adversarial Nets, Ian...","categories": ["migration"],
        "tags": [],
        "url": "http://localhost:4000/migration/old-posts/",
        "teaser":null},{
        "title": "RNN for Quick drawing ",
        "excerpt":"Tutorial : https://www.tensorflow.org/tutorials/sequences/recurrent_quickdraw Code : https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/train_model.py 발번역 주의 Recurrent Neural Networks for Drawing Classification Quick, Draw!는 플레이어가 물체를 그리고, 컴퓨터가 그림을 인식해서 어떤 물체를 그린것인지 맞출 수 있는지 확인하는 게임입니다. Quick, Draw!에서는 사용자가 그린 그림에서 x,y의 점의 시퀀스를 입력으로 받아 학습된 딥러닝 모델이 사용자가 그렸던 물체의 카테고리를 맞추는 것으로 동작합니다....","categories": ["Tensorflow Tutorial"],
        "tags": [],
        "url": "http://localhost:4000/tensorflow%20tutorial/quick-draw/",
        "teaser":null},{
        "title": "cs231n - 이해하기",
        "excerpt":"cs231n http://cs231n.stanford.edu/이 포스팅은 딥러닝에 대한 기본 지식을 상세히 전달하기보다는 간략한 핵심과 실제 모델 개발에 유용한 팁을 위주로 정리하였습니다. activation functions 1) sigmoid saturated neurons kill the gradient sigmoid outputs are not zero-centered2) tanh zero-cented but staturated neurons kill the gradient3) relu doest not saturate computationally efficient4) leaky relu 5) exponential...","categories": ["cs231n"],
        "tags": [],
        "url": "http://localhost:4000/cs231n/cs231n/",
        "teaser":null},{
        "title": "Quick drawing - dogs and cats",
        "excerpt":"개와 고양이는 어떻게 구분되는가 quick drawing은 구글에서 공개하는 오픈소스 데이터셋입니다. 345개 종류의 5백만장의 그림으로 이루어져있습니다. 이 포스팅에서는 그 중 개와 고양이 그림을 이용해 개와 고양이 그림을 구분하는 모델을 학습하고, 모델이 그림을 어떻게 인식하는지 시각화해보았습니다.import numpy as npimport matplotlib.pyplot as plt## Quick! drawing dataset## https://quickdraw.withgoogle.com/data## https://github.com/googlecreativelab/quickdraw-dataset## download : https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmapdogs = np.load('full%2Fnumpy_bitmap%2Fdog.npy')cats...","categories": ["Keras","Visualizing filters"],
        "tags": [],
        "url": "http://localhost:4000/keras/visualizing%20filters/quick-drawing-exec/",
        "teaser":null}]
