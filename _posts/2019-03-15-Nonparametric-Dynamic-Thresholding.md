---
title: "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"
categories: 
 - Deep Learning paper
 - Time-series
comments: true
mathjax : true
published: false

---
<b>Kyle Hundman et al (KDD 2018, NASA)</b>

Implementation : [https://github.com/khundman/telemanom](https://github.com/khundman/telemanom)

## Abstract
* NASA의 우주선은 많은 양의 원격 데이터를 송신합니다. NASA 연구원들은 엔지니어의 모니터링 부담과 운영 비용을 줄이기 위해서 어노말리 디텍션 시스템을 개선하고 있습니다. 
* 이 논문은 expert-labeled 어노말리 데이터가 포함된 Soil Moiture Active Passive(SMAP) satelite와 Mars Science Laboratory(MSL) rover 데이터에 LSTM을 이용해 효율적으로 어노말리 디텍션 방법을 제안하였습니다. 
* 또한 파일럿 구현을 통해서 unsupervised and nonparametric anomaly thresholding approach를 제안하였으며, false positive를 줄이기 위한 방법들을 논의하였습니다.

## Introduction
* 우주선에서는 온도, 방사선, 전력, 소프트웨어의 계산량 등 복잡하고 방대한 양의 데이터들이 수집되며 각 데이터들의 어노말리 디텍션은 매우 중요한 문제입니다. 
* 기존의 어노말리 디텍션 방법들은 사전에 정의된 제한값을 벗어나면 발생하는 알람 형식이거나 수작업이 포함된 시각화 방법과 채널별 통계 분석을 이용합니다. 이러한 방식은 적지않은 전문가적 지식을 요구하며 각 규칙을 정의하거나 노말 범위를 업데이트하는데 수기 작업이 필요합니다. 특히 하루에 85테라바이트의 데이터가 생성되는 방대한 양의 데이터를 처리하는 빅데이터 시스템에서는 더욱 문제가 악화됩니다.
* 다변량 시계열 데이터의 어려운 이슈이 우주선 데이터를 분석하는데 여전히 유효하고, 라벨링 부족한 상황에서 비지도학습방식이나 세미지도학습방법의 필요성 역시 존재합니다. 또한 대부분의 실제 시계열 데이터가 그러하듯이 non-stationary한 특징과 현재 컨텍스에 매우 종속된 특징을 갖는 것들도 어려운 점입니다. 또한 엔지니어에게 인사이트를 줄수 있도록 interpretability도 필요한 요소입니다. 마지막으로 false positive와 false negative를 최소화하면서 적절한 발런스를 찾는 것 역시 중요합니다.

* <b> Contributions </b>
    * we describe our use of Long Short-Term Memory (LSTM) recurrent neural networks (RNNs) to achieve high prediction performance while maintaining interpretability throughout the system
    * Once model predictions are generated, we offer a nonparametric, dynamic, and unsupervised thresholding approach for evaluating residuals. This approach addresses diversity, non-stationarity, and noise issues associated with automatically setting thresholds for data streams characterized by varying behaviors and value ranges. Methods for utilizing user-feedback and historical anomaly data to improve system performance are also detailed

## Background and Related Work

## Method
### Telemetry Value Prediction with LSTMs
* single channel models : A single model is created for each telemetry channel and each model is used to predict values for that channel.
    * traceability down to the channel level
    * low-level anomalies can later be aggregated into various groupings and ultimately subsystems. --> granular control of the system
* predicting values for a channel : our aim is to predict telemetry values for a single channel we consider the situation where d = 1. We also use lp = 1 to limit the number of predictions for each step t and decrease processing time. the inputs $$x^{(t)}$$ into the LSTM consist of prior telemetry values for a given channel and encoded command information sent to the spacecraft
### Dynamic Error Thresholds
### Mitigating False Positives