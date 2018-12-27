---
title: "Clustering and Unsupervised Anomaly Detection with l2 Normalized Deep Auto-Encoder Representations"
categories: 
  - Clustering
  - Deep Learning paper
comments: true
mathjax : true
published: true

---

<b> Caglar Aytekin, Xingyang Ni, Francesco Cricri and Emre Aksu (Nokia) 2017 </b>

## Introduction
* Recently, there are many works on learning deep unsupervised representations for clustering analysis.
* Works rely on variants of auto-encoders and use encoder outputs as representation/features for cluster.
* In this paper, l<sub>2</sub> normalization constraint during auto-encoder training makes the representations more separable and compact in the Euclidean space.

## Related Work
* DEC : First, dense auto-encoder is trained with minimizing reconstruction error. Then, as clustering optimization state, minimizing the KL divergence between auto-encoder representation and an auxiliary target distribution.
	* [DEC paper](https://github.com/yjucho1/articles/blob/master/DEC/readme.md)

* IDEC : proposes to jointly optimize the clustering loss and reconstruction loss of the auto-encoder
* DCEC : adopts a convolutional auto-encoder
* GMVAE : adopts variational auto-encoder

## Proposed Method
* Clustering on l<sub>2</sub> normalized deep auto-encoder representations

$$
L = \frac{1}{|J|} \sum_{j \in J} (I_j - D(E_c(I_j)))^2, \\
E_c(I) = \frac{E(I)}{\parallel E(I) \parallel _2}
$$

* after training auto-encoder with loss function, the clustering is simply performed by k-means algorithm.

* Unsupervised Anomaly Detection using l<sub>2</sub> normalized deep auto-encoder representations

$$
v_i = max_j (E_c(I_i) \cdot \frac{C_j}{\parallel C_j \parallel _2} )
$$


## Experimental result
* clustering : evaluation metrics - accuracy 

<img src = "/assets/img/2018-11-22/dense-AE.png" width="600">

<img src = "/assets/img/2018-11-22/conv-AE.png" width="600">

<img src = "/assets/img/2018-11-22/comparison-norm.png" width="600">

* comparision of normalization method : neither batch nor layer normalization provides a noticeable accuracy increase over CAE + k-means. Moreover in MNIST dataset, layer and batch normalization results into a significant accuracy decrease. 
* This is an important indicator showing that the performance upgrade of our method is not a result of a input conditioning, but it is a result of the specific normalization type that is more fit for clustering in Euclidean space.


* anomaly detection : evaluation metrics - AUC 

<img src = "/assets/img/2018-11-22/anomaly-detection.png" width="600">

