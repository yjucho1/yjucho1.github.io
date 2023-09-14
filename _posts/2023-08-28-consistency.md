---
title: "time series representation learning with consistency "
categories: 
 - Time-series
 - consistency injection
comments: true
mathjax : true
published: true

---

## Temporal consistency 
Tonekaboni, Sana, Danny Eytan, and Anna Goldenberg. "Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding." International Conference on Learning Representations. 2020 

## Subseries consistency 
Franceschi, Jean-Yves, Aymeric Dieuleveut, and Martin Jaggi. "Unsupervised scalable representation learning for multivariate time series." Advances in neural information processing systems 32 (2019) 

## Transformation consistency 
Eldele, Emadeldeen, et al. "Time-series representation learning via temporal and contextual contrasting." arXiv preprint arXiv:2106.14112 (IJCAI 2021).
- weak augmentation : jitter-and scale 
- strong augmentation : permutation-and-jitter
- a tough cross-view prediction task by using the context of the strong
augmentation $$c^s_t$$ to predict the future timesteps of the weak augmentation $$z^w
_{t+k}$$ and vice versa  (temporal contrasting module)
- $$c_t^{i+}$$ as the positive sample of $$c_t^{i}$$ that comes from the other augmented view of the same inut (contextual contrasting module)

<img src = "/assets/img/2023-08-28/fig_transf.png" width=500><br>

## Contextual consistency 
Yue, Zhihan, et al. "Ts2vec: Towards universal representation of time series." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 8. 2022.
- random cropping : randomly samples two overlapping time segments [a1, b1], [a2, b2] a1<=a2<=b1<=b2 (the overlapped segment [a2, b1] should be consistent for two context views)
- timestamp masking : randomly mask the latent vector $$z_i$$
- hierarchical constrative loss encapsulate representations in different levels of granularity 

<img src = "/assets/img/2023-08-28/fig_hierarchical.png" ><br>
<img src = "/assets/img/2023-08-28/fig_contextual.png" ><br>

## Frequency consistency 
Zhang, Xiang, et al. "Self-supervised contrastive pre-training for time series via time-frequency consistency." Advances in Neural Information Processing Systems 35 (2022): 3988-4003.



TBU