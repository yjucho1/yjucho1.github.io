---
title: "Multi-Granularity Residual Learning with Confidence Estimation for Time Series Prediction"
categories: 
 - Time-series
 - consistency injection
comments: true
mathjax : true
published: true

---

<b>Hou, Min, et al. "Multi-Granularity Residual Learning with Confidence Estimation for Time Series Prediction." Proceedings of the ACM Web Conference 2022. 2022. </b>


$$
\mathcal{L} = \sum^S_{s=1}||y^s - \hat{y}^s||^2 + \lambda_1 \sum_{s=1}^{S} \mathcal{L}_{Rec} + \frac{\lambda_{\theta}}{2}||\Theta||_F^2
$$

$$
\mathcal{L}_{Rec} = \sum_{m=1,2,..m} ||F^m - G^{m-1}||^2_F
$$