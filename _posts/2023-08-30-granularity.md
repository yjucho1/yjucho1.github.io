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



$$
\mathcal{L} = \sum^S_{s=1}||y^s - \hat{y}^s||^2 + \lambda_1 \sum_{s=1}^{S} \mathcal{L}_{Rec} + \lambda_2 \sum_{x=1}^S \sum_{m=1}^M \sum_{t=1}^T \mathcal{L}_N^C + \frac{\lambda_{\theta}}{2}||\Theta||_F^2
$$

$$
\mathcal{L}_N^C = - E_{\mathcal{P}}
$$


$$ 
c_m^{t} = AR(h_m^{<t})
$$

$$
\hat{y} =\mathcal{F}_\theta(X^1, ..., X^M) \\
X^m = [x_1^m, ..., x_T^m] \in \mathit{R}^{D \times K^m \times T}
$$

$$
\begin{align}
F^m &= \mathcal{F}_{Linear}^m(X^m) \\ F^m &\in \mathit{R}^{D \times K \times T}
\end{align}
$$

$$
\hat{y}
$$

dd

$$
\hat{y}_1 \\
\\
\hat{y}_2 \\
\\
\hat{y}_3 \\
$$