---
title: "recommender system"
categories: 
  - recommender system
comments: true
mathjax : true

---

추천시스템에 대해서 알아보자! 앤드류응의 머신러닝 강의 중 추천시스템 부분에 대해서 정리하였습니다.

## problem formulation

| Movie | Alice(1) | Bob(2) | Carol(3) | Dave(4) |
|-------|----------|--------|----------|---------|
|Love at last| 5 | 5 | 0 | 0 |
|Romance forever | 5 | <b>?</b> | <b>?</b> | 0 |
|Cute puppies of love | <b>?</b> | 4| 0 | <b>?</b> |
|Nonstop car chases | 0 | 0 | 5 | 4 |
|Swords vs. karete | 0 | 0 | 5 | <b>?</b> |

$$ n_u $$ = number of users \\
$$ n_m $$ = number of movies \\
$$ r(i, j) $$ = 1 if user $$j$$ has rated movie $$i$$ \\
$$ y^{(i, j)} $$ = rating given by user $$j$$ to movie $$i$$ (defined only if $$ r(i, j) = 1 $$ )

> given $$r(i, j)$$ and $$y^{(i, j)}$$, we try to predict what these values of the question mark should be

## Content Based Recommendations

| Movie | Alice(1) | Bob(2) | Carol(3) | Dave(4) | $$ x_1 \\ (romance) $$ |  $$ x_2 \\ (action) $$
|----------|----------|--------|----------|---------|---|---|
|Love at last| 5 | 5 | 0 | 0 | 0.9 | 0 |
|Romance forever | 5 | <b>?</b>  | <b>?</b>  | 0 | 1.0 | 0.01|
|Cute puppies of love | <b>?</b>  | 4| 0 | <b>?</b>  | 0.99 | 0 |
|Nonstop car chases | 0 | 0 | 5 | 4 | 0.1 | 1.0 |
|Swords vs. karete | 0 | 0 | 5 | <b>?</b>  | 0 | 0.9 

For each user j, learn a prarameter $$\theta ^{(j)} \in \mathbb{R}^{n+1}$$ ( n is the number of features. In above example, n = 2 because of $$x_1,\ x_2 $$ ). 
Predict user $$j$$ as rating movie $$i$$ with $$(\theta ^{(j)})^T x^{(i)}$$ stars. ( $$x^{(i)}$$ = feature vector of movie $$i$$ )

<b>_example of Alice's rating of "Cute pupples of love"_</b> <small>(Assumed $$\theta ^{(1)}$$ learned by model) </small>

$$ 
x^{(3)} = 
\begin{bmatrix}
1  \\
0.99 \\
0 
\end{bmatrix} \ \ \ 
\theta^{(1)} = 
\begin{bmatrix}
0  \\
5 \\
0 
\end{bmatrix}
\\
(\theta ^{(1)})^T x^{(3)} = 5 * 0.99 = 4.95
$$

$$ \theta ^{(j)} $$ = parameter vector of user $$j$$ \\
$$ x ^{(i)} $$ = feature vector for movie $$i$$ \\
For user $$j$$, movie $$i$$, predicted rating : $$(\theta ^{(j)})^T x^{(i)}$$

<b>To learn $$\theta ^{(j)}$$ (parameter for user $$j$$)</b>: 

$$
\min_{\theta ^{(j)}} \frac{1}{2} \sum_{i:r(i,j)=1} ((\theta ^{(j)})^T x^{(i)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{k=1}^{n}(\theta_k^{(j)})^2
$$

<b>To learn  $$\theta ^{(1)},\theta ^{(2)}, ..., \theta ^{(n_u)} $$ </b>:

$$
\min_{\theta ^{(j)}} \frac{1}{2} \sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} ((\theta ^{(j)})^T x^{(i)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n}(\theta_k^{(j)})^2
$$

<b>Gradient descent update</b> :

$$ 
\theta_k^{(j)} := \theta_k^{(j)} -  \alpha \sum_{i:r(i,j)=1} ((\theta ^{(j)})^T x^{(i)} - y^{(i,j)})x_k^{(i)} \ \ (for \ k=0) \\
\theta_k^{(j)} := \theta_k^{(j)} -  \alpha \left( \sum_{i:r(i,j)=1} ((\theta ^{(j)})^T x^{(i)} - y^{(i,j)})x_k^{(i)} + \lambda \theta_k^{(j)} \right) \ \ (for \ k \neq 0) 
$$ 

## Collaborative Filtering


## Reference : 
