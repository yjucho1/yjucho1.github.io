---
title: "recommender system"
categories: 
  - recommender system
comments: true
mathjax : true

---

## problem formulation

| Movie | Alice(1) | Bob(2) | Carol(3) | Dave(4) |
|-------|----------|--------|----------|---------|
|Love at last| 5 | 5 | 0 | 0 |
|Romance forever | 5 | ? | ? | 0 |
|Cute puppies of love | ? | 4| 0 | ? |
|Nonstop car chases | 0 | 0 | 5 | 4 |
|Swords vs. karete | 0 | 0 | 5 | ? |

$$ n_u $$ = number of users \\
$$ n_m $$ = number of movies \\
$$ r(i, j) $$ = 1 if user $$j$$ has rated movie $$i$$ \\
$$ y^{(i, j)} $$ = rating given by user $$j$$ to movie $$i$$ (defined only if $$ r(i, j) = 1 $$ )

> given $$r(i, j)$$ and $$y^{(i, j)}$$, we try to predict what these values of the question mark should be

## Content Based Recommendations

| Movie | Alice(1) | Bob(2) | Carol(3) | Dave(4) | $$ x_1 \\ (romance) $$ |  $$ x_2 \\ (action) $$
|----------|----------|--------|----------|---------|---|---|
|Love at last| 5 | 5 | 0 | 0 | 0.9 | 0 |
|Romance forever | 5 | ? | ? | 0 | 1.0 | 0.01|
|Cute puppies of love | ? | 4| 0 | ? | 0.99 | 0 |
|Nonstop car chases | 0 | 0 | 5 | 4 | 0.1 | 1.0 |
|Swords vs. karete | 0 | 0 | 5 | ? | 0 | 0.9 

For each user j, learn a prarameter $$\theta ^{(j)} \in \mathbb{R}^{n+1}$$ ( n is the number of features. In above example, n = 2 because of $$x_1,\ x_2 $$ ). 
Predict user $$j$$ as rating movie $$i$$ with $$(\theta ^{(j)})^T x^{(i)}$$ stars. ( $$x^{(i)}$$ = feature vector of movie $$i$$ )

_example of Alice's rating of Cute pupples of love_ <small>(Assumed $$\theta ^{(1)}$$ learned by ML) </small>

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

$$m^{(j)}$$ = no. of movies rated by user $$j$$\\

To learn $$\theta ^{(j)}$$ (parameter for user $$j$$): 

$$
\min_{\theta ^{(j)}} \frac{1}{2} \sum_{i:r(i,j)=1} ((\theta ^{(j)})^T x^{(i)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{k=1}^{n}(\theta_k^{(j)})^2
$$

To learn  $$\theta ^{(1)},\theta ^{(2)}, ..., \theta ^{(n_u)} $$ :

$$
\min_{\theta ^{(j)}} \frac{1}{2} \sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} ((\theta ^{(j)})^T x^{(i)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n}(\theta_k^{(j)})^2
$$

<img src = "/assets/img/2018-10-20/gradient_descent_update.png" width="500">

## Collaborative Filtering


## Reference : 
