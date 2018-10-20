---
title: "recommender systems"
categories: 
  - recommender systems
comments: true
mathjax : true

---

추천시스템에 대해서 알아보자! 앤드류응의 머신러닝 강의 중 추천시스템 부분에 대해서 정리하였습니다.

## problem formulation

아래와 같이 4명의 유저가 5개 영화를 평가한 데이터가 있다고 하겠습니다. 추천시스템은 이와 같은 평점 데이터를 이용해, 유저가 아직 평가하지 않은 영화를 몇점으로 평가할지 예측하는 문제로 생각할 수 있습니다. 

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

우선 모든 영화가 로맨스인지, 액션인지 평가된 특징 정보가 있다고 가정하겠습니다. 아래 테이블은 각 영화의 특징에 해당하는 정보(피쳐 벡터), $$x_1$$과 $$x_2$$를 추가한 것입니다. 이 경우, 각 사용자의 평점은 피쳐벡서를 입력으로 하는 회귀 문제가 됩니다. 회귀문제에서 가중치 $$\theta$$는 사용자마다 다르며, 이미 평가한 데이터와 예측값의 오차를 최소화하는 방향으로 회귀식의 가중치을 학습할 수 있습니다. 

| Movie | Alice(1) | Bob(2) | Carol(3) | Dave(4) | $$ x_1 \\ (romance) $$ |  $$ x_2 \\ (action) $$
|----------|----------|--------|----------|---------|---|---|
|Love at last| 5 | 5 | 0 | 0 | 0.9 | 0 |
|Romance forever | 5 | <b>?</b>  | <b>?</b>  | 0 | 1.0 | 0.01|
|Cute puppies of love | <b>?</b>  | 4| 0 | <b>?</b>  | 0.99 | 0 |
|Nonstop car chases | 0 | 0 | 5 | 4 | 0.1 | 1.0 |
|Swords vs. karete | 0 | 0 | 5 | <b>?</b>  | 0 | 0.9 

For each user $$j$$, learn a prarameter $$\theta ^{(j)} \in \mathbb{R}^{n+1}$$ \\
( n is the number of features. In above example, n = 2 because of $$x_1,\ x_2 $$. By default $$x_0$$=0 )

Predict user $$j$$ as rating movie $$i$$ with $$(\theta ^{(j)})^T x^{(i)}$$ \\
( $$x^{(i)}$$ = feature vector of movie $$i$$ )

<b>_example of Alice's rating of "Cute pupples of love"_</b> \\
<small>(Assumed $$\theta ^{(1)}$$ learned by model) </small>

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
이제까지는 모든 영화가 로맨스인지, 액션인지 평가된 피쳐벡터가 존재한다는 가정을 하였습니다. 하지만 모든 영화를 보고 특징 정보를 정리하는 것은 매우 비용이 많이 드는 작업입니다. 이를 극복하기 위해서 반대로 사용자에게 로맨스 영화를 얼마나 좋아하는지, 액션 영화를 얼마나 좋아하는지를 조사하고, 이 값을 토대로 피쳐 벡터를 추정하는 방법을 택할 수 있습니다. 사용자 $$i$$가 응답한 정보를 $$\theta^{(i)}$$로 이용하여 아래와 같이 계산할 수 있습니다. 

| Movie | Alice(1) | Bob(2) | Carol(3) | Dave(4) | $$ x_1 \\ (romance) $$ |  $$ x_2 \\ (action) $$
|----------|----------|--------|----------|---------|---|---|
|Love at last| 5 | 5 | 0 | 0 | <b>?</b>| <b>?</b> |
|Romance forever | 5 | <b>?</b>  | <b>?</b>  | 0 | <b>?</b> | <b>?</b>|
|Cute puppies of love | <b>?</b>  | 4| 0 | <b>?</b>  | <b>?</b> | <b>?</b> |
|Nonstop car chases | 0 | 0 | 5 | 4 | <b>?</b> | <b>?</b>|
|Swords vs. karete | 0 | 0 | 5 | <b>?</b>  | <b>?</b> | <b>?</b>|

$$
\theta^{(1)} = 
\begin{bmatrix}
0  \\
5 \\
0 
\end{bmatrix} \ \ \ 
\theta^{(2)} = 
\begin{bmatrix}
0  \\
5 \\
0 
\end{bmatrix} \ \ \ 
\theta^{(3)} = 
\begin{bmatrix}
0  \\
0 \\
5 
\end{bmatrix} \ \ \ 
\theta^{(4)} = 
\begin{bmatrix}
0  \\
0 \\
5 
\end{bmatrix} \ \ \ 
$$

<b> Given $$\theta ^{(1)},\theta ^{(2)}, ..., \theta ^{(n_u)} $$, to learn $$x^{(i)}$$ </b> :

$$
\min_{x^{(i)}} \frac{1}{2} \sum_{j:r(i,j)=1} ((\theta ^{(j)})^T x^{(i)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{k=1}^{n}(x_k^{(i)})^2
$$

<b> Given $$\theta ^{(1)},\theta ^{(2)}, ..., \theta ^{(n_u)} $$, to learn $$x^{(1)}, ..., x^{(n_m)}$$ </b> :

$$
\min_{x^{(1)},...,x^{(n_m)}} \frac{1}{2} \sum_{i=1}^{n_m} \sum_{j:r(i,j)=1} ((\theta ^{(j)})^T x^{(i)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n}(x_k^{(i)})^2
$$

> Collaborative filtering refers to the observation that when you run this algorithm with a large set of users, all of these users are effectively doing collaboratively to get better movie ratings for everyone. because with every user rating, some subset of the movies, every user is helping the algorithm a little bit to learn better features. By rating a few movies myself, I would be hoping the systerm learn better features and then these features can be used by the system to make better movies predictions for everyone else. <u>So there's a sense of collaboration where every user is helping the system learn better features for the common good.</u>

Content based 방법은 영화별 특징 정보가 존재할 때, 유저별 가중치를 추정하는 것이고, Collaborative filtering은 유저별 가중치가 존재할 때 영화별 특징 정보를 추정하는 것입니다. 따라서 우리는 이 두가지를 결합하여, forward and backward 방식으로 초기값 $$\theta^{(i)}$$를 랜덤하게 설정한 후 피쳐벡터를 추정하고, 추정된 피쳐벡터로 다시 유저 가중치를 구할수 있습니다. 더 간단하게는 $$\theta^{(i)}$$와 $$x^{j}$$를 동시에 추정할 수 있습니다. 

if we are given $$ x^{(i)}$$, we can estimate $$\theta^{(i)}$$. Likewise if we are given $$\theta^{(i)}$$, we can estimate $$ x^{(i)}$$.

So we can initialize $$\theta^{(i)}$$ randomly, then estimate $$ x^{(i)}$$. After that, we update  $$\theta^{(i)}$$ and repeat. 

Putting togeter, we can solve for theta and x simultaneously.

$$
\min_{x^{(1)}, ... , x^{(n_m)}, \theta^{(1)}, ..., \theta ^{(n_u)}} J(x^{(1)}, ... , x^{(n_m)}, \theta^{(1)}, ..., \theta ^{(n_u)}) \\
J(x^{(1)}, ... , x^{(n_m)}, \theta^{(1)}, ..., \theta ^{(n_u)}) = \frac{1}{2} \sum_{(i, j):r(i,j)=1}  ((\theta ^{(j)})^T x^{(i)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n}(x_k^{(i)})^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n}(\theta_k^{(j)})^2
$$

note : in this version, $$x \in \mathbb{R}^{n}, \theta \in \mathbb{R}^{n}$$ 

1. Initialize $$x^{(1)}, ... , x^{(n_m)}, \theta^{(1)}, ..., \theta ^{(n_u)}$$ to small random values
2. Minimize $$ J(x^{(1)}, ... , x^{(n_m)}, \theta^{(1)}, ..., \theta ^{(n_u)})$$ using gradient descent(or an advanced optimization algorithm). E.g. for every $$j=1, ..., n_u, i=1,...,n_m$$ :
$$ 
x_k^{(i)} := x_k^{(i)} -  \alpha \left( \sum_{j:r(i,j)=1} ((\theta ^{(j)})^T x^{(i)} - y^{(i,j)})\theta_k^{(j)} + \lambda x_k^{(i)} \right) \\
\theta_k^{(j)} := \theta_k^{(j)} -  \alpha \left( \sum_{i:r(i,j)=1} ((\theta ^{(j)})^T x^{(i)} - y^{(i,j)})x_k^{(i)} + \lambda \theta_k^{(j)} \right) $$
3. For a user with parameters $$\theta$$ and a movie with (learned) features $$x$$, predict a star rating of $$\theta^Tx$$.

## Vectorization : Low Rank Matrix Factorization
위에서 설명한 것을 각 element별로 구하는 것이 아니라 matrix 형태로 vectorization하여 계산할 수도 있습니다. 

| Movie | Alice(1) | Bob(2) | Carol(3) | Dave(4) |
|-------|----------|--------|----------|---------|
|Love at last| 5 | 5 | 0 | 0 |
|Romance forever | 5 | <b>?</b> | <b>?</b> | 0 |
|Cute puppies of love | <b>?</b> | 4| 0 | <b>?</b> |
|Nonstop car chases | 0 | 0 | 5 | 4 |
|Swords vs. karete | 0 | 0 | 5 | <b>?</b> |

$$ 
Y = 
\begin{bmatrix}
5 & 5 & 0 & 0 \\
5 & ? & ? & 0 \\
? & 4 & 0 & ? \\
0 & 0 & 5 & 4 \\
0 & 0 & 5 & 0 \\
\end{bmatrix} \ \ \ \

predicted \ ratings : 
\begin{bmatrix}
(\theta ^{(1)})^T x^{(1)} & (\theta ^{(2)})^T x^{(1)} & \cdots & (\theta ^{(n_u)})^T x^{(1)} \\
(\theta ^{(1)})^T x^{(2)} & (\theta ^{(2)})^T x^{(2)} & \cdots & (\theta ^{(n_u)})^T x^{(2)} \\
\vdots & \vdots & \vdots & \vdots \\
(\theta ^{(1)})^T x^{(n_m)} & (\theta ^{(2)})^T x^{(n_m)} & \cdots & (\theta ^{(n_u)})^T x^{(n_m)} 
\end{bmatrix} 
$$

$$
X = 
\begin{bmatrix}
-(x^{(1)})^T- \\
-(x^{(2)})^T- \\
\cdots\\
-(x^{(n_m)})^T- \\
\end{bmatrix} \ \ \ 

\Theta = 
\begin{bmatrix}
| & | & \ & | \\
(\Theta^{(1)})^T & (\Theta^{(2)})^T & \cdots &(\Theta^{(n_u)})^T \\
| & | & \ & | \\
\end{bmatrix} 
\\
then, \\
predicted \ ratings = X\Theta
$$

> $$X\Theta$$ has mathematical property of low rank matrix

<b> How to find movies $$j$$ related to movie $$i$$ </b>
* small $$\parallel x^{(i)} - x^{(j)} \parallel$$ \to moving $$j$$ and $$i$$ are "similar"


## Implementation Detail Mean Normalization
지금까지 설명한 collaborative filtering은 평점 데이터가 없은 유저에 대해서는 항상 0값을 예측한다는 단점이 있습니다. 

| Movie | Alice(1) | Bob(2) | Carol(3) | Dave(4) | Eve(5) |
|-------|----------|--------|----------|---------|--------|
|Love at last| 5 | 5 | 0 | 0 | <b>?</b> |
|Romance forever | 5 | <b>?</b> | <b>?</b> | 0 | <b>?</b> |
|Cute puppies of love | <b>?</b> | 4| 0 | <b>?</b> | <b>?</b> |
|Nonstop car chases | 0 | 0 | 5 | 4 | <b>?</b> |
|Swords vs. karete | 0 | 0 | 5 | <b>?</b> | <b>?</b> |

<b>For Eve</b>, compute $$\theta^{(5)}$$ : 
let's say that n is equal to 2.

Since Eve rated no movies, there are no movies for which r(i, j) is equal to one. So the first term of the objective plays no role at all in determining $$\theta^{(5)}$$.

the only term that affects  $$\theta^{(5)}$$ is the last term

$$
\min \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n}(\theta_k^{(j)})^2 = \frac{\lambda}{2} 
\begin{bmatrix}
(\Theta_1^{(5)})^2 + (\Theta_2^{(5)})^2
\end{bmatrix} 
$$

Minimizing above term, we're going to end up with $$(\theta^{(5)})^T =[0 \ \ 0]$$. So all predicted ratings for Eve($$(\theta^{(5)})^T x^{(i)}$$) is equal to zero.

> This approach is not useful. The idea of mean normalization will let us fix this problem.

이를 해결하기 위해 일반적으로 mean normalization이라는 전처리 과정을 추가합니다. 직관적으로 평점데이터가 없는 유저의 경우, 영화별 평균 평점으로 예측하도록 하는 방법입니다.

$$ 
Y = 
\begin{bmatrix}
5 & 5 & 0 & 0 & ? \\
5 & ? & ? & 0 & ? \\
? & 4 & 0 & ? & ? \\
0 & 0 & 5 & 4 & ? \\
0 & 0 & 5 & 0 & ? \\
\end{bmatrix} \ \ \ \

\mu = 
\begin{bmatrix}
2.5\\
2.5\\
2\\
2.25\\
1.25\\
\end{bmatrix} \ \ \ \
\to
Y = 
\begin{bmatrix}
2.5 & 2.5 & -2.5 & -2.5 & ? \\
2.5 & ? & ? & -2.5 & ? \\
? & 2 & -2 & ? & ? \\
-2.25 & -2.25 & 2.75 & 1.75 & ? \\
-1.25 & -1.25 & 3.75 & -1.25 & ? \\
\end{bmatrix} \ \ \ \
$$

<b>For user $$j$$, on movie $$i$$ predict :
$$
(\theta^{(j)})^T(x^{i}) + \mu_i
$$

<b>For Eve :
$$
(\theta^{(j)})^T(x^{i}) + \mu_i = 0 + \mu_i = \mu_i
$$

> Mean normalization as a solid pre-processing step for collaborative filtering.

감사합니다!

## Reference : 
[Recommender Systems - Problem Formulation](https://www.youtube.com/watch?v=giIXNoiqO_U)

[Recommender Systems - Content Based Recommendations](https://www.youtube.com/watch?v=9siFuMMHNIA)

[Recommender Systems - Collaborative Filtering_1](https://www.youtube.com/watch?v=9AP-DgFBNP4)

[Recommender Systems - Collaborative Filtering_2](https://www.youtube.com/watch?v=YW2b8La2ICo)

[Recommender Systems - Vectorization Low Rank Matrix Factorization](https://www.youtube.com/watch?v=5R1xOJOFRzs)

[Recommender Systems - Implementational Detail Mean Normalization](https://www.youtube.com/watch?v=Am9fhp2Q91o)
