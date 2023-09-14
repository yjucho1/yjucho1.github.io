---
title: "Twin neural network regression is a semi-supervised regression algorithm"
categories: 
 - Time-series
 - consistency injection
comments: true
mathjax : true
published: true

---

<b>Wetzel, Sebastian J., Roger G. Melko, and Isaac Tamblyn. "Twin neural network regression is a semi-supervised regression algorithm." Machine Learning: Science and Technology 3.4 (2022): 045007.</b>

$$F(x_i, x_j) = y_i -y_j$$

$$
\begin{align}
F(x_1, x_2) + F(x_2, x_3) + f(x_3, x_1) &= (y_1 - y_2) + (y_2 -y_3) + (y_3 - y_1) \\
&= 0
\end{align}
$$

$$ loss = \frac{1}{n^2}\sum_{ij}(F(x_i, x_j)-(y_i-y_j))^2 + \lambda \frac{1}{n^3} \sum_{ijk}(F(x_i, x_j) + F(x_j, x_k) + f(x_k, x_i))^2$$