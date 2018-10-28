---
title: "recommender systems 2"
categories: 
  - recommender systems
comments: true
mathjax : true

---

추천시스템에 대해서 알아보자! - 지난 1편에서는 앤드류 응의 강의를 통해서 추천시스템의 전반적인 내용에 대해 알아보았습니다. 이번에는 Collaboratvie Filtering에 대해서 더 자세히 알아보고자 합니다. 

Collaborative filtering을 이용해 상품을 추천하는 방법은 크게 2가지 접근 방식이 있습니다. `neighborhood method`와 `latent factor models` 입니다.

## Neighborhood method
`neighborhood method`는 아이템간 혹은 유저간 관계를 계산하는 것에 중점을 둡니다. 

유저 기반의 방법은 해당 유저와 유사한 다른 유저를 찾은 후, 비슷한 유저가 좋아하는 아이템을 추천하는 방식입니다. 그림1에서처럼, 세가지 영화를 좋아하는 Joe를 위해서, 세가지 영화를 동일하게 좋아하는 비슷한 유저를 찾습니다. 이들이 좋아하는 영화 중에서 가장 인기있는 영화인 Saving Private Ryan(라이언 일병 구하기, denoted #1)를 추천할 수 있습니다. 

<img src = "/assets/img/2018-10-28/user-based-CF.png" width="500">

<small>*그림1. user-oriented neighborhood method (Image source: Fig 1 in [Yehuda Koren et al., 2009](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf))*</small>

아이템 기반의 방법은 해당 유저가 좋아하는 아이템과 유사한 아이템을 추천하는 방식입니다. 유사한 아이템은 해당 유저에게 동일한 평가를 받을 가능성이 크기 때문입니다. 예를 들어, Saving Private Ryan와 유사한 영화는 전쟁 영화거나, 톰행크스가 나오거나, 스필버그 감동의 다른 영화일 수 있습니다. 만약 누군가가 Saving Private Ryan를 어떻게 평가할지 궁금하다면, 그 사람이 실제로 본 영화 중에서 Saving Private Ryan와 유사한 영화를 어떻게 평가했는지 찾는 것과 같은 맥락입니다. 

## similarity : pearson correlation vs. cosine similarity 

## Matrix Factorization

## SGD vs. ALS
