---
title: "Big Data Analysis with Scala and Spark "
categories: 
  - spark
  - scala
comments: true
published: false

---
https://www.coursera.org/learn/scala-spark-big-data/home/welcome

> foldLft 와 aggregate 둘다 inputType과 outputType이 다른데 왜 aggregate 만 병렬 처리가 가능한지 설명해주세요.

`foldLeft는 시퀀셜하게 처리되기 때문에 병렬처리가 불가능함. 만약 두개 콜렉션으로 나눠서 병렬처리한다고 했을때, 아웃풋 타입이 바뀌기때문에 두개의 아웃풋을 합치려고 할 때 타입 에러가 나서 더이상 동일한 함수를 적용할수 없음. 
aggregate는 seqop과 combop 펑션으로 이루어져있어, chunk로 나눠 처리된 결과를 combop함수를 통해 합칠수 있기 때문에 리턴 타입 변환과 병렬처리가 모두 가능함 `

> pairRDD는 어떤 데이터 구조에 적합한지 설명해주세요. 또 pairRDD는 어떻게 만드나요?
Key-value 형태로 구조화된 데이터를 다룰 때 유용함.

`이미 존재하는 RDD에서 map을 이용해서 아래와 같이 만들수 있음` <br>
`val rdd: RDD[WikipediaPage] = … `<br>
`val pairRdd = red.map(page => (page.title, page.text))`

> groupByKey()와 mapValues()를 통해 나온 결과를 reduceByKey()를 사용해서도 똑같이 만들 수 있습니다. 그렇지만 reduceByKey를 쓰는 것이 더 효율적인 이유는 무엇일까요? 

`reduceByKey는 싱글머신에 있는 데이터끼리 합친 후 셔플됨. 반면 groupByKey()는 모든 key-value 값들이 셔플되기때문에 네트워크 상에 데이터가 불필요하게 많이 이동하면서 계산됨
https://databricks.gitbooks.io/databricks-spark-knowledge-base/content/best_practices/prefer_reducebykey_over_groupbykey.html`


> join 과 leftOuterJoin, rightOuterJoin이 어떻게 다른지 설명하세요. 

`join(inner join) return a new RDD containing combined pairs whose keys are present in both input RDDs —inner join(join)은 두개의 RDD에 모두 포함되는 key만 포함하여 새로운 pair-RDD를 리턴함
Outer join(leftOuterJoin, rightOuterJoin) return a new RDD containing combined pairs whose keys don’t have to be present in both input RDDs — 왼쪽 혹은 오른쪽 RDD 중에서 유지하고 싶은 keys를 중심으로 결합하여 새로운 RDD를 만듦`


> Shuffling은 무엇인가요? 이것은 어떤 distributed data paraellism의 성능에 어떤 영향을 가져오나요? 

