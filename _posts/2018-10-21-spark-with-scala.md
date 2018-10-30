---
title: "Big Data Analysis with Scala and Spark "
categories: 
  - spark
  - scala
comments: true
published: false

---
https://www.coursera.org/learn/scala-spark-big-data/home/welcome

`Shared Memory Data Parallelism (SDP)와 Distributed Data Parallelism (DDP)의 공통점과 차이점을 얘기해주세요.`

공통점 : 데이터를 나눠서, 병렬로 데이터를 처리한 후 결과를 합침(data-parallel programming). Collection abstraction을 처리할 수 있음. <br>차이점 : SDP의 경우 한 머신 내 메모리 상에서 데이터가 나눠져  처리가 일어나지만, DDP는 여러개의 노드(머신)에서 처리가 됨. DDP는 노드간의 통신이 필요하기 때문에 latency를 고려해야함

`분산처리 프레임워크 Haddop의 Fault Tolerance는 DDP의 어떤 문제를 해결했나요?`

Computations on unthinkably large data sets to succeed to completion. 수백~수천개의 노드로 확장가능하도록 함. 노드 중 한개라도 failure 발생하더라도 recover할 수 있음

`Spark가 하둡과 달리 데이터를 메모리에 저장하면서 개선한 것 무엇이고, 왜 메모리에 저장하면 그것이 개선이 되나요?`

latency를 개선함. 같은 job에 대해서 100배 이상의 성능을 보임. Functional programming을 통해 latency를 줄임

`val ramyons = List("신라면", "틈새라면", "너구리")` <br>
`val kkodulRamyons = ramyons.map(ramyon => "꼬들꼬들 " + ramyon)` <br>
`kkodulRamyonsList.map()을 사용하여 ramyons 리스트에서 kkodulRamyon List를 새로 만들었습니다. kkodulRamyons랑 똑같이 생긴 List를 만드는 Scala 코드를 써주세요.`

val kkodulRamyons = List(“꼬들꼬들 신라면", "꼬들꼬들 틈새라면", "꼬들꼬들 너구리")

`val noodles = List(List("신라면", "틈새라면", "너구리"), List("짜파게티", "짜왕", "진짜장"))`
`val flatNoodles = noodles.flatMap(list => list)`
`flatNoodlesList.flatmap() 을 사용하여 noodles 리스트에서 flatNoodles List를 새로 만들었습니다. flatNoodles랑 똑같이 생긴 List를 만드는 Scala 코드를 써주세요.`

val flatNoodles = List("신라면", "틈새라면", "너구리”, “짜파게티", "짜왕", "진짜장")

`val jajangs = flatNoodles.filter(noodle => noodle.contains("짜"))`
`jajangsList.filter() 를 사용하여 flatNoodles 리스트에서 jajangs List를 새로 만들었습니다. jajangs랑 똑같이 생긴 List를 만드는 Scala 코드를 써주세요.`

val jajangs = List(“짜파게티”, "짜왕", "진짜장")

`val jajangMenu = jajangs.reduce((first, second) => first +"," + second)`
`jajangMenuList.reduce()를 사용하여 jajangs 리스트에서 jajangMenu String을 만들었습니다. jajangMenu랑 똑같이 생긴 String을 만드는 Scala 코드를 써주세요.`

var jajangMenu = “짜파게티,짜왕,진짜장”

`Eager execution와 Lazy execution의 차이점은 무엇인가요?`

Lazy execution은 결과값이 바로 계산되지 않고 eager execution은 결과가 바로 계산됨. spark의 transformation은 lazy execution이라 action이 나타날때까지 실제로는 아무것도 수행되지 않음

`Transformation과 Action의 결과물 (Return Type)은 어떻게 다를까요?`

Transformation은 새로운 RDD를 결과물로 리턴하고, Action은 HDFS같은 외부 저장소에 값을 리턴함

`RDD.cache()는 어떤 작동을 하고, 언제 쓰는 것이 좋은가?`

Action을 수행할때마다 RDD를 다시 계산하는 것은 시간적 비용이 크기때문에 메모리에 캐시 저장함

`Lazy Execution이 Eager Execution 보다 더 빠를 수 있는 예를 얘기해주세요.`

logistic regression처럼 Iterative algoritm의 경우, 값(weight)이 업데이트될때마다 데이터를 평가하하는 경우 매 iteration마다 반복 계산되는 것은 계산 비용이 큼. 따라서 RDD를 persist()나 cache()를 이용해 메모리에 저장하는 방법이 효과적임


`foldLft 와 aggregate 둘다 inputType과 outputType이 다른데 왜 aggregate 만 병렬 처리가 가능한지 설명해주세요.`

foldLeft는 시퀀셜하게 처리되기 때문에 병렬처리가 불가능함. 만약 두개 콜렉션으로 나눠서 병렬처리한다고 했을때, 아웃풋 타입이 바뀌기때문에 두개의 아웃풋을 합치려고 할 때 타입 에러가 나서 더이상 동일한 함수를 적용할수 없음. aggregate는 seqop과 combop 펑션으로 이루어져있어, chunk로 나눠 처리된 결과를 combop함수를 통해 합칠수 있기 때문에 리턴 타입 변환과 병렬처리가 모두 가능함

`pairRDD는 어떤 데이터 구조에 적합한지 설명해주세요. 또 pairRDD는 어떻게 만드나요?`

Key-value 형태로 구조화된 데이터를 다룰 때 유용함.
이미 존재하는 RDD에서 map을 이용해서 아래와 같이 만들수 있음
val rdd: RDD[WikipediaPage] = … 
val pairRdd = red.map(page => (page.title, page.text))

`groupByKey()와 mapValues()를 통해 나온 결과를 reduceByKey()를 사용해서도 똑같이 만들 수 있습니다. 그렇지만 reduceByKey를 쓰는 것이 더 효율적인 이유는 무엇일까요?`

reduceByKey는 싱글머신에 있는 데이터끼리 합친 후 셔플됨. 반면 groupByKey()는 모든 key-value 값들이 셔플되기때문에 네트워크 상에 데이터가 불필요하게 많이 이동하면서 계산됨. 참고 - https://databricks.gitbooks.io/databricks-spark-knowledge-base/content/best_practices/prefer_reducebykey_over_groupbykey.html


`join 과 leftOuterJoin, rightOuterJoin이 어떻게 다른지 설명하세요. `

join(inner join) return a new RDD containing combined pairs whose keys are present in both input RDDs —inner join(join)은 두개의 RDD에 모두 포함되는 key만 포함하여 새로운 pair-RDD를 리턴함. Outer join(leftOuterJoin, rightOuterJoin) return a new RDD containing combined pairs whose keys don’t have to be present in both input RDDs — 왼쪽 혹은 오른쪽 RDD 중에서 유지하고 싶은 keys를 중심으로 결합하여 새로운 RDD를 만듦


`Shuffling은 무엇인가요? 이것은 어떤 distributed data paraellism의 성능에 어떤 영향을 가져오나요?`

