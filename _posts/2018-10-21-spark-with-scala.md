---
title: "Big Data Analysis with Scala and Spark "
categories: 
  - spark
comments: true
published: true

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
other's best answer : RDD에서 transformation 이 수행 된 데이터를 재사용하는 경우 Lazy Execution이 Eager Excution보다 더 빠르다. 예를 들어 로그파일에서 "ERROR"가 포함된 로그만 필터링하기 위해 filter를 예약한 뒤 캐싱 기능을 호출한다. 이후 take를 이용하여 10개의 에러로그를 획득하는데, 액션이 호출되어 필터링이 수행되면서 필터링된 로그를 메모리에 캐싱을 하게 된다. 이처럼 캐싱된 에러 로그는 이후 총 개수를 구한다던가 하는 다른 actino이 호출될 때 메모리에 적재된 데이터를 이용하여 성능이 향상된다. 최초로 호출되는 action은 disk에서 읽어오는 것보다 조금 느릴 수 있다.
참고 - http://knight76.tistory.com/entry/%ED%8E%8C-lazy-evaluation%EB%8A%90%EA%B8%8B%ED%95%9C-%EA%B3%84%EC%82%B0%EB%B2%95%EC%97%90-%EB%8C%80%ED%95%9C-%EC%A2%8B%EC%9D%80-%EC%84%A4%EB%AA%85-%EA%B7%B8%EB%A6%BC-%EC%9E%90%EB%A3%8C

`foldLft 와 aggregate 둘다 inputType과 outputType이 다른데 왜 aggregate 만 병렬 처리가 가능한지 설명해주세요.`

foldLeft는 시퀀셜하게 처리되기 때문에 병렬처리가 불가능함. 만약 두개 콜렉션으로 나눠서 병렬처리한다고 했을때, 아웃풋 타입이 바뀌기때문에 두개의 아웃풋을 합치려고 할 때 타입 에러가 나서 더이상 동일한 함수를 적용할수 없음. aggregate는 seqop과 combop 펑션으로 이루어져있어, chunk로 나눠 처리된 결과를 combop함수를 통해 합칠수 있기 때문에 리턴 타입 변환과 병렬처리가 모두 가능함

`pairRDD는 어떤 데이터 구조에 적합한지 설명해주세요. 또 pairRDD는 어떻게 만드나요?`

Key-value 형태로 구조화된 데이터를 다룰 때 유용함. 이미 존재하는 RDD에서 map을 이용해서 아래와 같이 만들수 있음
val rdd: RDD[WikipediaPage] = … 
val pairRdd = rdd.map(page => (page.title, page.text))

`groupByKey()와 mapValues()를 통해 나온 결과를 reduceByKey()를 사용해서도 똑같이 만들 수 있습니다. 그렇지만 reduceByKey를 쓰는 것이 더 효율적인 이유는 무엇일까요?`

reduceByKey는 싱글머신에 있는 데이터끼리 합친 후 셔플됨. 반면 groupByKey()는 모든 key-value 값들이 셔플되기때문에 네트워크 상에 데이터가 불필요하게 많이 이동하면서 계산됨. 참고 - https://databricks.gitbooks.io/databricks-spark-knowledge-base/content/best_practices/prefer_reducebykey_over_groupbykey.html


`join 과 leftOuterJoin, rightOuterJoin이 어떻게 다른지 설명하세요. `

join(inner join) return a new RDD containing combined pairs whose keys are present in both input RDDs —inner join(join)은 두개의 RDD에 모두 포함되는 key만 포함하여 새로운 pair-RDD를 리턴함. Outer join(leftOuterJoin, rightOuterJoin) return a new RDD containing combined pairs whose keys don’t have to be present in both input RDDs — 왼쪽 혹은 오른쪽 RDD 중에서 유지하고 싶은 keys를 중심으로 결합하여 새로운 RDD를 만듦


`Shuffling은 무엇인가요? 이것은 어떤 distributed data paraellism의 성능에 어떤 영향을 가져오나요?`

we typically have to move data from one node to another to be "grouped with" its key. Doing this is called "shuffling". 인메모리 대비 노드간 네트워크 통신이 필요함으로 되도록이면 최소화하는 것이 좋음

`셔플링은 무엇이고 언제 발생하나요?`

데이터를 키값을 기준으로 그룹핑하여 한 노드에서 다른 노드로 이동시키기는 것. groupByKey()를 수행할 때 발생.

`파티션은 무엇인가요? 파티션의 특징을 2가지 알려주세요.`

pair-RDD를 키값을 중심으로 여러 노드에 나눠 저장하는 것
- 동일한 파티션에 있는 데이터들은 반드시 같은 머신에 존재한다.
- 클러스터 안에 한개의 머신에는 적어도 하나 이상의 파티션이 존재할수 있다. 
- 파티션의 수는 설정할수 있다. 기본적으로는 executor node의 코어 수와 같다. 


`스파크에서 제공하는 partitioning 의 종류 두가지를 각각 설명해주세요.`

Hash partitioning :튜플(k, v)마다 p=k.hashCode()%numPartitions 해서 p가 같은 데이터끼리 모아 파티셔닝하는 것. <br>
Range partitionin :ordering이 있는 key일 경우, 범위별로 데이터를 나누는 것

`파티셔닝은 어떻게 퍼포먼스를 높여주나요?`

키값을 중심으로 데이터가 어떤 머신에 있는지 알기 때문에 데이터가 셔플링되는 것을 최소화할수 있음

`rdd 의 toDebugString 의 결과값은 무엇을 보여주나요?`

RDD's lineage를 시각적으로 보여줌

`파티션 된 rdd에 map 을 실행하면 결과물의 파티션은 어떻게 될까요? mapValues의 경우는 어떤가요?`

map을 실행하면 파티션이 없어짐. map은 키를 바꿀수 있는 오퍼레이션 이기때문임. mapValues는 키값을 유지하기때문에 파티션도 유지됨.

`Narrow Dependency 와 Wide Dependency를 설명해주세요. 각 Dependency를 만드는 operation은 무엇이 있을까요?`

Narrow Dependency - 1개의 부모RDD는 최대 1개의 자식RDD에만 영향을 줌. map, filter, union, join with co-partitioned inputs <br>
Wide Dependency - 1개의 부모RDD가 여러개의 자식RDD에 영향을 줌. groupByKey, join with inputs not co-partitioned 

`Lineage 는 어떻게 Fault Tolerance를 가능하게 하나요?`
RDD는 immutable하고, 우리가 higher-order function을 사용하고, 그 function 역시 RDD를 리턴하기 때문에 lineage graphs를 통해서 dependency information을 추적하여 잃어버린 파티션을 다시 계산하여 failure로부터 회복할수 있다. 
