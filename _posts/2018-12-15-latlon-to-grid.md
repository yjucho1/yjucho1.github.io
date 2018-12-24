---
title: "위경도 - 기상청 격자 맵핑"
categories: 
  - Spatio-Temporal Data
comments: true
mathjax : false
published: true

---

기상청은 전국을 5km×5km 간격의 촘촘한 격자화하여 읍,면,동 단위로 상세한 날씨를 제공하는 동네예보를 제공합니다. 구역별 기상데이터를 관리하기 위해 한반도를 가로로 149개, 세로로 253개의 선을 그어 그리드형태로 관리하며, 위경도 데이터를 이 그리드 상의 좌표로 변화하는 알고리즘을 제공하고 있습니다. 

위경도 정보가 포함된 다양한 데이터를 기상청의 격자와 맵핑하면 날씨 데이터를 이용한 다양한 분석을 수행할 수 있습니다.

위경도 좌표를 기상청 격자로 변환하는 프로그램은 아래 오픈API의 활용가이드 문서 내에 공개되어 있습니다. 
* https://www.data.go.kr/dataset/15000099/openapi.do

C로 구현된 프로그램을 파이썬 버전으로 변경한 것은 아래와 같습니다.

```python
import math
NX = 149            ## X축 격자점 수
NY = 253            ## Y축 격자점 수

Re = 6371.00877     ##  지도반경
grid = 5.0          ##  격자간격 (km)
slat1 = 30.0        ##  표준위도 1
slat2 = 60.0        ##  표준위도 2
olon = 126.0        ##  기준점 경도
olat = 38.0         ##  기준점 위도
xo = 210 / grid     ##  기준점 X좌표
yo = 675 / grid     ##  기준점 Y좌표
first = 0

if first == 0 :
    PI = math.asin(1.0) * 2.0
    DEGRAD = PI/ 180.0
    RADDEG = 180.0 / PI


    re = Re / grid
    slat1 = slat1 * DEGRAD
    slat2 = slat2 * DEGRAD
    olon = olon * DEGRAD
    olat = olat * DEGRAD

    sn = math.tan(PI * 0.25 + slat2 * 0.5) / math.tan(PI * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(PI * 0.25 + slat1 * 0.5)
    sf = math.pow(sf, sn) * math.cos(slat1) / sn
    ro = math.tan(PI * 0.25 + olat * 0.5)
    ro = re * sf / math.pow(ro, sn)
    first = 1

def mapToGrid(lat, lon, code = 0 ):
    ra = math.tan(PI * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / pow(ra, sn)
    theta = lon * DEGRAD - olon
    if theta > PI :
        theta -= 2.0 * PI
    if theta < -PI :
        theta += 2.0 * PI
    theta *= sn
    x = (ra * math.sin(theta)) + xo
    y = (ro - ra * math.cos(theta)) + yo
    x = int(x + 1.5)
    y = int(y + 1.5)
    return x, y

def gridToMap(x, y, code = 1):
    x = x - 1
    y = y - 1
    xn = x - xo
    yn = ro - y + yo
    ra = math.sqrt(xn * xn + yn * yn)
    if sn < 0.0 :
        ra = -ra
    alat = math.pow((re * sf / ra), (1.0 / sn))
    alat = 2.0 * math.atan(alat) - PI * 0.5
    if math.fabs(xn) <= 0.0 :
        theta = 0.0
    else :
        if math.fabs(yn) <= 0.0 :
            theta = PI * 0.5
            if xn < 0.0 :
                theta = -theta
        else :
            theta = math.atan2(xn, yn)
    alon = theta / sn + olon
    lat = alat * RADDEG
    lon = alon * RADDEG

    return lat, lon

print(mapToGrid(37.579871128849334, 126.98935225645432))
print(mapToGrid(35.101148844565955, 129.02478725562108))
print(mapToGrid(33.500946412305076, 126.54663058817043))
### result :
#(60, 127)
#(97, 74)
#(53, 38)

print(gridToMap(60, 127))
print(gridToMap(97, 74))
print(gridToMap(53, 38))
### result
# 37.579871128849334, 126.98935225645432
# 35.101148844565955, 129.02478725562108
# 33.500946412305076, 126.54663058817043
```

위 알고리즘을 이용해 환경공단 제공의 초미세먼지 데이터를 시각화 예시 입니다.

<img src = "/assets/img/2018-12-15/fine-dust.png" width="400">

미세먼지 측정소 리스트를 조회할수 있는 OPEN API를 이용하면 아래와 같은 형태로 397개의 측정소 위치를 얻을 수 있습니다.

| station | lat | lon |
|---------|-----|-----|
|빛가람동|35.02174|126.790413|
|장성읍|35.303241|126.785419|
| ... | ....| .....|
|송파구|37.521597|127.124264|

<small><i>table : airkorea_stations</i></small> 

또한 대기오염 정보 조회 OPEN API를 이용하면 측정소별 실시간(1시간 단위) 대기오염 데이터를 얻을 수 있습니다.

| station | datatime  | PM2.5 | PM10 | 
|---------|-----|-----|-----|
|빛가람동|2018-12-13 14:00 |24|28|
|장성읍|2018-12-13 14:00 |37|31|
| ... | ....| .....| ....|
|송파구|2018-12-13 14:00 |45|35|

<small><i>table : airkorea_data</i></small> 

airkorea_stations에 있는 위경도를 기상청 격자 좌표로 변경하여 `gridx` 와 `gridy`로 저장합니다. 

이때, 주의할점은 격자 (1,1)에 대칭되는 점은 그리드의 좌하단이기때문에 실제 어레이의 포지션은 `grid_array[253+1-data.gridy, data.gridx]`로 되어야합니다.

시각화 예제 코드는 아래와 같습니다.

```python
## read data
con = sqlite3.connect("MyDataBase")
df = pd.read_sql("select * from airkorea_data a join airkorea_stations b on a.station=b.station;", con)

gridx, gridy = [], []
for idx, data in df.iterrows():
    x, y = mapToGrid(data.lat, data.lon)
    gridx.append(x), gridy.append(y)
df = df.assign(gridx = gridx, gridy = gridy)

startdate=dt(2018, 12, 13, 9, 0, 0)
background = plt.imread('background.png')

grid_array = np.empty((253+1, 149+1))
for idx, data in df.iterrows():
    try :
        grid_array[253+1-data.gridy, data.gridx] = int(data.pm10Value) 
    except :
        pass
fig = plt.figure(figsize=(10, 15))
masked_data = np.ma.masked_where(grid_array < 0.5, grid_array)
plt.imshow(background)
plt.imshow(masked_data, cmap='jet', vmin=0, vmax=100, alpha=1)
plt.colorbar()
plt.title(now.strftime("2018-12-13 14:00"))
plt.show()
```

<img src = "/assets/img/2018-12-15/background.png" width="230"> ►►
<img src = "/assets/img/2018-12-15/fine-dust.png" width="300">

Good Bye ~ !
