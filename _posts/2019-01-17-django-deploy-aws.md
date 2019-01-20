---
title: "django - AWS 배포하기"
categories: 
  - django
comments: true
mathjax : true
published: true
toc : true
---

django application을 Amazon Web Service(AWS)에 배포하는 과정을 요약한 포스팅입니다. [이 블로그](https://nachwon.github.io/django-deploy-1-aws/)를 주로 참고하였고, 수행 중 발생하는 문제에 대한 trouble shooting 과정을 기억하기 위해 작성하였습니다.  

## 아마존 웹 서비스 (AWS) 가입하기 

1. 계정 가입 후 콘솔 로그인 
    - 서비스 검색에 IAM(Identity and Access Management)

    <img src = "/assets/img/2019-01-17/1.png" width='400'>
    <img src = "/assets/img/2019-01-17/2.png" width='400'>

2. 사용자 탭에 사용자 추가
    - 엑세스 유형 : 프로그래밍 방식 액세스
    - 기존 정책 직접 연결 : AmazonEC2FullAccess
    - 완료 창의 Access key ID 와 Secret access key는 꼭 저장해두어야 합니다. "download.csv"를 눌러 저장합니다.

3. EC2 서비스로 이동 
4. 키페어 생성 - pem 파일 다운로드 
    - 다운로드한 pem 파일은 ~/.ssh 폴더에 보관한다.
    - chmod 400 pem파일 로 권한을 변경합니다.
5. 인스턴스 생성
    - Ubuntu Server 16.04
    - 보안 그룹 이름 및 설명 입력 
    - 검토 후 시작 클릭 후 생성한 키페어 선택합니다. 
6. 생성된 인스턴스에 sss 접속 
    - ssh -i 키페어경로 유저명@EC2퍼블릭DNS주소
7. 서버 환경 설정 
    - locale 설정

    ```
    sudo vi /etc/default/locale

    LC_CTYPE="en_US.UTF-8"
    LC_ALL="en_US.UTF-8"
    LANG="en_US.UTF-8"
    ```

    ```
    sudo apt-get update
    sudo apt-get dist-upgrade
    sudo apt-get install python-pip
    sudo apt-get install zsh
    sudo curl -L http://install.ohmyz.sh | sh
    sudo chsh ubuntu -s /usr/bin/zsh
    ```
8. pyenv 설치
    - 먼저 Ubuntu에서 Build 할 때 공통적으로 발생하는 문제를 방지하기 위해 필요한 패키지들을 설치해준다.
    ```
    $ sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev
    ```
    - git clone 후 설치해줍니다. 
    ```
    $ git clone https://github.com/pyenv/pyenv.git ~/.pyenv
    ```
    - ~/.zshrc 의 pyenv 환경변수 설정을 해줍니다. 
    ```
    export PATH="/home/ubuntu/.pyenv/bin:$PATH"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    ```
9. Python 설치
    - pyenv를 통해서 Python을 설치합니다.
    ```
    pyenv install 3.6.7
    ```

    - Pillow를 위한 Python 라이브러리 설치합니다. 
    ```
    sudo apt-get install python-dev python-setuptools
    ```

10. scp를 사용하여 django 프로젝트 파일 업로드하기
    ```
    scp -i 키페어경로 -r 보낼폴더경로 유저명@퍼블릭DNS:받을폴더경로
    ```

11. 서버에서 Python 가상환경 설치하기
    - AWS 서버에 로컬 서버에서 생성했던 pyenv 가상환경 이름과 동일한 이름으로 가상환경을 생성합니다. 
    ```
    pyenv virtualenv 3.6.7 mysite
    ```
    - 다음의 명령어를 입력하여 requirements.txt 에 기재되어있는 패키지들을 설치해줍니다. 
    ```
    pip install -r requirements.txt
    ```
    - 만약 pip 버전이 최신버전이 아니라는 에러가 날 경우 아래 명령어를 입력해준 다음 다시 설치합니다. 
    ```
    pip install --upgrade pip
    ```

12. 보안 그룹에 포트 추가하기
    - EC2 관리 화면으로 접속한 뒤, 보안 그룹 화면으로 이동합니다.
    - 보안 그룹 목록에서 생성한 보안 그룹을 체크하고 인바운드 탭의 편집 버튼을 누릅니다.
    - 규칙 추가 버튼을 누른 다음, 포트 범위에 8080 을 입력하고 저장을 누릅니다. 

<img src = "/assets/img/2019-01-17/3.png" width='400'>

13. runserver 실행하기
    - srv 폴더안의 프로젝트 폴더로 이동하여 runserver 를 포트 8080에 실행합니다.
    ./manage.py runserver 0:8080
    - 위의 모든 과정이 올바르게 수행되었다면 django application 화면이 보일 것입니다.


## WSGI와 NGINX


웹 서버 게이트웨이 인터페이스(`WSGI`, `Web Server Gateway Interface`)는 웹서버와 웹 애플리케이션의 인터페이스를 위한 파이선 프레임워크입니다. runserver 는 개발용이므로 실제 서비스를 운영하는데 부적합하기 때문에 실제로 어플리케이션을 서비스할 때는 웹서버를 사용하게 됩니다. 또한 웹서버가 직접적으로 Python으로 된 장고와 통신할 수 없기 때문에 그 사이에서 WSGI Server(middleware) 가 실행되어 웹서버와 장고를 연결해주는 역할을 합니다.  웹서버가 전달받은 사용자의 요청을 WSGI Server에서 처리하여 Django로 넘겨주고, 다시 Django가 넘겨준 응답을 WSGI Server가 받아서 웹서버에 전달하게 됩니다. WSGI Server에는 여러 가지 종류가 있는데, 그 중 기능이 강력하고 확장성이 뛰어난 `uWSGI` 를 사용하도록 하겠습니다. 

웹 서버(`Web Server`)는 HTTP를 통해 웹 브라우저에서 요청하는 HTML 문서나 오브젝트(이미지 파일 등)을 전송해주는 서비스 프로그램을 말합니다. 웹 서버의 주된 기능은 웹 페이지를 클라이언트로 전달하는 것입니다. 주로 그림, CSS, 자바스크립트를 포함한 HTML 문서가 클라이언트로 전달됩니다. 주된 기능은 콘텐츠를 제공하는 것이지만 클라이언트로부터 콘텐츠를 전달 받는 것도 웹 서버의 기능에 속하고, 클라이언트에서 제출한 웹 폼을 수신하는 것이 그 예에 해당합니다. 여기서는 성능에 중점을 둔 차세대 웹 서버 소프트웨어인 `Nginx`를 사용하겠습니다. 

[referece](https://nachwon.github.io/django-deploy-2-wsgi/)


1. uWSGI 설치
    - `ssh`로 접속 후 배포에 사용할 유저 `deploy` 를 생성합니다. 
    ```
    sudo adduser deploy
    ```
    - uWSGI를 설치할 별도의 python 가상환경을 생성합니다. 
    ```
    pyenv virtualenv 3.6.7 uwsgi-env
    ```
    - 이 가상환경을 지금 현재의 가상 컴퓨터 셸에만 일시적으로 적용하도록 설정해줍니다. 서버 전체에서 하나의 uwsgi를 사용하게 하기 위함입니다. 
    ```
    pyenv shell uwsgi-env
    ```
    - 이제 가상환경에 uwsgi 를 설치합니다.
    ```
    pip install uwsgi
    ```

2. uWSGI로 서버 열어보기
    - uWSGI를 실행하려면 pyenv shell uwsgi-env 를 입력해 uwsgi-env를 적용한 다음, 아래와 같이 입력합니다. 
    ```
    uwsgi \
    --http :[포트번호] \
    --home [virtualenv 경로] \
    --chdir [장고프로젝트폴더 경로] \
    -w [wsgi 모듈명].wsgi
    ```
    ```
    uwsgi \
    --http :8080 \
    --home /home/ubuntu/.pyenv/versions/mysite \
    --chdir /srv/air-pollution/mysite \
    -w  mysite.wsgi
    ```

3. ini 파일로 uWSGI 실행하기
    - 매번 uWSGI를 실행할 때마다 위의 복잡한 명령을 입력하기 번거로우므로, 미리 옵션을 파일로 만들어 저장해놓고 실행할 수 있습니다.
    - 로컬에서 장고 프로젝트 폴더에 .config 라는 폴더를 하나 새로 생성하고 그 안에 다시 uwsgi 폴더를 생성하고, uwsgi 폴더 안에 `mysite.ini` 파일을 만들어 줍니다. 
    
    ```
    air-pollution
    ├── .config
    │   └── uwsgi
    │       ├── mysite.ini
    ```
    
    `mysite.ini` :
    ```
    [uwsgi]
    chdir = /srv/air-pollution/mysite
    module = mysite.wsgi:application
    home = /home/ubuntu/.pyenv/versions/mysite

    uid = deploy
    gid = deploy

    http = :8080

    enable-threads = true
    master = true
    vacuum = true
    pidfile = /tmp/mysite.pid
    logto = /var/log/uwsgi/mysite/@(exec://date +%%Y-%%m-%%d).log
    log-reopen = true
    ```
    - uWSGI를 실행하기 전에 mysite.ini 파일에 설정해주었던 `logto` 옵션의 디렉토리를 생성합니다. 
    ```
    sudo mkdir -p /var/log/uwsgi/mysite
    ```
    - 그 다음 아래의 명령을 실행해 ini 파일로 uWSGI를 실행합니다. sudo 권한으로 실행해야 하기 때문에, uwsgi-env 가상환경 폴더 안에 있는 uwsgi를 직접 실행해주어야 합니다. 
    ```
    sudo /home/ubuntu/.pyenv/versions/uwsgi-env/bin/uwsgi -i /srv/air-pollution/.config/uwsgi/mysite.ini 
    ```
4. Nginx 설치
    ```
    # PPA 추가를 위한 필요 패키지
    sudo apt-get install software-properties-common python-software-properties

    # nginx 안정화 최신버전 PPA 추가
    sudo add-apt-repository ppa:nginx/stable

    # PPA 저장소 업데이트
    sudo apt-get update

    # nginx 설치
    sudo apt-get install nginx
    ```
    - 유저 설정
    배포에 관한 작업은 `deploy` 유저가 담당하므로 Nginx 의 유저를 `deploy` 로 바꿔줍니다.
    Nginx 관련 설정은 /etc/nginx/nginx.conf 에서 관리합니다.
    ```
    sudo vi /etc/nginx/nginx.conf
    ```
    - 파일의 첫 줄 user www-data; 를 user deploy; 로 수정해줍니다.

    ```
    user deploy;
    worker_processes auto;
    pid /run/nginx.pid;
    include /etc/nginx/modules-enabled/*.conf;

    events {
            worker_connections 768;
            # multi_accept on;
    }

    http {
        ...
    ```

    - Nginx 설정 파일 생성 및 연결
    이제 로컬 서버로 빠져나가서 장고 프로젝트 폴더로 이동합니다. 
    uWSGI 설정을 저장했던 .config 폴더에 nginx 폴더를 새로 만들고 그 아래에 `mysite.conf` 파일을 생성합니다. 
    ```
    air-pollution
    ├── .config
    │   ├── nginx
    │   │   └── mysite.conf
    │   └── uwsgi
    │       ├── mysite.ini
    ```
    `mysite.conf` :
    ```
    server {
        listen 80;
        server_name *.compute.amazonaws.com;
        charset utf-8;
        client_max_body_size 128M;

        location / {
            uwsgi_pass  unix:///tmp/mysite.sock;
            include     uwsgi_params;
        }
    }
    ```
    - 장고 프로젝트 폴더 내의 `mysite.conf` 파일을 `/etc/nginx/sites-available/` 경로에 복사해줍니다.
    ```
    sudo cp -f /srv/air-pollution/.config/nginx/mysite.conf /etc/nginx/sites-available/mysite.conf
    ```
    - 이제 다음 명령을 입력하여 sites-available 에 있는 설정파일을 sites-enabled 폴더에 링크해줍니다.
    ```
    sudo ln -sf /etc/nginx/sites-available/mysite.conf /etc/nginx/sites-enabled/mysite.conf
    ```
    - sites-enabled 폴더의 default 링크는 삭제해줍니다. 
    ```
    sudo rm /etc/nginx/sites-enabled/default
    ```

5. uWSGI 설정
    - 이제 uWSGI를 Nginx와 통신하도록 설정해줍니다. 
    - 리눅스에서 관리하는 service 파일을 만들어 서버가 실행될 때 자동으로 uWSGI를 백그라운드에 실행시켜주도록 해야합니다.
    - /장고 프로젝트 폴더/.config/uwsgi/ 에 uwsgi.service 파일을 생성합니다.
    ```
    air-pollution
    ├── .config
    │   ├── nginx
    │   │   └── mysite.conf
    │   └── uwsgi
    │       ├── mysite.ini
    │       └── uwsgi.service
    ```
    - uwsgi.service 파일안에 아래와 같이 작성합니다.

    ```
    [Unit]
    Description=uWSGI service
    After=syslog.target

    [Service]
    ExecStart=/home/ubuntu/.pyenv/versions/uwsgi-env/bin/uwsgi -i /srv/air-pollution/.config/uwsgi/mysite.ini

    Restart=always
    KillSignal=SIGQUIT
    Type=notify
    StandardError=syslog
    NotifyAccess=all

    [Install]
    WantedBy=multi-user.target
    ```

    - AWS 서버에 접속해서 uwsgi.service 파일을 /etc/systemd/system/ 에 하드링크를 걸어줍니다. 
    ```
    sudo ln -f /srv/air-pollution/.config/uwsgi/uwsgi.service /etc/systemd/system/uwsgi.service
    ```
    - 파일을 연결해준 뒤 아래 명령을 실행해서 데몬을 리로드 해줍니다. 
    ```
    sudo systemctl daemon-reload
    ```
    - 그 다음 아래 명령어로 uwsgi 데몬을 활성화 해줍니다. 이제 서버에 접속하기만 해도 uwsgi와 Nginx가 백그라운드에서 실행됩니다.
    ```
    sudo systemctl enable uwsgi
    ```
    - 소켓 통신 설정 : `mysite.ini` 파일을 열어 http = :8080 을 삭제하고 그 부분에 아래와 같이 추가합니다. uWSGI가 http 요청을 받는 대신, /tmp/mysite.sock 파일을 통해 요청을 받도록 소켓 통신을 설정해주는 것입니다.

    `mysite.ini` 
    ```
    [uwsgi]
    chdir = /srv/air-pollution/mysite
    module = mysite.wsgi:application
    home = /home/ubuntu/.pyenv/versions/mysite

    uid = deploy
    gid = deploy

    socket = /tmp/mysite.sock
    chmod-socket = 666
    chown-socket = deploy:deploy

    enable-threads = true
    master = true
    vacuum = true
    pidfile = /tmp/mysite.pid
    logto = /var/log/uwsgi/mysite/@(exec://date +%%Y-%%m-%%d).log
    log-reopen = true
    ```
    - 데몬 리로드로 다시 불러와주고, Nginx와 uWSGI를 재부팅해줍니다. 
    ```
    sudo systemctl daemon-reload
    sudo systemctl restart nginx uwsgi
    ```

5. AWS 서버 설정
    - mysite.conf 파일을 보면 `listen 80`과 `server_name *.compute.amazonaws.com`부문이 있습니다. listen 80 은 요청을 80번 포트를 통해 받도록 설정하는 것이고, server_name 의 *.compute.amazonaws.com 는 서버의 URL 주소입니다. 
    - 80번 포트는 웹 브라우저에서 기본적으로 요청을 보내는 포트인데, 아직 AWS 서버의 보안 그룹에 등록되어 있지 않기 때문에 80번 포트를 등록시켜주어야합니다.

    <img src = "/assets/img/2019-01-17/4.png" width='400'>

    - 설정이 끝났으므로 브라우저에서 접속해보면 django app 모습이 나타납니다. 
    - 만약 에러가 난다면 아래의 명령으로 에러 로그를 확인해서 문제점을 찾을 수 있습니다.

    ```
    # Nginx 에러 로그
    cat /var/log/nginx/error.log

    # uWSGI 로그
    cat /var/log/uwsgi/mysite/로그작성날짜.log
    ```

## static files을 S3에 저장하기

## 외부도메인 연결을 위해 Route 53 사용하기 

## trouble shooting

> <b>어드민 로그인 시 "attempt to write a readonly database" 에러 발생</b><br>
> db.sqlite3의 권한 문제로 아래 명령어를 이용해 권한 부여를 해주어야합니다. 이때 db.sqlite3가 위치한 부모 디렉토리에도 권한이 부여되여있어야하니 주의하셔야합니다.

```
sudo chgrp [deploy를 수행하는 유저그룹] [path-to-db.sqlite3]
sudo chown [deploy를 수행하는 유저] [path-to-db.sqlite3]
sudo chown [deploy를 수행하는 유저] [path-to-parent-directory-of-db.sqlite3]
sudo chgrp [deploy를 수행하는 유저그룹] [path-to-parent-directory-of-db.sqlite3]
```


<b>Reference</b>

- [https://nachwon.github.io/django-deploy-1-aws/](https://nachwon.github.io/django-deploy-1-aws/)
- [https://nachwon.github.io/django-deploy-2-wsgi/](https://nachwon.github.io/django-deploy-2-wsgi/)
- [https://nachwon.github.io/django-deploy-3-nginx/](https://nachwon.github.io/django-deploy-3-nginx/)