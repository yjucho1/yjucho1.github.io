---
title: "django - AWS 배포하기"
categories: 
  - django
comments: true
mathjax : true
published: true

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

[https://nachwon.github.io/django-deploy-1-aws/](https://nachwon.github.io/django-deploy-1-aws/)

