# NIPA 낙상 감지 알고리즘

## 1. Requirement

### 1-1. conda install

#### for Window

    $ winget install miniconda3
    $ conda -V  //설치 확인

#### for macOS

    $ brew install miniconda?
    $ conda init zsh
    $ conda -V  //설치 확인

#### for Linux(This is anaconda3 installing code but miniconda is lighter than anaconda)

    $ sudo apt-get update
    $ sudo apt-get install curl -y

    $ curl --output anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
    $ sha256sum anaconda.sh
    $ bash anaconda.sh
    $ sudo vi ~/.bashrc
    	export PATH=~/anaconda3/bin:~/anaconda3/condabin:$PATH	// 환경 변수 경로 설정
    $ source ~/.bashrc

    $ conda -V      //설치 확인

#### basic manual

    $ conda env list                  // 가상환경 목록 조회
    $ conda activate [envName]        // 가상환경 활성화
    $ conda deactivate [envName]      // 가상환경 비활성화
    $ conda remove -n[envName] --all  // 가상환경 삭제
    $ conda create -clone [envName] -n[newEnvName]  // 가상환경 복제
    $ conda config --set auto_auctivate_base false  // 가상환경 자동활성화 해제
    $ conda install [packageName]     // 패키지 설치
    $ conda list                      // 패키지 조회

### 1-2. conda env setting

    conda create -n [envName] python=3.10  // 가상환경 생성

### 1-3. nvidia driver install

#### for Window

[NVIDIA Homepage Download](https://developer.nvidia.com/cuda-toolkit-archive)

#### for Linux

    $ sudo apt-get remove --purge nvidia-*	// nvidia 제거
    $ sudo apt-get autoremove		        // 제거
    $ sudo apt-get update	                // apt-get 최신 버전 업그레이드

    $ ubuntu-drivers devices	                // 설치가능한 드라이버 확인
    $ sudo apt-get install nvidia-driver-515	// 드라이서 설치
    $ sudo apt-get install dkms nvidia-modprobe	// NVIDIA kernel module의 load를 도와주는 modprobe 패키지를 설치
    $ sudo apt-get update	// 업데이트
    $ sudo apt-get upgrade	// 업그레이드
    $ sudo sync     // 메모리에 있는 디스크로 동기화 (데이터 손실 최소화 작업)
    $ sudo reboot   // 재부팅

    $ nvidia-smi	        //nvidia 드라이버 상태, 디바이스 상태 등을 확인한다.

### 1-3. CUDA 11.7.0 install

#### for Linux

    $ wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run	// wget으로 cuda 인스톨러를 다운로드
    $ sudo sh cuda_11.7.1_515.65.01_linux.run	// sh로 실행
    //  “Continu” 버튼 클릭
    // 라이선스 동의 “accept”
    // 드라이버 이미 설치 했으므로 CUDA Toolkit, CUDA Demo Suite, CUDA Documentation 설치

    // CUDA Toolkit 관련 설정을 환경 변수에 추가
    $ sudo sh -c "echo 'export PATH=$PATH:/usr/local/cuda-11.7/bin'>> /etc/profile"
    $ sudo sh -c "echo 'export LD_LIBRARY_PATH=	$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64'>> /etc/profile"
    $ sudo sh -c "echo 'export CUDADIR=/usr/local/cuda-11.7'>> /etc/profile"
    $ source /etc/profile

    $ nvcc -V	// 설치 확인

### 1-4. cuDNN 8.2.1 install

#### for Window

[NVIDIA Homepage Download](https://developer.nvidia.com/rdp/cudnn-archive)

#### for Linux

    - cuDNN 다운로드
    $ tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz	// 압축 풀기
    $ cd cudnn-linux-x86_64-8.6.0.163_cuda11-archive
    $ sudo cp include/cudnn* /usr/local/cuda/include	// /usr/local/cuda 디렉토리로 복사
    $ sudo cp lib/libcudnn* /usr/local/cuda/lib64		// /usr/local/cuda 디렉토리로 복사
    $ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

    $ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.6.0 /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_adv_train.so.8
    $ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.6.0  /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8
    $ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.6.0  /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8
    $ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.6.0  /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
    $ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.6.0  /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_ops_train.so.8
    $ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.6.0 /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8
    $ sudo ln -sf /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn.so.8.6.0 /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudnn.so.8

### 1-5. torch insatll

#### for Window(CUDA 11.7)

    conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

#### for macOS

    conda install

### 1-6. detectron2 install

#### for Window

    pip install cython
    pip install “git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
    git clone https://github.com/facebookresearch/detectron2.git    // (경로 설정 요망)
    python -m pip install -e detectron2

#### for Linux

    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

### 1-7. openCV install

    pip install opencv-python

## 2. Labeling(wit COCO Anotator)

[COCO Anotator](https://github.com/jsbroks/coco-annotator)

### 2-1. pre Install

#### for Window

    - coco annotator를 설치하기 이전에 docker와 docker-compose가 설치되어 있어야한다.

#### wsl isntall

    - docker를 설치하기 위해서 wsl과 리눅스가 설치되어 있어야한다.
    $ wsl --install
    $ wsl --set-default-version 2

    // wsl을 설치하면 자동으로 우분투 설치된다. (우분투 설치 시 0x80370114 ERROR가 발생할 수 있다. -> 인터넷에 해결방법 검색)

#### docker install

[Docker Homepage Download](https://www.docker.com/products/docker-desktop/)

###### (dcoker-compose는 도커 홈페이지에서 윈도우 버전으로 다운 받아서 설치한 경우 자동 설치)

#### Checking install

    $ wsl -l -v
    $ docker version
    $ docker ps     // 실행중인 컨테이너 확인 명령
    $ docker-compose -v

    $ sudo docker kill $(sudo docker ps -qa)    # 모든 컨테이너 종료

### 2-2. COCO Anotator install

    $ git clone https://github.com/jsbroks/coco-annotator.git   // (경로 설정 요망)

### 2-3. COCO Annotator act

    $ docker-compose up // coco-annotator dir 이동 후

-   localhost:5000 접속 // 안되는 경우 방화벽 풀어주기

### 2-4. img dataset upload

-   coco annotator/dataset/ 경로 폴더에 넣으면 된다.

#### CVAT

[CVAT](https://github.com/opencv/cvat)

    $ git clone https://github.com/opencv/cvat
    $ cd cvat
    $ export CVAT_HOST=[your-ip-address]

    $ sudo docker-compose up -d    # 도커 컨테이너 실행
    (대안 :docker-compose -f docker-compose.yml -f docker-compose.dev.yml build docker-compose up -d)
    $ sudo docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'    # 슈퍼유저 생성
    $ sudo docker exec -t cvat_server python manage.py health_check    # 동작 확인

#### 사용법

-   접속 : localhost:8080

## 3. Train

### 3-1. PATH

-   dataset = /gpu-home/hsenet_dataset
-   execution = /home/gaion/WTC_Seoul/wtc_seoul/HSENet_share/train_net_gaion.py
-   config = /home/gaion/WTC_Seoul/wtc_seoul/HSENet_share/configs // 이 폴더에 yaml 파일을 수정 및 저장
-   checkpoint = /home/gaion/WTC_Seoul/wtc_seoul/HSENet_share/checkpoints // 이 폴더에 모델 및 checkpoint, log 등이 저장

### 3-2. Train

#### easy to train

    $ cd /home/gaion/WTC_Seoul/wtc_seoul/HSENet_share   // 모델이 있는 경로로 이동
    $  python train_net_gaion.py --config configs/실행할_config_파일_이름.yaml --num-gpus 8(사용할 GPU 개수) –resum

    $ python train_net_gaion.py --config configs/ihp_hsenet_V_39_FPN_3x_gaion.yaml --num-gpus 8 --resume

### 3-3. Tensor Board

-   모델 학습 중인 가상환경 실행

    $ tensorboard --logdir checkpoints/현재*학습중인*모델의*checkpoint*폴더\_이름/ --port 9999(포트번호) --bind_all

    $ tensorboard --logdir checkpoints/hsenet_V_39_FPN_3x_gaion_epoch_30000/ --port 9999 --bind_all

-   Localhost:9999(포트번호)로 접속 후 tensorboard 실행 확인
