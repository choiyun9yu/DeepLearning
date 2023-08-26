## Vitual Environment

## Requirement

## Pytorch Library
- torch : 메인 네임스페이스, 텐서 등의 다양한 수학 함수가 포함된 라이브러리
- torch.autograd : 자동 미분 기능을 제공하는 라이브러리
- torch.nn : 신경망 구축을 위한 데이터 구조나 레이어 등의 라이브러리
- torch.multiprocessing : 병럴처리 기능을 제공하는 라이브러리
- torch.optim : 가중치 파라미터 최적화 알고리즘 제공하는 라이브러리
- torch.utils : 데이터 조작 등 유틸리티 기능 제공하는 라이브러리
- torch.onnx : ONNX(Open Neural Network Exchange), 서로 다른 프레임워크 간의 모델을 공유할 때 사용

## Directory Architecure
- model.py : 모델 클래스 정의하는 코드
- trainer.py : 데이터 받아와서 모델 객체를 학습하기 위한 트레이너 정의하는 코드
- dataloader.py : 데이터 파일을 읽어와 전처리 수행하고 신경망의 입력 포맷에 맞는 형태로 변환하는 코드
- train.py : 사용자로부터 하이퍼파라미터를 입력받아 필요한 객체들을 준비하여 학습하는 코드
- predict.py : 사용자로부터 기학습된 모델의 추론을 위한 샘플을 읿력받아 추론 수행하는 코드

## Work Flow 
### 1. 문제 정의
- 주어진 복잡한 문제를 단순화
- feature와 label을 정의 

### 2. 데이터 수집
- 문제 정의에 따라 데이터를 수집
- 필요에 따른 라벨링 

### 3. 데이터 전처리 및 분석
- 수집 데이터를 학습/검증/평가 데이터 셋으로 분할
- 데이터 유형에 따른 전처리
    - 테뷸러 데이터 : Null제거, 스케일링(정규화)
    - 이미지 데이터 : 데이트 증강, 크롭핑
    - 텍스트 데이터 : 정제, 분절, 저빈도 단어 제거

### 4. 알고리즘 적용

- 가설 설정
- 외형 구성
  
    class modelName(torch.nn.Module):
        def __init__():
            super.__init__()
        self.layer = torch.nn.Sequential(
            # 은닉층 정의
        )
        
        def forward():
            y = self.layer(X)
            return y

- Regularization 
    - Underfitting : Model Capacity 키우기
    - Overfitting : Batch Normalization, Dropout 

- Activation Fucntion : 이전 층의 값을 다음 층으로 비선형 변환해서 전달
    - torch.nn.functional.Sigmoid : 입력, 은닉층 사용 (이진분류의 출력층)
    - torch.nn.functional.ReLU : 입력, 은닉층 사용 (생성모델의 출력층)
    - torch.nn.functional.softmax : 주로 출력층에서 사용 (다중 분류, 원핫인코딩 이진분류의 출력층)
    - torch.nn.functional.tanh : 입력, 은닉층 사용 

- Loss Function : 가중치 파라미터 최적화를 위한 손실 비용 계산 함수  
    - torch.nn.BCELoss : 이진 분류를 위해 사용
    - torch.nn.CrossEntropyLoss : 다중 클래스 분류를 위해 사용
    - torch.nn.MSELoss : 회귀 모델에서 사용

- Optimaizer : 가중치 파라미터 최적화를 위한 경사하강법
    - torch.optim.Adam : 잘 모르겠으면 아담 사용
    - Scheduler 
        - optim.lr_scheduler.LambdaLR : 람다(lambda) 함수를 이용해 그 결과를 학습률로 설정
        - optim.lr_scheduler.StepLR : 단계(step)마다 학습률을 감마(gamma) 비율만큼 감소
        - optim.lr_scheduler.MultiStepLR : `StepLR`과 비슷하지만 특정 단계가 아니라 지정된 에포크에만 감마 비율로 감소
        - optim.lr_scheduler.ExponentialLR : 에포크마다 이전 학습률에 감마만큼 곱함
        - optim.lr_scheduler.CosineAnnealingLR : 학습률을 코사인(cosine) 함수의 형태처럼 변화시켜 학습률일 커지기도 하고 작아지기도 함
        - optim.lr_scheduler.ReduceLROnPlateau : 학습이 잘되는지 아닌지에 따라 동적으로 학습률 변화

### 5. 평가
- Train dataset으로 Weight Parameter 결정
- Validation dataset으로 Hyper Parameter 결정
- Test dataset으로 Model 결정

- Monitering
    - torchmetrics.BinaryAccuracy
    - torchmetrics.MulticlassAccuracy
    - torchmetrics.MeanSquaredError


### 6. 배포
- RESTful API를 통한 배포
- 유지보수

