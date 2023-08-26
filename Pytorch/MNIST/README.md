## Vitual Environment

## Requirement

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
- Regularization 
    - Underfitting : Model Capacity 키우기
    - Overfitting : Batch Normalization, Dropout 
- Loss Function  
    - 이진 분류 : binary cross entropy, 
    - 다중 분류 : cross entropy
    - 회귀 : MSE
- Optimaizer : Adam

### 5. 평가
- Train dataset으로 Weight Parameter 결정
- Validation dataset으로 Hyper Parameter 결정
- Test dataset으로 Model 결정

### 6. 배포
- RESTful API를 통한 배포
- 유지보수

