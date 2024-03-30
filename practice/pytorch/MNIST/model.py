import torch
import torch.nn as nn

# 계층이 모여서 모듈이되고, 모듈이 모여서 모델이 된다!

# 모델은 입출력 크기만 바뀐 모듈(선형계층 + 비선형 활성화 함수 + 정규화)이 반복된다.
# 모듈을 서브모듈 클래스로 정의하고 선형계층, 비선형 활성화함수, 정규화를 객체로 넣어주는 건 어떨까?

# 서브 모듈 정의
class Block(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 use_batch_norm=True,
                 dropout_p=.4):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        
        super().__init__()
        
        # 입력받은 인자에 따라 정규화과정 처리하는 함수
        def get_regularizer(use_batch_norm, size):
            # use_batch_norm 변수로 배치정규화와 드롭아웃 중 양자 택일
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)
        
        # 서브모듈에서 사용할 선형계층, 활성화 함수, 정규화 정의
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            get_regularizer(use_batch_norm, output_size),
        )
        
    # 서브 모듈에서 실행되어야 하는 연산 정의
    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.block(x)
        # |y| = (batch_size, output_size)
        
        return y

    
class ImageClassifier(nn.Module):
    # 사용될 모듈과 활성화 함수 등을 정의
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=[500, 400, 300, 200, 100],
                 use_batch_norm=True,
                 dropout_p=.3):
        
        super().__init__()

        # assert는 뒤의 조건이 True가 아니면 AssertError를 발생시킨다.
        assert len(hidden_sizes) > 0, "You need to specify hidden layers"

        last_hidden_size = input_size
        blocks = []
        # 반복문으로 서브 모듈 정의
        for hidden_size in hidden_sizes:
            blocks += [Block(
                last_hidden_size,
                hidden_size,
                use_batch_norm,
                dropout_p
            )]
            last_hidden_size = hidden_size
        
        self.layers = nn.Sequential(
            # 반복문으로 생산된 서브 모듈 블럭
            *blocks,
            # 출력 부분
            nn.Linear(last_hidden_size, output_size),
            nn.LogSoftmax(dim=-1),
        )

    # 모델에서 실행되어야할 연산 정의    
    def forward(self, x):
        # |x| = (batch_size, input_size)        
        y = self.layers(x)
        # |y| = (batch_size, output_size)
        
        return y
