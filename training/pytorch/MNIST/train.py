import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer

from utils import load_mnist
from utils import split_data
from utils import get_hidden_sizes


# 사용자 입력 파라미터 config에 담는다.
def define_argparser():
    p = argparse.ArgumentParser()                           # argparse 라이브러리를 통해 다양한 입력 파라미터들을 손쉽게 정의하고 처리

    p.add_argument('--model_fn', required=True)             # 모델 가중치가 저장될 파일 경로
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)    # 학습이 수행될 그래픽 카드 인덱스 번호

    p.add_argument('--train_ratio', type=float, default=.8) # 학습 데이터 내에서 검증 데이터가 차지할 비율

    p.add_argument('--batch_size', type=int, default=256)   # 미니배치 크기
    p.add_argument('--n_epochs', type=int, default=20)      # 에포크 개수 

    p.add_argument('--n_layers', type=int, default=5)       # 모델의 계층 개수
    p.add_argument('--use_dropout', action='store_true')    # 드롭아웃 사용 여부
    p.add_argument('--dropout_p', type=float, default=.3)   # 드롭아웃 사용 시 드롭 확률

    p.add_argument('--verbose', type=int, default=1)        # 확습 시 로그 출력의 정도

    config = p.parse_args()

    return config

# 작성한 코드들을 모아서 학습이 진행되도록 하는 함수
def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    x, y = load_mnist(is_train=True, flatten=True)
    x, y = split_data(x.to(device), y.to(device), train_ratio=config.train_ratio)

    print("Train:", x[0].shape, y[0].shape)
    print("Valid:", x[1].shape, y[1].shape)

    input_size = int(x[0].shape[-1])
    output_size = int(max(y[0])) + 1

    model = ImageClassifier(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=get_hidden_sizes(input_size,
                                      output_size,
                                      config.n_layers),
        use_batch_norm=not config.use_dropout,
        dropout_p=config.dropout_p,
    ).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(model, optimizer, crit)

    trainer.train(
        train_data=(x[0], y[0]),
        valid_data=(x[1], y[1]),
        config=config
    )

    # Save best model weights.
    torch.save({
        'model': trainer.model.state_dict(),
        'opt': optimizer.state_dict(),
        'config': config,
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
