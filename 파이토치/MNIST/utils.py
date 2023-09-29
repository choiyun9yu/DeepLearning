import torch

# 데이터 로드 함수
def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)   # view(rows, columns) : 텐서 크기 변경, -1은 알아서 맞추라는 옵션

    return x, y


# 데이터 분할
def split_data(x, y, train_ratio=.8):
    train_cnt = int(x.size(0) * train_ratio)    # 학습 데이터셋 수
    valid_cnt = x.size(0) - train_cnt           # 검증 데이터셋 수

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0))  # torch.randperm(n): 0부터 n-1개의 랜덤한 정수 순열을 리턴
    x = torch.index_select(     # 텐서 슬라이싱 : torch.index_select(검색대상, aixs, index) → Tensor
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)      # 학습셋과 검증셋으로 분할
    # 타겟에 대해서도 마찬가지
    y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    return x, y


# model.py에서 hidden_sizes[] 를 통해 쌓을 Block의 크기를 지정했다.
# 사용자가 일일이 블럭 크기 지정하는 것은 번거로울 수 있기에 사용자가 모델의 계층 개수만 정해주면 자동으로 등차수열을 적용하여 hidden_size 구성하는 함수
def get_hidden_sizes(input_size, output_size, n_layers):
    step_size = int((input_size - output_size) / n_layers)

    hidden_sizes = []
    current_size = input_size
    for i in range(n_layers - 1):
        hidden_sizes += [current_size - step_size]
        current_size = hidden_sizes[-1]

    return hidden_sizes
