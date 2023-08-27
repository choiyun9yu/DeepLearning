from copy import deepcopy
import numpy as np

import torch



class Trainer():

    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()


    # 매 에포크마다 SGD를 수행하기 위해 셔플링 후 미니배치를 만드는 과정
    def _batchify(self, x, y, batch_size, random_split=True):
        if random_split:
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices)
            y = torch.index_select(y, dim=0, index=indices)

        x = x.split(batch_size, dim=0)
        y = y.split(batch_size, dim=0)

        return x, y


    # 한 이터레이션의 학습을 위한 for 반복문 구현
    def _train(self, x, y, config):
        self.model.train() # train() 호출하여 학습 모드로 전환

        x, y = self._batchify(x, y, config.batch_size)
        total_loss = 0
        # 작은 루프 담당 (미니배치의 피드포워드, 역전파, 경사하강법 파라미터)
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze())

            # Initialize the gradients of the model.
            self.optimizer.zero_grad()
            loss_i.backward()

            self.optimizer.step()

            if config.verbose >= 2: # config.verbose에 따라 현재 학습 현황 ㅜㄹ력
                print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

            # Don't forget to detach to prevent memory leak.
            total_loss += float(loss_i)

        return total_loss / len(x)


    # 대부분 _train()과 비슷하지만 가장 바깥쪽에 torch.no_grad()가 호출되어 있음!
    def _validate(self, x, y, config):
        # Turn evaluation mode on.
        self.model.eval()

        # Turn on the no_grad mode to make more efficintly.
        with torch.no_grad():
            x, y = self._batchify(x, y, config.batch_size, random_split=False)  # 검증에서는 random_split=False
            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                if config.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

                total_loss += float(loss_i)

            return total_loss / len(x)


    # 학습과 검증을 아우르는 큰 loop와 학습과 검증 내 작은 loop 중 큰 loop 정의
    # config는 가장 바깥 train.py에서 사용자의 실행 시 파라미터 입력에 따른 설정값이 포함된 객체
    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)

            # You must use deep copy to take a snapshot of current best weights.
            if valid_loss <= lowest_loss:   
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())  # sate_dict() : 모델의 가중치 파라미터값을 JSON 형태로 변환 리턴

            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                epoch_index + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))

        # Restore to best model.
        # load_state_dict() : 학습이 종료되면 best모델을 self.model에 다시 로딩
        self.model.load_state_dict(best_model)
