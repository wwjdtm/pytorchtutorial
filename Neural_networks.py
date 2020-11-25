#모델을 정의하고 미분하는데 autograd를 사용.
# nn.Module은 계층(layer)과 output을 반환하는 forward(input) 메서드포함
# 신경망의 일반적인 학습과정
# 1. 학습가능한 매개변수(가중치)를 갖는 신경망을 정의

# CNN(Convolutional Neural Network)
# image 전체를 보는 것이 아니라 부분을 보는 것이 핵심. 이 ‘부분’에 해당하는 것을 filter


import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #컨볼루션 커널 정의
        #입력 이미지 채널 1개, 출력 채널 6개, 3x3의 정사각 컨볼루션 행렬
        self.conv1 = nn.Conv2d(1,6,3)
        #입력 이미지 채널 6개, 출력 채널 16개, 3x3의 정사각 컨볼루션 행렬
        self.conv2 = nn.Conv2d(6,16,3)
        #아핀(affine) 연산 : y = wx+b
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        # (2, 2) 크기 윈도우에 대해 맥스 풀링(max pooling)
        # 최댓값을 뽑아내는 max pooling - overfitting을 방지하기 위함
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        # 크기가 제곱수라면 하나의 숫자만을 특정
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self,x):
        # 배치 차원을 제외한 모든 차원
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
#모델의 학습 가능한 매개변수들은 net.parameters() 에 의해 반환
params = list(net.parameters())
print(len(params))
print(params[0].size())





