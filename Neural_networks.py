#모델을 정의하고 미분하는데 autograd를 사용.
# nn.Module은 계층(layer)과 output을 반환하는 forward(input) 메서드포함

# 신경망의 일반적인 학습과정
#학습 가능한 매개변수(또는 가중치(weight))를 갖는 신경망을 정의합니다.
#데이터셋(dataset) 입력을 반복합니다.
#입력을 신경망에서 전파(process)합니다.
#손실(loss; 출력이 정답으로부터 얼마나 떨어져있는지)을 계산합니다.
#변화도(gradient)를 신경망의 매개변수들에 역으로 전파합니다.
#신경망의 가중치를 갱신합니다. 일반적으로 다음과 같은 간단한 규칙을 사용합니다: 새로운 가중치(weight) = 가중치(weight) - 학습률(learning rate) * 변화도(gradient)

# CNN(Convolutional Neural Network)
# image 전체를 보는 것이 아니라 부분을 보는 것이 핵심. 이 ‘부분’에 해당하는 것을 filter


import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module): #nn.module 상속받아 클래스Net 정의
  def __init__(self):
    super(Net, self).__init__()
    # 1 input image channel, 6 output channels, 5x5 square convolution kernel
    self.conv1 = nn.Conv2d(1, 6, 5)
    #필터를 사용하여 하나의 채널이 들어가면 6개의 특징맵을 출력함
    self.conv2 = nn.Conv2d(6, 16, 5)
    # an affine operation: y = Wx + b
    #선형회귀모델 구현
    self.fc1 = nn.Linear(16 * 5 * 5, 120) # 이미지 차원은 5*5
    #위에서 특징맵이 16채널이 나옴 , 특징맵의 면적 5*5/
    #120개의 차원을 가지는 특징벡터를 뽑아냄
    self.fc2 = nn.Linear(120, 84)
    #84개 차원을 가지는 특징벡터를 뽑아냄
    self.fc3 = nn.Linear(84, 10)
    #최종으로 10개의 score를 출력

  def forward(self, x):
    #2.2 크기 윈도우에 대해 맥스풀링
    #최대값을 뽑아주는 max pooling - overfitting 방지하기 위함
    # Max pooling over a (2, 2) window
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    #x를 self.conv1을 통과시켜줌->relu통과->maxpool(2*2의 윈도우크기를가지고)->절반으로 줄어드는 결과
    # If the size is a square you can only specify a single number
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    #self.conv2통과- - > (2,2)를 2로 특정가능
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc1(x))
    #flat한 값(fully connected layer 통과가능)을 fc1->relu통과
    x = F.relu(self.fc2(x))
    #flat한 값(fully connected layer 통과가능)을 fc2->relu통과
    x = self.fc3(x)
    return x

  def num_flat_features(self, x):
    size = x.size()[1:] # all dimensions except the batch dimension
    #배치차원 제외한 모든 차원
    num_features = 1
    for s in size:
      num_features *= s
    return num_features

net = Net()
#net이라는 신경망 정의함
print(net)
# net.parameters()를 사용하여 정의된 신경망의 학습가능한 매개변수들을 확인할 수 있음
params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1's .weight
#conv필터의 면적이 5*5이고,입력이미지가 1채널, 출력채널수 6

# 다음의 임의의 32*32 입력을 가정함
# 참고로 크기가 다른 입력을 받을 때는 입력의 크기를 재조정하거나 신경망 수정함
input = torch.randn(1, 1, 32, 32)
#사이즈만 맞는 이미지형태의 가짜값을 인풋형태로 넣어줌, 32*32 1채널 넣어줌, 배치=1
out = net(input)
print(out)

# 오류역전파를 통해 그레이디언트를 구하기 전에 모든 가중치의 그레이디언트 버퍼들을 초기화
# cnn네트워크 학습하기 전에는 zero_grad해줘야함 . 그레디언트가 누적이되기때문
net.zero_grad()
# 랜덤값으로 역전파
out.backward(torch.randn(1, 10))

#LOSS FUNCTION
#손실함수는 output,target 을 한쌍으로 입력받아 output이 target으로부터
#얼마나 멀리 떨어져있는지(틀렸는지) 추정
# 손실 함수 정의 및 임의의 값들에 대해서 오차 결과 확인
# nn 패키지는 많이 사용되는 손실함수들을 제공하며, 해당 예제는 단순한 MSE 를 사용
output = net(input)
target = torch.randn(10) # a dummy target, for example
print(target) #랜덤값 들어있음
target = target.view(1, -1) # make it the same shape as output
#shape을 아웃풋값과 같게 만들어줌
print(target)
criterion = nn.MSELoss()
#손실함수계산 함수 정의 (평균제곱오차를 계산)
loss = criterion(output, target)
print(loss)

# 앞에 코드에서 언급한 것과 같이 오류 역전파하기 전, 그레이디언트를 초기화해야 함
# backward() 수행 후 어떤 변화가 있는지 확인하고, 초기화의 필요성을 확인함
net.zero_grad() # zeroes the gradient buffers of all parameters
#기존의 변화도를 없애는 작업
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
#zero_grad했기때문에 0으로 나옴
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# 스토캐스틱 경사하강법 ( (미래)가중치 = (현재)가중치 – 학습률 * 그레이디언트 )을 이용하여
#가중치 갱신하는 코드는 다음과 같음
learning_rate = 0.01
for f in net.parameters():
  f.data.sub_(f.grad.data * learning_rate)


# 하지만 위 구현 코드보다 실제, torch.optim 에서 구현되는 SDG, Adam, RMSProp 등을 사용함
# 오류 역전파에서 최적화하는 방법을 보인 예제 코드
import torch.optim as optim
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
#optimizer SGD 사용, 파라미터에 대해 옵티마이저 사용(학습할 파라미터 넣어줌)
#lr 학습률은 0.01

# in your training loop:
optimizer.zero_grad() # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # Does the update
#가중치 갱신됨



