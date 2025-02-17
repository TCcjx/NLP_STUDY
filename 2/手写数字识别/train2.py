"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: train2.py
 @DateTime: 2025-02-17 15:59
 @SoftWare: PyCharm
"""
import torch.nn as nn
import torch
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os

torch.manual_seed(888)

# 超参数
epochs = 1
batch_size = 50
lr = 0.001
download_mnist = True
DOWNLOAD_MNIST = False

# 训练集
train_data = torchvision.datasets.MNIST(
    root = './data/',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST
)

# 测试集
test_data = torchvision.datasets.MNIST(
    root = './data/',
    train = False,
    download = DOWNLOAD_MNIST
)

train_loader = Data.DataLoader(
    dataset = train_data,
    batch_size = batch_size,
    shuffle = True
)

# 进行测试
# 为节约时间，测试时只测试前2000个
test_x = torch.unsqueeze(test_data.train_data, dim=1).type(torch.FloatTensor)[:2000] / 255
# torch.unsqueeze(a) 是用来对数据维度进行扩充，这样shape就从(2000,28,28)->(2000,1,28,28)
# 图像的pixel本来是0到255之间，除以255对图像进行归一化使取值范围在(0,1)
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,out_channels=16,
                kernel_size=5,stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.out = nn.Linear(32*7*7,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        output = self.out(x)
        return output

model = CNN()
# print(model)
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

# 训练过程实现
# for epoch in range(epochs):
#     for step,(train_data,train_label) in enumerate(train_loader):
#         output = model(train_data)
#         loss = loss_func(output,train_label)
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()
#
#         if step % 50 == 0:
#             test_output = model(test_x)
#             pred_y = torch.argmax(test_output,dim=1)
#             accuracy = float((pred_y == test_y).sum()) / float(test_y.size(0))
#             print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
# torch.save(model.state_dict(), 'cnn.pkl')#保存模型

# 加载模型
model.load_state_dict(torch.load('cnn.pkl'))
model.eval()
inputs = test_x[:101]
test_output = model(inputs)
pred_y = torch.argmax(test_output,dim=1)
print('prediction number:',pred_y)
print('True number:',test_y[:101])
print('acc:',sum(pred_y==test_y[:101]).item()/101)

