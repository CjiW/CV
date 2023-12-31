# 任务要求：设计一个前馈神经网络，对一组数据实现分类任务。
#
# 下载“dataset.csv”数据集，其中包含四类二维高斯数据和它们的标签。
# 设计至少含有一层隐藏层的前馈神经网络来预测二维高斯样本(data1,data2)所属的分类label。
# 这个数据集需要先进行随机排序，然后选取90%用于训练，剩下的10%用于测试。

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt


class FeedForward(torch.nn.Module):
    def __init__(self, g, layers):
        super(FeedForward, self).__init__()
        self.fcs = torch.nn.ParameterList()
        for i in range(len(layers) - 1):
            self.fcs.append(torch.nn.Linear(layers[i], layers[i + 1]))
        self.g = g

    def forward(self, x):
        for i in range(len(self.fcs) - 1):
            x = self.g(self.fcs[i](x))
        x = self.fcs[-1](x)
        return x


def train(num_epochs, batch_size, learning_rate, layer_sizes, g):
    device = torch.device("cuda")
    # read data
    inputs = pd.read_csv("dataset.csv").sample(frac=1).reset_index(drop=True)

    data_size = inputs.shape[0]
    train_size = int(data_size * 0.9)
    test_size = data_size - train_size

    train_data = torch.utils.data.TensorDataset(torch.tensor(inputs.iloc[:train_size, :-1].values, dtype=torch.float32),
                                                torch.tensor(inputs.iloc[:train_size, -1].values, dtype=torch.long))
    test_data = torch.utils.data.TensorDataset(torch.tensor(inputs.iloc[train_size:, :-1].values, dtype=torch.float32),
                                               torch.tensor(inputs.iloc[train_size:, -1].values, dtype=torch.long))

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=test_size,
                                              shuffle=True)

    model = FeedForward(g, layer_sizes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_corrects = []

    test_losses = []
    test_corrects = []

    for epoch in range(num_epochs):
        # train
        model.train()
        train_loss = []
        train_correct = 0
        for i, (inputs, targets) in enumerate(train_loader):
            targets = (targets - 1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = torch.nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            optimizer.step()
            _, pred = torch.max(outputs, 1)
            train_correct += pred.eq(targets).sum().item()
            train_loss.append(loss.item())
        loss = np.mean(train_loss)
        correct = train_correct / train_size
        print('Train: {} Loss: {:.6f} Accuracy: {}/{} ({:.6f})'.format(epoch, loss, train_correct, train_size, correct))
        train_losses.append(loss)
        train_corrects.append(correct)
        # test
        model.eval()
        test_loss = []
        test_correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                targets = (targets - 1).to(device)
                outputs = model(inputs.to(device))
                loss = torch.nn.CrossEntropyLoss()(outputs, targets)
                _, pred = torch.max(outputs, 1)
                test_correct += pred.eq(targets).sum().item()
                test_loss.append(loss.item())
        loss = np.mean(test_loss)
        correct = test_correct / test_size
        print('Test: {} Loss: {:.6f} Accuracy: {}/{} ({:.6f})\n'.format(epoch, loss, test_correct, test_size, correct))
        test_losses.append(loss)
        test_corrects.append(correct)

    return train_losses, train_corrects, test_losses, test_corrects


plt.figure(figsize=(10, 15))
l11, a11, l12, a12 = train(50, 64, 0.001, [2, 20, 4], F.relu)
l21, a21, l22, a22 = train(50, 64, 0.001, [2, 200, 4], F.relu)
l31, a31, l32, a32 = train(50, 64, 0.001, [2, 2000, 4], F.relu)
plt.subplot(221)
plt.title('train_loss_relu_2-x-4')
plt.plot(l11, label='[20]')
plt.plot(l21, label='[200]')
plt.plot(l31, label='[2000]')
plt.legend()
plt.subplot(222)
plt.title('train_accuracy_relu_2-x-4')
plt.plot(a11, label='[20]')
plt.plot(a21, label='[200]')
plt.plot(a31, label='[2000]')
plt.subplot(223)
plt.title('test_loss_relu_2-x-4')
plt.plot(l12, label='[20]')
plt.plot(l22, label='[200]')
plt.plot(l32, label='[2000]')
plt.legend()
plt.subplot(224)
plt.title('test_accuracy_relu_2-x-4')
plt.plot(a12, label='[20]')
plt.plot(a22, label='[200]')
plt.plot(a32, label='[2000]')
plt.legend()
plt.savefig('figs1/diff_size.png')

plt.figure(figsize=(10, 15))
l11, a11, l12, a12 = train(50, 64, 0.001, [2, 20, 4], F.relu)
l21, a21, l22, a22 = train(50, 64, 0.001, [2, 20, 20, 4], F.relu)
l31, a31, l32, a32 = train(50, 64, 0.001, [2, 20, 20, 20, 20, 4], F.relu)
plt.subplot(221)
plt.title('train_loss_relu_2-x-4')
plt.plot(l11, label='[20]')
plt.plot(l21, label='[20-20]')
plt.plot(l31, label='[20-20-20-20]')
plt.legend()
plt.subplot(222)
plt.title('train_accuracy_relu_2-x-4')
plt.plot(a11, label='[20]')
plt.plot(a21, label='[20-20]')
plt.plot(a31, label='[20-20-20-20]')
plt.subplot(223)
plt.title('test_loss_relu_2-x-4')
plt.plot(l12, label='[20]')
plt.plot(l22, label='[20-20]')
plt.plot(l32, label='[20-20-20-20]')
plt.legend()
plt.subplot(224)
plt.title('test_accuracy_relu_2-x-4')
plt.plot(a12, label='[20]')
plt.plot(a22, label='[20-20]')
plt.plot(a32, label='[20-20-20-20]')
plt.legend()
plt.savefig('figs1/diff_depth.png')

plt.figure(figsize=(10, 15))
l11, a11, l12, a12 = train(100, 64, 0.001, [2, 20, 20, 4], F.relu)
l21, a21, l22, a22 = train(100, 64, 0.001, [2, 20, 20, 4], F.tanh)
l31, a31, l32, a32 = train(100, 64, 0.001, [2, 20, 20, 4], F.sigmoid)
plt.subplot(221)
plt.title('train_loss_x_2-20-20-4')
plt.plot(l11, label='relu')
plt.plot(l21, label='tanh')
plt.plot(l31, label='sigmoid')
plt.legend()
plt.subplot(222)
plt.title('train_accuracy_x_2-20-20-4')
plt.plot(a11, label='relu')
plt.plot(a21, label='tanh')
plt.plot(a31, label='sigmoid')
plt.subplot(223)
plt.title('test_loss_x_2-x-4')
plt.plot(l12, label='relu')
plt.plot(l22, label='tanh')
plt.plot(l32, label='sigmoid')
plt.legend()
plt.subplot(224)
plt.title('test_accuracy_x_2-20-20-4')
plt.plot(a12, label='relu')
plt.plot(a22, label='tanh')
plt.plot(a32, label='sigmoid')
plt.legend()
plt.savefig('figs1/diff_af.png')
