import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import time

device = torch.device('cuda')


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, down_sample):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.g = nn.ReLU(True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.down_sample = down_sample

    '''
    卷积 -> 标准化 -> 激活(ReLU) -> 卷积 -> 标准化 -> 残差 -> 激活(ReLU)
    '''

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.g(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down_sample:
            residual = self.down_sample(x)
        out += residual
        out = self.g(out)
        return out


class ResNet(nn.Module):
    """
    输入 -> 卷积 -> 标准化 -> 激活 -> 残差层1 -> 残差层2 -> 残差层3 -> 平均池化 -> 全连接层 -> 输出
    """

    def __init__(self, g):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.g = g
        self.layer1 = self.make_layer(16, 1)
        self.layer2 = self.make_layer(32, 2)
        self.layer3 = self.make_layer(64, 2)
        self.avg_pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(64, 10)

    def make_layer(self, out_channels, stride):
        down_sample = None
        if (stride != 1) or (self.in_channels != out_channels):
            down_sample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        layers = [ResidualBlock(self.in_channels, out_channels, stride, down_sample)]
        self.in_channels = out_channels
        layers.append(ResidualBlock(out_channels, out_channels, stride=1, down_sample=None))
        return nn.Sequential(*layers)

    def forward(self, x):
        # [batch_size, 1, 28, 28]
        out = self.conv(x)
        # [batch_size, 16, 28, 28]
        out = self.bn(out)
        # [batch_size, 16, 28, 28]
        out = self.g(out)
        # [batch_size, 16, 28, 28]
        out = self.layer1(out)
        # [batch_size, 16, 28, 28]
        out = self.layer2(out)
        # [batch_size, 32, 14, 14]
        out = self.layer3(out)
        # [batch_size, 64, 7, 7]
        out = self.avg_pool(out)
        # [batch_size, 64, 1, 1]
        out = out.view(out.size(0), -1)
        # [batch_size, 64]
        out = self.fc(out)
        # [batch_size, 10]
        return out


def train(batch_size, learning_rate, num_epoch, g):
    start = time.time()
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data2/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data2/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=True)

    model = ResNet(g).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_corrects = []
    test_losses = []
    test_corrects = []

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        model.train()
        train_loss = 0
        train_correct = 0
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += preds.eq(targets).sum().item()
            optimizer.step()
        loss = train_loss / len(train_loader.dataset)
        correct = train_correct / len(train_loader.dataset)
        print("[Train] Loss: {:.6f}, Accuracy: {:.4f}".format(loss, correct))
        train_losses.append(loss)
        train_corrects.append(correct)

        model.eval()
        test_loss = 0
        test_correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        test_total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, pred = torch.max(outputs.data, 1)
                for i in range(len(targets)):
                    target = targets[i]
                    test_correct[target] += pred[i].eq(target).item()
                    test_total[target] += 1
            loss = test_loss / len(test_loader.dataset)
            print(test_correct)
            print(test_total)
            correct = sum(test_correct) / len(test_loader.dataset)
            print("[Test] Loss: {:.6f}, Accuracy: {:.4f}".format(loss, correct))
            test_losses.append(loss)
            test_corrects.append(correct)
            print('Accuracy for each num:')
            for i in range(10):
                print('{:.4f}'.format(test_correct[i] / test_total[i]), end='  ')
            print()
    end = time.time()
    h, m = divmod(end - start, 3600)
    m, s = divmod(m, 60)
    print('Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s))
    return train_losses, train_corrects, test_losses, test_corrects


# test_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST('./data2/', train=False, download=True,
#                                transform=torchvision.transforms.Compose([
#                                    torchvision.transforms.ToTensor(),
#                                    torchvision.transforms.Normalize(
#                                        (0.1307,), (0.3081,))
#                                ])),
#     batch_size=10, shuffle=True)
# print(len(test_loader.dataset))
num_epochs = 2
# print("batch_size=100, learning_rate=0.001, num_epochs=20, activation=relu")
a0, b0, c0, d0 = train(100, 0.001, num_epochs, F.relu)
# # different activation function
# print("batch_size=100, learning_rate=0.001, num_epochs=20, activation=tanh")
# a11, b11, c11, d11 = train(100, 0.001, num_epochs, F.tanh)
# print("batch_size=100, learning_rate=0.001, num_epochs=20, activation=sigmoid")
# a12, b12, c12, d12 = train(100, 0.001, num_epochs, F.sigmoid)
# # different batch size
# print("batch_size=10, learning_rate=0.001, num_epochs=20, activation=relu")
# a21, b21, c21, d21 = train(10, 0.001, num_epochs, F.relu)
# print("batch_size=1000, learning_rate=0.001, num_epochs=20, activation=relu")
# a22, b22, c22, d22 = train(1000, 0.001, num_epochs, F.relu)
# # different learning rate
# print("batch_size=100, learning_rate=0.1, num_epochs=20, activation=relu")
# a31, b31, c31, d31 = train(100, 0.1, num_epochs, F.relu)
# print("batch_size=100, learning_rate=0.00001, num_epochs=20, activation=relu")
# a32, b32, c32, d32 = train(100, 0.00001, num_epochs, F.relu)
#
# # save data
# with open('./figs2/data.txt', 'w') as f:
#     f.write('a0:' + str(a0) + '\n')
#     f.write('b0:' + str(b0) + '\n')
#     f.write('c0:' + str(c0) + '\n')
#     f.write('d0:' + str(d0) + '\n')
#     f.write('a11:' + str(a11) + '\n')
#     f.write('b11:' + str(b11) + '\n')
#     f.write('c11:' + str(c11) + '\n')
#     f.write('d11:' + str(d11) + '\n')
#     f.write('a12:' + str(a12) + '\n')
#     f.write('b12:' + str(b12) + '\n')
#     f.write('c12:' + str(c12) + '\n')
#     f.write('d12:' + str(d12) + '\n')
#     f.write('a21:' + str(a21) + '\n')
#     f.write('b21:' + str(b21) + '\n')
#     f.write('c21:' + str(c21) + '\n')
#     f.write('d21:' + str(d21) + '\n')
#     f.write('a22:' + str(a22) + '\n')
#     f.write('b22:' + str(b22) + '\n')
#     f.write('c22:' + str(c22) + '\n')
#     f.write('d22:' + str(d22) + '\n')
#     f.write('a31:' + str(a31) + '\n')
#     f.write('b31:' + str(b31) + '\n')
#     f.write('c31:' + str(c31) + '\n')
#     f.write('d31:' + str(d31) + '\n')
#     f.write('a32:' + str(a32) + '\n')
#     f.write('b32:' + str(b32) + '\n')
#     f.write('c32:' + str(c32) + '\n')
#     f.write('d32:' + str(d32) + '\n')
#     f.close()

# # read data
# with open('./figs2/data.txt', 'r') as f:
#     a0 = eval(f.readline()[3:])
#     b0 = eval(f.readline()[3:])
#     c0 = eval(f.readline()[3:])
#     d0 = eval(f.readline()[3:])
#     a11 = eval(f.readline()[4:])
#     b11 = eval(f.readline()[4:])
#     c11 = eval(f.readline()[4:])
#     d11 = eval(f.readline()[4:])
#     a12 = eval(f.readline()[4:])
#     b12 = eval(f.readline()[4:])
#     c12 = eval(f.readline()[4:])
#     d12 = eval(f.readline()[4:])
#     a21 = eval(f.readline()[4:])
#     b21 = eval(f.readline()[4:])
#     c21 = eval(f.readline()[4:])
#     d21 = eval(f.readline()[4:])
#     a22 = eval(f.readline()[4:])
#     b22 = eval(f.readline()[4:])
#     c22 = eval(f.readline()[4:])
#     d22 = eval(f.readline()[4:])
#     a31 = eval(f.readline()[4:])
#     b31 = eval(f.readline()[4:])
#     c31 = eval(f.readline()[4:])
#     d31 = eval(f.readline()[4:])
#     a32 = eval(f.readline()[4:])
#     b32 = eval(f.readline()[4:])
#     c32 = eval(f.readline()[4:])
#     d32 = eval(f.readline()[4:])
#     f.close()
#
# plt.figure(figsize=(10, 15))
# plt.subplot(2, 2, 1)
# plt.ylim(0, 0.02)
# plt.plot(a0, label='relu')
# plt.plot(a11, label='tanh')
# plt.plot(a12, label='sigmoid')
# plt.title('train_loss')
# plt.legend()
# plt.subplot(2, 2, 2)
# plt.ylim(0, 0.02)
# plt.plot(c0, label='relu')
# plt.plot(c11, label='tanh')
# plt.plot(c12, label='sigmoid')
# plt.title('test_loss')
# plt.legend()
# plt.subplot(2, 2, 3)
# plt.ylim(0.7, 1)
# plt.plot(b0, label='relu')
# plt.plot(b11, label='tanh')
# plt.plot(b12, label='sigmoid')
# plt.title('train_accuracy')
# plt.legend()
# plt.subplot(2, 2, 4)
# plt.ylim(0.7, 1)
# plt.plot(d0, label='relu')
# plt.plot(d11, label='tanh')
# plt.plot(d12, label='sigmoid')
# plt.title('test_accuracy')
# plt.legend()
# plt.savefig('./figs2/activation.png')
#
# plt.figure(figsize=(10, 15))
# plt.subplot(2, 2, 1)
# plt.ylim(0, 0.02)
# plt.plot(a21, label='10')
# plt.plot(a0, label='100')
# plt.plot(a22, label='1000')
# plt.title('train_loss')
# plt.legend()
# plt.subplot(2, 2, 2)
# plt.ylim(0, 0.02)
# plt.plot(c21, label='10')
# plt.plot(c0, label='100')
# plt.plot(c22, label='1000')
# plt.title('test_loss')
# plt.legend()
# plt.subplot(2, 2, 3)
# plt.ylim(0.7, 1)
# plt.plot(b21, label='10')
# plt.plot(b0, label='100')
# plt.plot(b22, label='1000')
# plt.title('train_accuracy')
# plt.legend()
# plt.subplot(2, 2, 4)
# plt.ylim(0.7, 1)
# plt.plot(d21, label='10')
# plt.plot(d0, label='100')
# plt.plot(d22, label='1000')
# plt.title('test_accuracy')
# plt.legend()
# plt.savefig('./figs2/batch_size.png')
#
# plt.figure(figsize=(10, 15))
# plt.subplot(2, 2, 1)
# plt.ylim(0, 0.02)
# plt.plot(a31, label='0.1')
# plt.plot(a0, label='0.001')
# plt.plot(a32, label='0.00001')
# plt.title('train_loss')
# plt.legend()
# plt.subplot(2, 2, 2)
# plt.ylim(0, 0.02)
# plt.plot(c31, label='0.1')
# plt.plot(c0, label='0.001')
# plt.plot(c32, label='0.00001')
# plt.title('test_loss')
# plt.legend()
# plt.subplot(2, 2, 3)
# plt.ylim(0.7, 1)
# plt.plot(b31, label='0.1')
# plt.plot(b0, label='0.001')
# plt.plot(b32, label='0.00001')
# plt.title('train_accuracy')
# plt.legend()
# plt.subplot(2, 2, 4)
# plt.ylim(0.7, 1)
# plt.plot(d31, label='0.1')
# plt.plot(d0, label='0.001')
# plt.plot(d32, label='0.00001')
# plt.title('test_accuracy')
# plt.legend()
# plt.savefig('./figs2/learning_rate.png')
