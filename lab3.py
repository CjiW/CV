"""
任务要求：设计一个卷积神经网络，输入为两张MNIST手写体数字图片，如果两张图片为同一个数字（注意，非同一张图片），输出为1，否则为0。

从MNIST数据集的训练集中选取10%作为本实验的训练图片，从MNIST数据集的测试集中选取10%作为本实验的测试图片。
请将该部分图片经过适当处理形成一定数量的用于本次实验的训练集和测试集。

解
这里的训练集生成的步骤如下：
1. 从MNIST数据集的训练集中选取10%作为本实验的训练图片 A
2. 从MNIST数据集的测试集中选取10%作为本实验的训练图片 B
3. 将上述两个集合进行笛卡尔积，得到的集合中的每个元素都是一个二元组<a, b>，target = 1 if a == b else 0
神经网络将输入的两张图片进行拼接，然后进行卷积，最后输出一个二分类的结果。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import random
import time

print("import done")
device = torch.device('cuda')

num_epochs = 200
batch_size_train = 10
batch_size_test = 10

full_train_dataset = torchvision.datasets.MNIST('./data3/', train=True, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.1307,), (0.3081,))
                                                ]))
full_test_dataset = torchvision.datasets.MNIST('./data3/', train=False, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       (0.1307,), (0.3081,))
                                               ]))
print("load done")

print(len(full_train_dataset) // 10)
random_train_indices = random.sample(range(len(full_train_dataset)), len(full_train_dataset) // 10)
random_test_indices = random.sample(range(len(full_test_dataset)), len(full_test_dataset) // 10)

train_dataset0 = torch.utils.data.Subset(full_train_dataset, random_train_indices)
test_dataset0 = torch.utils.data.Subset(full_test_dataset, random_test_indices)


class CartesianProductDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.length = len(dataset)

    """
    随机返回 image1, image2, target
    """

    def __getitem__(self, index):
        i = random.randint(0, self.length - 1)
        j = random.randint(0, self.length - 1)
        img1, label1 = self.dataset[i]
        img2, label2 = self.dataset[j]
        return img1, img2, int(label1 == label2)

    def __len__(self):
        return self.length


train_dataset = CartesianProductDataset(train_dataset0)
test_dataset = CartesianProductDataset(test_dataset0)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size_train,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size_test,
                                          shuffle=True)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.g = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.pool(self.g(self.conv1(x)))
        x = self.pool(self.g(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.g(self.fc1(x))
        x = self.g(self.fc2(x))
        x = self.fc3(x)
        return x


model = ConvNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

train_losses = []
train_corrects = []

test_losses = []
test_corrects = []
print("start training")
start = time.time()


class DrawFigure:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 12))
        plt.ion()
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.ax1.set_title('Loss')
        self.ax1.set_xlim(-1, 201)
        self.ax1.set_ylim(0.0, 0.5)
        self.ax2.set_title('Correct')
        self.ax2.set_xlim(-1, 201)
        self.ax2.set_ylim(0.85, 1.0)
        self.line11, = self.ax1.plot(train_losses)
        self.line12, = self.ax1.plot(test_losses)
        self.line21, = self.ax2.plot(train_corrects)
        self.line22, = self.ax2.plot(test_corrects)
        self.ax1.legend(['Train', 'Test'])
        self.ax2.legend(['Train', 'Test'])

    def draw(self):
        self.line11.set_data(range(len(train_losses)), train_losses)
        self.line12.set_data(range(len(test_losses)), test_losses)
        self.line21.set_data(range(len(train_corrects)), train_corrects)
        self.line22.set_data(range(len(test_corrects)), test_corrects)
        plt.draw()
        plt.pause(0.01)


df = DrawFigure()


def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))


for epoch in range(num_epochs):
    # train
    model.train()
    train_loss = []
    train_correct = 0
    print("Epoch: {}/{}".format(epoch+1, num_epochs))
    for i, (inputs1, inputs2, targets) in enumerate(train_loader):
        inputs1, inputs2, targets = inputs1.to(device), inputs2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs1, inputs2)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        _, preds = torch.max(outputs.data, 1)
        train_correct += preds.eq(targets).sum().item()
    loss = sum(train_loss) / len(train_loss)
    correct = train_correct / len(train_dataset)
    print('Train Loss: {:.6f} Train Accuracy: {}/{} ({:.6f})'.format(loss, train_correct, len(train_dataset), correct))
    train_losses.append(loss)
    train_corrects.append(correct)

    # test
    model.eval()
    test_loss = []
    test_correct = 0
    with torch.no_grad():
        for i, (inputs1, inputs2, targets) in enumerate(test_loader):
            inputs1, inputs2, targets = inputs1.to(device), inputs2.to(device), targets.to(device)
            outputs = model(inputs1, inputs2)
            loss = F.cross_entropy(outputs, targets)
            test_loss.append(loss.item())
            _, preds = torch.max(outputs.data, 1)
            test_correct += preds.eq(targets).sum().item()
    loss = sum(test_loss) / len(test_loss)
    correct = test_correct / len(test_dataset)
    print('Test Loss: {:.6f} Test Accuracy: {}/{} ({:.6f})'.format(loss, test_correct, len(test_dataset), correct))
    test_losses.append(loss)
    test_corrects.append(correct)
    print("Remaining time: ", format_time((time.time() - start) / (epoch + 1) * (num_epochs - epoch - 1)), "\n")

print("Total time:", format_time(time.time() - start))
df.draw()
plt.savefig("./figs3/lab3.png")
