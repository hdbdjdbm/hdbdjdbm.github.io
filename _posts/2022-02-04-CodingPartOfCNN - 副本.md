---
layout:     post
title:      神经网络的代码笔记
subtitle:   持续更新
date:       2022-02-04
author:     hdbdjdbm
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - 机器学习
---



### 一些notice

#### Tensors (张量)
Tensors 类似于 NumPy 的 ndarrays, 同时 Tensors 可以使用 GPU 进行计算。

```python
from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)

#output
tensor(1.00000e-04 *
       [[-0.0000,  0.0000,  1.5135],
        [ 0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000]])


#根据现有的  tensor 建立新的  tensor
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # 重载 dtype!
print(x)                                      # 结果size一致

```

```python
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

#output
tensor([[ 0,  0,  0],
        [ 0,  0,  0],
        [ 0,  0,  0],
        [ 0,  0,  0],
        [ 0,  0,  0]])

```

```python

# 获取它的维度信息
print(x.size())
torch.Size([5, 3])

# 加法1
y = torch.rand(5, 3)
print(x + y)

# 加法2
print(torch.add(x, y))

# 加法3
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# 加法4
# adds x to y
y.add_(x)
print(y)


# 注意 任何使张量会发生变化的操作都有一个前缀 '_'。例如：
# ex.copy_(y)
# x.t_()
# x



```

改变形状：如果想改变形状，可以使用  torch.view 
```python

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

#output
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

torch 的 Tensor 和 NumPy 互相转换
```python
a = torch.ones(5)
print(a)
tensor([1., 1., 1., 1., 1.])
b = a.numpy()

a = np.ones(5)
b = torch.from_numpy(a)

```







### PyTorch Autograd自动求导
PyTorch中，神经网络的核心是autograd包。 autograd包为张量上的所有操作提供了自动求导机制。


创建一个张量并设置requires_grad=True用来追踪其计算历史


```python
import torch
x = torch.ones(2, 2, requires_grad=True)
print(x)
#对张量做一次运算
y = x + 2
print(y)

print(y.grad_fn)

#y是计算的结果，所以它有grad_fn属性。

#.requires_grad_(...) 原地改变了现有张量的 requires_grad 标志。如果没有指定的话，默认输入的这个标志是False

```



```python
z = y * y * 3
out = z.mean()
print(z, out)

#因为out是一个标量。所以让我们直接进行反向传播，out.backward()和out.backward(torch.tensor(1.))等价

out.backward()
print(x.grad)
```
现在我们来看一个雅可比向量积的例子

```python
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)


#在这种情况下，y不再是标量。torch.autograd不能直接计算完整的雅可比矩阵，但是如果我们只想要雅可比向量积，只需将这个向量作为参数传给backward

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

#为了防止跟踪历史记录（和使用内存），可以将代码块包装在with torch.no_grad():中。在评估模型时特别有用，因为模型可能具有requires_grad = True的可训练的参数，但是我们不需要在此过程中对他们进行梯度计算。 也可以通过将代码块包装在 with torch.no_grad(): 中，来阻止autograd跟踪设置了 .requires_grad=True 的张量的历史记录

print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)

#output:
#T
#T
#F
```



### 神经网络

nn包则依赖于autograd包来定义模型并对它们求导。一个nn.Module包含各个层和一个forward(input)方法，该方法返回output

```python

import torch 
import torch.nn as nn 
import torch.nn.functional as F

class Net(nn.Module):
        def init(self): 
                super(Net, self).init()
                self.conv1 = nn.Conv2d(1, 6, 5)
                self.conv2 = nn.Conv2d(6, 16, 5)
                # an affine operation: y = Wx + b
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)
        #2x2 Max pooling
        def forward(self, x):
                # Max pooling over a (2, 2) window
                x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
                # If the size is a square you can only specify a single number
                x = F.max_pool2d(F.relu(self.conv2(x)), 2)
                x = x.view(-1, self.num_flat_features(x))
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        def num_flat_features(self, x):
                size = x.size()[1:]  # all dimensions except the batch dimension
                num_features = 1
                for s in size:
                        num_features *= s
                return num_features



```
* 卷积神将网络的计算公式为：
N=(W-F+2P)/S+1
其中N：输出大小
W：输入大小
F：卷积核大小
P：填充值的大小
S：步长大小
上述代码：
输入图像channel：1；输出channel：6；5x5卷积核

* 只需要定义 forward 函数，backward函数会在使用autograd时自动定义，backward函数用来计算导数。我们可以在 forward 函数中使用任何针对张量的操作和计算。 一个模型的可学习参数可以通过net.parameters()返回 

* 计算损失值

```python
output = net(input)
target = torch.randn(10)  # 本例子中使用模拟数据
target = target.view(1, -1)  # 使目标值与数据值尺寸一致
criterion = nn.MSELoss()

```

如果使用`loss`的`.grad_fn`属性跟踪反向传播过程，会看到计算图如下

input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss


* 反向传播

只需要调用loss.backward()来反向传播误差。我们需要清零现有的梯度，否则梯度将会与已有的梯度累加。 现在，我们将调用loss.backward()，并查看 conv1 层的偏置 (bias）在反向传播前后的梯度。 `net.zero_grad() # 清零所有参数(parameter）的梯度缓存

* 更新权重

```python
import torch.optim as optim
# 创建优化器(optimizer）
optimizer = optim.SGD(net.parameters(), lr=0.01)
# 在训练的迭代中：
optimizer.zero_grad()   # 清零梯度缓存
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 更新参数

```


### 训练分类器

通常来说，当必须处理图像、文本、音频或视频数据时，可以使用python标准库将数据加载到numpy数组里。然后将这个数组转化成torch.*Tensor。

对于图片，有 Pillow，OpenCV 等包可以使用; 
对于音频，有 scipy 和 librosa 等包可以使用; 
对于文本，不管是原生 python 的或者是基于 Cython 的文本，可以使用 NLTK 和SpaCy; 
特别对于视觉方面，存在一个包, 名字叫torchvision，其中包含了针对Imagenet、CIFAR10、MNIST 等常用数据集的数据加载器 (data loaders），还有对图像数据转换的操作，即torchvision.datasets和torch.utils.data.DataLoader

步骤：
1. 通过torchvision加载CIFAR10里面的训练和测试数据集，并对数据进行标准化
2. 定义卷积神经网络
3. 定义损失函数
4. 利用训练数据训练网络
5. 利用测试数据测试网络


#### step 1

```python
import torch
import torchvision
import torchvision.transforms as transforms

```
torchvision 数据集加载完后的输出是范围在 [ 0, 1 ] 之间的 PILImage。我们将其标准化为范围在 [ -1, 1 ] 之间的张量。

```python
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

#### step 2&3

```python

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


```

#### step4

```python
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

```

#### step5

```python

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


#那么哪些是表现好的类呢？哪些是表现的差的类呢？

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

```


#### 用alex解释一下参数
https://blog.csdn.net/sinat_42239797/article/details/90646935