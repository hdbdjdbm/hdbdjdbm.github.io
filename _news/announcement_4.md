---
layout: post
title: cs231n的学习笔记 & 计算机底层
date: 2022-01-25 07:59:00-0400
inline: false
---

假期学图像处理的时候上的课程，存档记录当作news

***

### 前言

此篇为 Stanford Winter Quarter 2016 class: CS231n: Convolutional Neural Networks for Visual Recognition课程个人笔记。

五子棋(MCTS)是AlphaZero的一个抛砖引玉，借此机会学习一下神经网络。

> 人类的感知50%以上是通过视觉处理获取的。

这个结论让我回忆起小学三四年级的时候喜欢跳皮筋 ;五六年级喜欢玩4399的flash游戏 ;初一初二开始看小说 ;高一高二开始看视频(追星)。这样梳理发现一切都是在朝着“感知多元化”的方向发展。

卷积模型在1998年就已经被LeCun提出。但当时碍于数据以及硬件大家并不知道这个模型能work out。

Professor Li(FeiFei Li) 把整个data-driven的model类比为小孩的成长过程。眼睛看到的view为输入项，我们的基因就是已经train好的model。由于数据的高维化，让人不得不使用多重卷积去拟合高维。

在听完课程之后，发现了机器学习领域的现状：practice总是走在theory的前面。

我们常说神经网络是个“黑箱”，大致有两层意思。一个是说，我们不能刻画网络具体是在做什么（比如在每一层提取什么样的特征）。另一个是说，我们不知道它为什么在做这些，为什么会有效。

有网友拿牛顿被苹果砸中类比

> 深度学习显然是做对了something, 在某个角度触碰到了真理，但如果不打开黑箱，我们无法知道它到底做对了什么，真理是什么。在牛顿之前，大家都见到了苹果落地。但在当时人们的视角中他们一定认为，苹果落地不是很自然的吗，需要解释、需要知道为什么吗？当时的人们也会认为解释这种现象简直无从下手。跟今天的深度学习有点像吧？但是当牛顿告诉我们为什么苹果会落地之后，世界从此就不一样了。

深度学习目前就好比实验物理，有很多实验观察，但没有理论能够解释。也跟物理学的发展一样，实验总是走在前面。

随着人们接触知识的门槛越来越低(这是好事)，不断有很多提升效果的trick出现，比如Relu替换sigmod函数， batch traning和 normalization提升判别能力，drop out ,利用一些loss fuction等等。有些无谓的提升来源于经验，但是有一些来源于人类已经构建的认知。

实践和理论是分不开的，实践促进对理论的思考，理论引导实践的发展。就像学界和工业界的关系。我们既要展示一个东西能work out, 更讨论为什么能work out。

我的现有感觉是：这个过程是从高维数据提取低维特征。UCB的professor Ma 也提到过这个点(强行碰瓷了属于是，写在todo-list, Mayi老师的高维数据与低维模型)。类似人类区分猫和狗，如果有n个特征：一看原边形状，二看纹理，三看颜色，四看...。到底多少看能区分出猫和狗,每一个看是否能从高维矩阵中提取出来。当然这并不是容易的。CS231n后面的一个lecture就在讨论每一层的意义是看不太出来的(transformer改进了这一点)。

当我们的识别从0到1迈出且到达了之后,就会从1-100飞速快进，还是拿小朋友类比：正常小朋友3岁能识字，5岁能背诗。突然有个小朋友3岁就能背诗,宣布这个结果后，有些小朋友2岁就能背诗(夸张)。💻在目前发展阶段不存在大量的伦理问题(此时必须提到2001太空漫游电影里面的HAL9000对人类说的话：如果有问题，一定是人类的问题)，所以人们会竭尽所能让三个月的“孩子”学会背诗。

最近这一个月沉迷科幻电影无法自拔,一直想写一些观后感,但下笔的时候总找不到一个合适的切入点。结果在学习有关的知识点的时候,反而又有了很多能抒发的角度。(此句话可以翻译：在学习的时候,其他东西都很有意思)

另外，我一直对滥用内卷保有否定态度，对于所有身处学术界的同学们(哪怕只是碰到了学术界的大门)更应该认识到这一点。在前人基础框架打好之后后人要做的就是stand on the shoulders of giant.学术界的内卷是人类更新迭代的必要条件。LeCun也未曾知十年后卷积模型会推动世界的进步，我们应该保有使命感(但要清楚认知自己也是普通人)。

最后，感谢所有在人工智能领域发光发热的人们。


### Lecture 1 Introduction

视觉信息 像素信息

"一些关键词记录"

* 历史先驱(1981诺贝尔医学奖)
每一列的神经元只识别某一种特定的边缘
* Block world(1966,MIT)
人们识别物体是基于基于物体的边缘和形状。
* David Marr：
visual start with a simple 
visual is  HIERARCHICA(分层)
-->边缘结构，2.5D，3D
* normlized cut
* Face Detection
* features(特征)
* part model
* PASCAL Visual Object Challenge
* 2012 年的Image Classification Challenge使用了卷积网络特征+SVM，7层。思路和1998Yann LeCun几乎一样。但那个时候硬件有限制。
* 2015深度残差网络(151层)
  
最终愿想的output:看图讲故事(RNN)


### Lecture 2&3 图像分类

**The problem: semantic gap**

Eg:300 x 100 x 3
* 0-255,三维(亮度，位置，RGB)数据记录局部图像。
1. 由于亮度会变化，需要**鲁棒性**的算法实现稳定性
2. 变形deformation
3. background clutter 等等

solution: data-driven approach

#### kNN classifier

* key point: 比较两个图像：距离的度量 ,the choice of distance and k are hyperpearameters(超参数)
--取决于问题--设置fold - cross validation。Split data into folds,try each fold as validation and average the results。

* notice: Useful for small datasets, but not used too frequently in deep learning. And KNN with pixel distance never used: (1) 一方面在高维度距离本身没有意义(可见上传的另一篇final paper)。(2) 并且距离提供的度量不太准确。(3) test时间比较长

* features
   (1) 距离的度量有不同的选择，比如 $L_1$ 和 $L_2$ 。需要注意不同距离的性质。
   (2) 在对矩阵进行处理的时候注意numpy的处理，比如axis和keepdim，还有一些简便运算的函数比如bincount,argmax等。
   (3) KNN并不是线性分类器，而是将空间分为凸多边形。

* 可供拓展: 对于KNN分类器来说。train需要O(1),但是predict需要 O(N). 很BAD，所以有一些fast/approximate的方法，比如k-d tree或者 ball- tree.


#### 线性分类

![image-1](https://pic.imgdb.cn/item/61dd3a492ab3f51d91818262.png)

![image-2](https://pic.imgdb.cn/item/61dd3a492ab3f51d91818266.png)

W: reshaping，只是在计数，消除失真。 

* key point: 用线性函数分隔空间

* features
Interpreting a Linear Classifier：W是类别的矩阵

* quantifing what it seems to have a god W:Loss function

#### SVM loss

the SVM loss has the form:

$$
L_i = \sum_{j\neq y_i}max(0,s_j - s_{y_i} + 1)
$$

$$
f(x,W)=  Wx
$$

$$
L = \frac{1}{N}\sum_{i=1}^N\sum_{j\neq y_i}max(0,f(x_i;W)_j-f(x_i;W)_{y_i} +1 )
$$

w is not unique, we want regularization:

$$
L(W) = \frac{1}{N}\sum_{i=1}^{N}L_i(f(x_i,W),y_i) + \lambda R(W)
$$


前者是Data loss: Model predictions
should match training data

后者是Regularization: Prevent the model
from doing too well on training data，$\lambda$ is regularization strength
(hyperparameter)。$R(W)$ = $\sum\sum w^2$





#### softmax loss(Multinamial Logistic Regression)

$$
s = f(x_i;W)
$$

$$
P(Y=k|X=x_i) = \frac{e^sk}{\sum_j e^{sj}}
$$

$$
L_i = -log(\frac{e^{sy_i}}{\sum_j e^{sj}})
$$

**SVM loss VS softmax loss**



* 如何最快的找到呢：
#### Optimization

numerical fradient
&
analytical gradient

> always use the analytical gradient but to check the numerical gradient this is called gradient check


Stochastic Gradient Descent (SGD)

Approximate sum
using a minibatch of examples

### Lecture 4 Neural Networks and Backpropagation

question：How to find the best W $\triangledown_WL$


#### features of pictures
Color Histogram and Histogram of Oriented Gradients (HoG)

#### Neural Networks

* defination

分层计算

![image-5](https://pic.imgdb.cn/item/61eebaff2ab3f51d9195a978.png)


* activation function

激活函数

![image-6](https://pic.imgdb.cn/item/61eebbb32ab3f51d91966888.png)
(ReLU is a good default
choice for most problems)

* training a 2-layer Neural Network needs:Define the network,Forward pass,Calculate the analytical gradients

* Setting the number of layers and their sizes
![image-7](https://pic.imgdb.cn/item/61eebd4c2ab3f51d91980d2a.png)

![image-8](https://pic.imgdb.cn/item/61eebd4c2ab3f51d91980d31.png)

$$
s = f(x;W_1,W_2) = W_2max(0,W_1x)
$$ 
Nonlinear score function
$$
L_i = \sum_{j\neq y_i}max(0,s_j - s_{y_i} + 1)
$$
SVM Loss on predictions
$$
R(W) = \sum_k W_k^2
$$
Regularization
$$
L = \frac{1}{N}\sum_{i=1}^N L_i + \lambda R(W_1) + \lambda R(W_2)
$$

question: compute the gradient of $W_1$ and $W_2$
solution: Backpropagation

![image-9](https://pic.imgdb.cn/item/61eebf462ab3f51d919a43de.png)



### Lecture 5 setup notice

#### Mini-batch SGD
![image-5](https://pic.imgdb.cn/item/61f3eaaa2ab3f51d912c093d.png)
1. Sample a batch of data
2. Forward prop it through the graph, get loss
3. Backprop to calculate the gradients
4. Update the parameters using the gradient

#### activation functions
1. Sigmoid

   $$
   \sigma(x) = \frac{1}{1+e^{-x}}
   $$


   festure:值域在[0,1]
   DISAD: 
   Saturated neurons “kill” the gradients; Sigmoid outputs are not zerocentered(如果输入是+的,那么gradient也是正的); 
   exp() is a bit compute expensive;

2. tanh(x)
   Squashes numbers to range [-1,1]
   zero centered (nice)
   still kills gradients when saturated



3. ReLU(Rectified Linear Unit)
   
   $$
   f(x) = max(0,x)
   $$

   AD:
   Converges much faster than sigmoid/tanh in practice

   DISAD:
   not zero-centered output
   an annoyance

4. Leaky ReLU
   
   $$
   f(x) = max(0.01x,x)
   $$

   Parametric Rectifier (PReLU):
   $$
   f(x) = max(ax,x)
   $$

5. ELU
   
   $$
   if x>0: x
   if x <= 0: \alpha(exp(x)-1)
   $$

6. Maxout 
   $max(w_1^Tx+b_1,w_2^Tx+b_2)$

summary:
- Use ReLU. Be careful with your learning rates
- Try out Leaky ReLU / Maxout / ELU
- Try out tanh but don’t expect much
- **Don’t use sigmoid**

#### data processing

some examples

![image-10](https://pic.imgdb.cn/item/61f3e2c42ab3f51d912255a8.png)

![image-11](https://pic.imgdb.cn/item/61f3e2c42ab3f51d912255ad.png)

* it's common to zero-centered data but not normalized(why?)
* use PCA and whitening(common in ML not in image)
* substract the maen image - substract per-cha nel mean

#### weight initialization

* W=0 is a bad idea cause每一个神经元表现的一样，不好寻找gradient
  
* instaed W = 0.01 * np.random.randn(D,H) (对small networks are okay, but can lead to non-homogeneous distributions of activsations across the layers of a network)

* when debug, sometimes try 0.1* np.random,rand(D,H)

#### Batch Normalization

![image-13](https://pic.imgdb.cn/item/61f3eaaa2ab3f51d912c0941.png)

AD:Improves gradient flow ;Allows higher learning rates ;Reduces the strong dependence on initialization ;Acts as a form of regularization ;slightly reduces the need for dropout

需要注意和data processing 的normlize都是在normlize, 目的不一样,前者是为了SGD


step to train :
* Step 1: Preprocess the data
* Step 2: Choose the architecture
  Double check that the loss is reasonable
  loss went up, good. (sanity check)


#### babysitting the learning process

* Double check that the loss is reasonable:
* Make sure that you can overfit very small portion of the training data(100% accurancy we said overfit the data)

tips:

loss not going down:learning rate too low

loss exploding:learning rate too high

#### Hyperparameter Optimization

Cross-validation strategy

reg and lr: it’s best to optimize
in log space


* gap between training and validation:
   big gap = overfitting
   => increase regularization strength

   no gap
   => increase model capacity?

* Evaluation
model ensembles

### Lecture 6 continue setup notice

#### parameter update

一些个下降方法：
* momentum update(soleve the SGD slow)
* Nesterov Monmentum update
* nag
* AdaGrad update
* PMSProp update
* Adam update(looks a bit like RMSProp with momentum)
* second order optimization methods(no hyperparameters)
* BFGS(Quasi-Newton methods)
* L-BFGS(some features)

**summary**
Adam is good default choice

If you can afford to do full batch updates then try out
L-BFGS 

#### Evaluation:
Model Ensembles

Train multiple independent models,At test time average their results

> Enjoy 2% extra performance

#### Regularization (dropout)
![image-1](https://pic.imgdb.cn/item/61f3ef922ab3f51d91318849.png)

reason:
* Forces the network to have a redundant representation
* Dropout is training a large ensemble of models (that share parameters).

drop in forward pass,scale at test time

#### explanation about the parameter

learning rate(决定了我们训练网络速率的快慢,how which step we used?)


regularization(dropout)
* no need so many features
* 另一种形式的ensemble
*  Inberted fropout

#### Convolutional Neural Networks


### Lecture 7 convolution neural network的结构


#### Convolution Layer

一些看图说话环节：

反正就是多维空变换,找到一些特征

![image-1](https://pic.imgdb.cn/item/61f3fc9a2ab3f51d9140f0cf.png)

![image-2](https://pic.imgdb.cn/item/61f3fc9a2ab3f51d9140f0d3.png)

![image-3](https://pic.imgdb.cn/item/61f3fc9a2ab3f51d9140f076.png)

![image-4](https://pic.imgdb.cn/item/61f3fc9a2ab3f51d9140f07d.png)


A closer look at spatial dimensions:可以change一些stride和pad, 由于会压缩, 可以在边缘加一些0

#### Pooling layer

* makes the representations smaller and more manageable
* operates over each activation map independently
![image-2](https://pic.imgdb.cn/item/61f3fef32ab3f51d9143f1bc.png)


#### fully connected layer
Contains neurons that connect to the entire input volume, as in ordinary Neural Networks


take a look:

![image-3](https://pic.imgdb.cn/item/61f3fef32ab3f51d9143f1b8.png)

#### case study

LeNet(1998)
AlexNet(2012)
ZFNet(2013)
VGG(2014)
GooLeNet(2014)
ResNet(2015)

all there are used to clssification

### Lecture 8 Localization and Detection

![image-1](https://pic.imgdb.cn/item/61f4017d2ab3f51d91473cc5.png)

#### claassification + Localization

* classification
input:image
output:class label
evaluation metric:accuracy

* Localization

input:image
output:box in the image(x,y,w,h)
evaluation metric:intersection over the Union
Classification + Localization: Do both


Idea #1: Localization as Regression

使用回归output box coordinate

![image-1](https://pic.imgdb.cn/item/61f4031e2ab3f51d91497e1b.png)


input:image --(Neural Net)--> output:Box coordinates(4 numbers)

compare with Correct output:box coordinates(4 numbers)


Idea #2 sliding window

使用窗口分类

![image-2](https://pic.imgdb.cn/item/61f4031e2ab3f51d91497e12.png)

* Run classification + regression network at multiple locations on a highresolution image


#### detection

如果使用regression则参数就太多了

Detection as Classification

Problem: Need to test many positions and scales

Solution: If your classifier is fast enough, just do it

Before CNNs:

2005 Histogram of Oriented Gradients(HoG)

2010 deformable parts modek(DPM)

* detection as clssification
region proposals: selective search


putting it together:R-CNN
step:
1. Train (or download) a classification model for ImageNet(AlexNet)
2. Fine-tune model for detection
3. extract features
4. Train one binary SVM per class to classify region features
5. bbox regression

Evaluation: mAP

problem:slow at test time,

solution:
fast R-CNN
fasyer R-CNN

### Lecture 9 deep understanding

layers越往后越不好解释

t-SNE visualization

Decinv approaches 

optimaize the image

一些trick

#### deep dream

#### Neural Style

#### Adversarial Examples

### Lecture 10 RNN

这一块暂略, 暂时永不太上。

#### Recurrent Neural Networks


![image-1](https://pic.imgdb.cn/item/61f40be02ab3f51d9155e8c9.png)

one to one: Vanilla Neural Networks

one to many: Image Captioning(image -> sequence of words)

many to one: Sentiment Classification sequence of words -> sentiment

many to many:Machine Translation(seq of words -> seq of words)

many to many: Video classification on frame level

#### LSTM

Long short Term Memory(1997)



### Lecture 11 practice with CNN


#### data augmentation
(useful for small dataset)
* horizontal flips
* random crops/scales
* color jitter

#### transfer learning
can freeze some parameter just train the last few layers

![image-1](https://pic.imgdb.cn/item/61f40f292ab3f51d915aad27.png)

![image-2](https://pic.imgdb.cn/item/61f40f292ab3f51d915aad2b.png)

#### how to stack the convolution

* Layers :change the size of 仿射结果

* im2col

#### computing convolution
* n2cool:easy to implement, but big memory overhead
* FFT:big speeduos for small kernels
* fast algorithms seem promising, not wideljy used

#### GPU,CPU

并行和浮点数


### Lecture 12 sogtware packages

#### Caffe

from U.C. Berkely

Main classes 
* Blob: stores data and derivatives
* Layer:Trandforms bottom blobs to top blobs
* Net: many layers; computes gradients
* solver:uses gradient toupdate weights

steps:
1. convert data
2. define net
   write prototxt
3. define solver
   write prototxt
4. train

caffe:model zoo

Pros/Cons
+ good for feedforward
- need to write C++
- not good recurrent networks

#### Torch

from NYU
wriiten in C and Lua

Torch: Tensors

Torch: nn
nn module lets you easily build and train neural nets
picture

Torch: cunn
picture

Torch: optim
picture

Torch: Modules

writing your own modules is easy

Torch: nngraph

Pros/Cons
- not great for RNNs
- 


#### Theano

#### TensorFlow

from Google

TensorFlow:Tensorboard
way to visualize what's happening inside your odels



### Lecture 13 

#### Segmentation

pictures

* semantic segmentation
label every pixel in the image(repeat for clssification for every picel)
unsample
unpooling
skip connections can help

* instance segmentation
very similiar to R-CNN
detect instance, generate mask
similar pipelines to object detection
 
region classification
region refinement

do logistic regression to classify


#### soft attention


soft attention for translation
soft attention for other field

#### hard attention
pass
need reinforcement learning

### Lecture 14 Videos and Unsupervised Learning
not done yet


### assignment1

#### Q1 K-Nearest Neighbor classifier

```python
class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k = 1, num_loops = 0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        if num_loops == 1:
            pass
        return self.predict_labels(sidts, k=k)
    
    def compute_distances_two_loops(self, X):
        pass

    def predict_labels(self,dists,k=1):
        pass
```

注意numpy的应用和boardcasting, **少用循环**。


#### Q2&3 Training a Support Vector Machine & Implement a Softmax classifier

> subtract the mean image from train and test data


```python

class LinearClassifier(object):
    def __init__(self):
        self.W = None
    def train(self, X, y, learning_rate=1e-3, reg=1e-5,num_iters=100,batch_size=200, verbose=False):
        pass
    
    def predict(self,X):
        pass

    def loss(self, X_batch, y_batch, reg):
        pass



class LinearSVM(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self, X_batch, y_batch, reg)
    
    
class Softmax(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)



```

同样, 注意向量的应用.


#### Q4 Two-Layer Neural Network

```python

class TwoLayerNet(object):
    def __init__(self,input_size, hidden_size, output_size, std=1e-4):

        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        pass

    def train(self, X, y, X_val, y_val,learning_rate=1e-3, learning_rate_decay=0.95,reg=5e-6, num_iters=100,batch_size=200, verbose=False):
        pass

    def prefict(self,X):
        pass
        
```

注意反向传播的part.

#### Q5 Higher Level Representations: Image Features

hog_feature 
color_histogram_hsv

### assignment2


#### Q1 Multi-Layer Fully Connected Neural Networks 

```python

class TwoLayerNet(object):
    pass
class FullyConnectedNet(object):
    pass
```

train又是另一个专门的类，cause model有很多, 但train的动作都是相似的。

#### Q2 Batch Normalization 

> with batch normalization we can avoid the problem of vanishing and exploding gradients because it normalizes every affine layer (xW+b), avoiding very large/small values. Moreover, its regularization properties allow to decrease overfitting.

> the batch size affects directly the performance of batch normalization (the smaller the batch size the worse). Even the baseline model outperforms the batchnorm model when using a very small batch size. This problem occurs because when we calculate the statistics of a batch, i.e., mean and variance, we try to find an approximation of the statistics of the entire dataset. Therefore with a small batch size, these statistics can be very noisy. On the other hand, with a large batch size we can obtain a better approximation.

other idea: **Layer Normalization**


#### Q3 Dropout 

dropout可以减少一些过拟合现象

#### Q4 Convolutional Neural Networks

记录一些小trick

#### Q5 PyTorch/TensorFlow on CIFAR-10 

We want you to stand on the shoulders of giants!


### assingment3

这一块暂时还没有目的性, 后续补

#### Q1: Image Captioning with Vanilla RNNs (30 points)

#### Q2: Image Captioning with Transformers (20 points)

#### Q3: Network Visualization: Saliency Maps, Class Visualization, and Fooling Images (15 points)

#### Q4: Generative Adversarial Networks (15 points)

#### Q5: Self-Supervised Learning for Image Classification (20 points)

### 后记


当然除了以上的前言以外, 更多的疑问是：未来走向何方？

中国传统制造业在上世纪崛起，这个世纪逐渐边缘。机器学习也会边缘吗？
<!--我心里有一个答案：时机不知道。谁知道下一个伟人什么时候出现。但是另一方面，出现的分布倒是有的。-->

<!--想象力和创造力并且有行动力。到底在哪一个领域会实现？-->

再往前，倒退回2005年, 复盘一下触屏手机的出现。手机一直都有，但乔布斯让时代进入next level。一直到现在手机还在迭代更新，there is no endless。2015年cs231n的开课，直到今年2022图像领域依旧在百花齐放。同理, there is no endless, 因为这是人类认知世界的方式和工具。我的观点是：这个领域没有问题。就像查字典和搜索引擎一样, 只是方式变了，纸质变成了电子。照着人类的进化来说，只要眼睛还在，人类的感知没有进化，那么认知世界的方式也不会进化。只是接收信息的方法变化。

所以只要是人类不进化，现有的任何满足当前人类处理各种事情的方法和方式都不会"过时"。


<!--那其实很明显了，认识世界的方式，比如硬件包括投影,电池，新能源就是next generation. 这是进化的方向。-->


# 计算机基础

# 写在开篇

本篇试读人士：只上过大学生计算机基础，其他一无所知。

以下大部分均摘自书籍：深入理解计算机系统。

此篇文章来源于想理解进程和线程的区别。但经曾老板指导后打算学习一些计算机底层的基本信息，此系列将会持续更新。踏上了一些未曾设想的读后感道路。

有人图某件事成，有人图某件事。我属于后者，我乐意做一些无用功，只是为了满足自己的求知欲。




# 从跟踪 hello 程序的生命周期开始

```c
#include <stdio.h>

int main()
{
    printf("hello, world\n");
    return O;
}
```

**信息就是位+(plus)上下文** 

hello程序的生命周期是从一个源程序开始的，文件名是 hello.c。实际上就是一个由值 0 和 1 组成的位(又称为比特)序列，8个位被组织成一组，称为字节。每个字节表示程序中的某些文本字符。大部分的现代计算机系统都使用 ASCII 标准来表示文本字符，这种方式实际上就是用一个唯一的单字节大小的整数值来表示每个字符。
![image-1](https://pic.imgdb.cn/item/61d68be42ab3f51d91084258.png)

>二进制

每个文本行都是以一个看不见的换行符来结束的，它所对应的整数值为 10。像 hello.c 这样只只由ASCII 字符构成的文件称为文本文件，所有其他文件都称为二进制文件。(其他文件指的是？)

*tips*
--
系统中所有的信息，包括磁盘文件、内存中的程序、内存中存放的用户数据以及网络上传送的数据，都是由一串比特表示的。区分不同数据对象的唯一方法是我们读到这些数据对象时的上下文。

**程序被其他程序翻译成不同的格式**

hello 程序的生命周期是从 C 语言程序开始的，因为这种形式能够被人读懂。然而，为了在系统上运行 hello.c 程序，每条 C 语句都必须被其他程序转化为一系列的低级机器语言指令。然后这些指令按照一种称为可执行目标程序的格式打好包，并以二进制磁盘文件的形式存放起来。目标程序也称为可执行目标文件。

在 Unix 系统上，从源文件到目标文件的转化是由编译器驱动程序完成的
```unix
linux> gcc -o hello hello.c
```
GCC 编译器驱动程序读取源程序文件 hello.c并把它翻译成一个可执行目标文件 hello。这个翻译过程可分为四个阶段完成。执行这四个阶段的程序(预处理器、编译器、汇编器和链接器)一起构成了编译系统（compilation system)

![image-2](https://pic.imgdb.cn/item/61d68e2d2ab3f51d910a2fc7.png)

>本人os: 抽烟.jpg 感谢前人，真心的


**处理器读并解释储存在内存中的指令** 

此刻，hello.c 源程序已经被编译系统翻译成了可执行目标文件 hello, 并被存放在磁盘上。要想在 Unix 系统上运行该可执行文件，我们将它的文件名输人到称为 shell 的应用程序中：
```shell
linux> ./hello
hello, world
linux>
```
shell 是一个命令行解释器，它输出一个提示符，等待输人一个命令行，然后执行这个命令。如果该命令行的第一个单词不是一个内置的 shell 命令，那么 shell 就会假设这是一个可执行文件的名字，它将加载并运行这个文件。所以在此例中，shell 将加载并运行hello 程序，然后等待程序终止。hello 程序在屏幕上输出它的消息，然后终止。shell随后输出一个提示符，等待下一个输人的命令行。

为了更好解释内部发生了什么，先了解一个典型系统的硬件组织。这张图是近期 Intel 系统产品族的模型。

![image-2](https://pic.imgdb.cn/item/61d6a4aa2ab3f51d911c9577.png)

* **总线** ： 贯穿整个系统的是一组电子管道，称作总线，它携带信息字节并负责在各个部件间传递。**通常总线被设计成传送定长的字节块，也就是字(word)。字中的字节数(即字长)是一个基本的系统参数**，各个系统中都不尽相同。现在的大多数机器字长要么是 4 个字节(32位)， 要么是 8 个字节(64 位)。

* **I/O(输入/输出)** ： I/O 设备是系统与外部世界的联系通道。我们的示例系统包括四个 I/O 设备：作为用户输入的键盘和鼠标，作为用户输出的显示器，以及用于长期存储数据和程序的磁盘驱动器(简单地说就是磁盘)。**每个 I/O 设备都通过一个控制器或适配器与 I/O 总线相连**。控制器是 I/O 设备本身或者系统的主印制电路板(通常称作主板)上的芯片组。而适配器则是一块插在主板插槽上的卡。但无论如何它们的功能都是在 I/O 总线和 I/O 设备之间传递信息。

* **主存** ： 主存是一个临时存储设备，在处理器执行程序时，用来存放程序和程序处理的数据。从物理上来说，主存是由一组动态随机存取存储器(DRAM)芯片组成的。**从逻辑上来说，存储器是一个线性的字节数组，每个字节都有其唯一的地址(数组索引)， 这些地址是从零开始的**。一般来说，组成程序的每条机器指令都由不同数量的字节构成。与 C 程序变量相对应的数据项的大小是根据类型变化的。比如，在运行 Linux 的 X86-64 机器上，short 类型的数据需要 2 个字节，int 和 float 类型需要 4 个字节，而 long 和 double 类型需要 8 个宇节。


* **处理器** ： 中央处理单元(CPU), 简称处理器，是解释(或执行)存储在主存中指令的**引擎**。处理器的核心是一个大小为一个字的**存储设备(或寄存器)， 称为程序计数器(PC)**。在任何时刻，PC 都指向主存中的某条机器语言指令(即含有该条指令的地址)。**处理器从程序计数器指向的内存处读取指令，解释指令中的位，执行该指令指示的简单操作，然后更新 PC使其指向下一条指令**，而这条指令并不一定和在内存中刚刚执行的指令相邻。这样的简单操作并不多，它们围绕着主存、寄存器文件(register file)和算术/逻辑单元(ALU)进行。**寄存器文件是一个小的存储设备，由一些单个字长的寄存器组成，每个寄存器都有唯一的名字**。ALU 计算新的数据和地址值。下面是一些简单操作的例子，CPU 在指令的要求下可能会执行这些操作。
> 加载：从主存复制一个字节或者一个字到寄存器，以覆盖寄存器原来的内容。

> 存储：从寄存器复制一个字节或者一个字到主存的某个位置，以覆盖这个位置上原
来的内容。

> 操作：把两个寄存器的内容复制到 ALU, ALU 对这两个字做算术运算，并将结果存放到一个寄存器中，以覆盖该寄存器中原来的内容。

> 跳转：从指令本身中抽取一个字，并将这个字复制到程序计数器（PC)中，以覆盖PC 中原来的值。

**运行 hello 程序** 

![image-4](https://pic.imgdb.cn/item/61d6aa752ab3f51d91211cbd.png)

初始时，shell 程序执行它的指令，等待我们输人一个命令。当我们在键盘上输人字符串/hello后，shell 程序将字符逐一**读入寄存器，再把它存放到内存中**。

当我们在键盘上敲回车键时，shell 程序就知道我们已经结束了命令的输人。然后shell 执行一系列指令来加载可执行的 hello 文件，**这些指令将 hello 目标文件中的代码和数据从磁盘复制到主存**。数据包括最终会被输出的字符串Hello,world。

**利用直接存储器存取(DMA)技术，数据可以不通过处理器而直接从磁盘到达主存。**

![image-5](https://pic.imgdb.cn/item/61d6ab8b2ab3f51d9121f901.png)

**一旦目标文件 hello 中的代码和数据被加载到主存，处理器就开始执行 hello 程序的 main 程序中的机器语言指令。这些指令将Hello, world字符串中的字节从主存复制到寄存器文件，再从寄存器文件中复制到显示设备，最终显示在屏幕上。**

![image-6](https://pic.imgdb.cn/item/61d6abed2ab3f51d91223c87.png)

**高速缓存**

这个简单的示例揭示了一个重要的问题，即系统花费了大量的时间把信息从一个地方挪到另一个地方。hello 程序的机器指令最初是存放在**磁盘**上，当程序加载时，它们被复制到**主存**；当处理器运行程序时，指令又从主存复制到**处理器**。相似地，**数据串**开始时在磁盘上，然后被复制到主存，最后从主存上复制到显示设备。从程序员的角度来看，这些复制就是开销，减慢了程序“真正”的工作。因此，系统设计者的一个主要目标就是使这些复制操作尽可能快地完成。


一个典型的寄存器文件只存储几百字节的信息，而主存里可存放几十亿字节。然而，**处理器从寄存器文件中读数据比从主存中读取几乎要快 100 倍**。更麻烦的是，随着这些年**半导体**技术的进步，这种处理器与主存之间的差距还在持续增大。**加快处理器的运行速度比加快主存的运行速度要容易和便宜得多**。

针对这种处理器与主存之间的差异，系统设计者采用了更小更快的存储设备，**称为高速缓存存储器(cache memory, 简称为 cache 或高速缓存)，作为暂时的集结区域，存放处理器近期可能会需要的信息。**


位于处理器芯片上的 L1 高速缓存的容量可以达到数万字节，访问速度几乎和访问寄存器文件一样快。一个容量为数十万到数百万字节的更大的 L2 高速缓存通过一条特殊的总线连接到处理器。进程访问 L2 高速缓存的时间要比访问 L1 高速缓存的时间长 5 倍，但是这仍然比访问主存的时间快 5-10 倍。L1 和 L2 高速缓存是用一种叫做静态随机访问存储器(SRAM)的硬件技术实现的。通过让高速缓存里存放可能经常访问的数据，大部分的内存操作都能在快速的高速缓存中完成。

![image-7](https://pic.imgdb.cn/item/61d6ad2d2ab3f51d9123373f.png)


**存储设备形成层次结构**

每个计算机系统中的存储设备都被组织成了一个存储器层次结构，在这个层次结构中，**从上至下，设备的访问速度越来越慢、容量越来越大，并且每字节的造价也越来越便宜。** 寄存器文件在层次结构中位于最顶部。

![image-8](https://pic.imgdb.cn/item/61d6ae042ab3f51d9123f19b.png)

存储器层次结构的主要思想是上一层的存储器作为低一层存储器的高速缓存。因此，寄存器文件就是 L1 的高速缓存，L1 是 L2 的高速缓存，L2 是 L3 的高速缓存，L3 是主存的高速缓存，而主存又是磁盘的高速缓存。在某些具有分布式文件系统的网络系统中，本地磁盘就是存储在其他系统中磁盘上的数据的高速缓存。

**操作系统管理硬件**

我们可以把操作系统看成是应用程序和硬件之间插人的一层软件。所有应用程序对硬件的操作尝试都必须通过操作系统。

操作系统有两个基本功能：
(1)防止硬件被失控的应用程序滥用
(2)向应用程序提供简单一致的机制来控制复杂而又通常大不相同的低级硬件设备。操作系统通过几个基本的抽象概念（进程、虚拟内存和文件）来实现这两个功能。文件是对 I/O 设备的抽象表示，虚拟内存是对主存和磁盘 I/O 设备的抽象表示，进程则是对处理器、主存和 I/O 设备的抽象表示。

![image-8](https://pic.imgdb.cn/item/61d6b16e2ab3f51d9127023d.png)


* **进程**

程序看上去是独占地使用处理器、主存和 I/O设备。处理器看上去就像在不间断地一条接一条地执行程序中的指令，即该程序的代码和数据是系统内存中唯一的对象。这些假象是通过进程的概念来实现的，进程是计算机科学中最重要和最成功的概念之一。


**进程是操作系统对一个正在运行的程序的一种抽象**。在一个系统上可以同时运行多个进程，而每个进程都好像在独占地使用硬件。而并发运行，则是说一个进程的指令和另一个进程的指令是交错执行的。在大多数系统中，**需要运行的进程数是多于可以运行它们的CPU个数的。** 传统系统在一个时刻只能执行一个程序，而先进的多核处理器同时能够执行多个程序。无论是在单核还是多核系统中，一个 CPU 看上去都像是在并发地执行多个进程，这是通过处理器在进程间切换来实现的。**操作系统实现这种交错执行的机制称为上下文切换。**



操作系统保持跟踪进程运行所需的所有状态信息。这种状态，也就是上下文，包括许多信息，比如 PC 和寄存器文件的当前值，以及主存的内容。**在任何一个时刻，单处理器系统都只能执行一个进程的代码。当操作系统决定要把控制权从当前进程转移到某个新进程时，就会进行上下文切换，即保存当前进程的上下文、恢复新进程的上下文，然后将控制权传递到新进程。新进程就会从它上次停止的地方开始。** 


示例场景中有两个并发的进程：shell 进程和 hello 进程。最开始，只有 shell 进程在运行，即等待命令行上的输人。**当我们让它运行 hello 程序时，shell 通过调用一个专门的函数，即系统调用，来执行我们的请求，系统调用会将控制权传递给操作系统。操作系统保存 shell 进程的上下文，创建一个新的 hello 进程及其上下文，然后将控制权传给新的 hello 进程。hello 进程终止后，操作系统恢复 shell 进程的上下文，并将控制权传回给它,shell 进程会继续等待下一个命令行输人。**

![image-9](https://pic.imgdb.cn/item/61d6b37d2ab3f51d9128cbc1.png)

**从一个进程到另一个进程的转换是由操作系统内核(kernel)管理的。内核是操作系统代码常驻主存的部分。** 当应用程序需要操作系统的某些操作时，比如读写文件，它就执行一条特殊的系统调用(system call)指令，**将控制权传递给内核**。然后内核执行被请求的操作并返回应用程序。**注意，内核不是一个独立的进程。相反，它是系统管理全部进程所用代码和数据结构的集合。** 

* **线程**

尽管通常我们认为一个进程只有单一的控制流，但是在现代系统中，**一个进程实际上可以由多个称为线程的执行单元组成，每个线程都运行在进程的上下文中，并共享同样的代码和全局数据。由于网络服务器中对并行处理的需求，线程成为越来越重要的编程模型**，因为多线程之间比多进程之间更容易共享数据，也因为线程一般来说都比进程更高效。当有多处理器可用的时候，多线程也是一种使得程序可以运行得更快的方法。


* **虚拟内存**

虚拟内存是一个抽象概念，它为每个进程提供了一个假象，即每个进程都在独占地使用主存。每个进程看到的内存都是一致的，称为虚拟地址空间。在 Linux 中，地址空间最上面的区域是保留给操作系统中的代码和数据的，这对所有进程来说都是一样。地址空间的底部区域存放用户进程定义的代码和数据。请注意，图中的地址是从下往上增大的。
![image-6](https://pic.imgdb.cn/item/61d6b46a2ab3f51d9129b10e.png)

>**程序代码和数据**。对所有的进程来说，代码是从同一固定地址开始，紧接着的是和C全局变量相对应的数据位置。代码和数据区是直接按照可执行目标文件的内容初始化的。

>**堆**。代码和数据区后紧随着的是运行时堆。代码和数据区在进程一开始运行时就被指定了大小，与此不同，**当调用像 malloc 和 free 这样的 C 标准库函数时，堆可以在运行时动态地扩展和收缩**。

> **共享库**。大约在地址空间的中间部分是一块用来存放像 C 标准库和数学库这样的共享库的代码和数据的区域。

> **栈**。位于用户虚拟地址空间顶部的是用户栈，编译器用它来实现函数调用。和堆一样，用户栈在程序执行期间可以动态地扩展和收缩。

> **内核虚拟内存**。地址空间顶部的区域是为内核保留的。不允许应用程序读写这个区域的内容或者直接调用内核代码定义的函数。相反，它们必须调用内核来执行这些操作。

* **文件**

文件就是字节序列，仅此而已。每个 I/O 设备，包括磁盘、键盘、显示器，甚至网络，都可以看成是文件。系统中的所有输人输出都是通过使用一小组称为 Unix I/O 的系统函数调用读写文件来实现的。

**系统之间利用网络通信**

从一台主机复制信息到另外一台主机已经成为计算机系统最重要的用途之一。比如，像电子邮件、即时通信、万维网等这样的应用都是基于网络复制信息的功能。

我们可以使用熟悉的 telnet 应用在一个远程主机上运行 hello 程序。假设用本地主机上的 telnet 客户端连接远程主机上的 telnet 服务器。在我们登录到远程主机并运行 shell 后，远端的 shell 就在等待接收输入命令。此后在远端运行 hello 程序。


![image-4](https://pic.imgdb.cn/item/61d7bc312ab3f51d91d23d43.png)


**并发(concurrency)与并行（parallelism)**

并发是一个通用的概念，指一个同时具有多个活动的系统；而并行指的是用并发来使一个系统运行得更快。并行可以在计算机系统的多个抽象层次上运用。在此，我们按照系统层次结构中由高到低的顺序重点强调三个层次。

**1 线程级并发**

构建在进程这个抽象之上，我们能够设计出同时有多个程序执行的系统，这就导致了并发。使用线程，我们甚至能够在一个进程中执行多个控制流。传统意义上，这种并发执行只是模拟出来的，是通过使一台计算机在它正在执行的进程间快速切换来实现的。当构建一个由单操作系统内核控制的多处理器组成的系统时，我们就得到了一个多处理器系统。

![image-7](https://pic.imgdb.cn/item/61d7bde52ab3f51d91d34853.png)

微处理器芯片有 4 个 CPU 核，每个核都有自己的 L1 和
L2 高速缓存，其中的 L1 高速缓存分为两个部分。一个保存最近取到的指令，另一个存放数据。这些核共享更高层次的高速缓存，以及到主存的接口。

**超线程，有时称为同时多线程(simultaneous multi-threading), 是一项允许一个CPU执行多个控制流的技术**。它涉及 CPU 某些硬件有多个备份，比如程序计数器和寄存器文件，而其他的硬件部分只有一份，比如执行浮点算术运算的单元。常规的处理器需要大约20 000 个时钟周期做不同线程间的转换，而超线程的处理器可以在单个周期的基础上决定要执行哪一个线程。**这使得 CPU 能够更好地利用它的处理资源**。比如，假设一个线程必须等到某些数据被装载到高速缓存中，那 CPU 就可以继续去执行另一个线程。


**2 指令级并行**

在较低的抽象层次上，现代处理器可以同时执行多条指令的属性称为指令级并行。

**3 单指令、多数据并行**

在最低层次上，许多现代处理器拥有特殊的硬件，允许一条指令产生多个可以并行执行的操作，这种方式称为单指令、多数据，即 SIMD 并行。例如，较新几代的 Intel 和AMD 处理器都具有并行地对 8 对单精度浮点数(C 数据类型 float)做加法的指令。


**抽象**

抽象的使用是计算机科学中最为重要的概念之一。例如，为一组函数规定一个简单的应用程序接口(AH)就是一个很好的编程习惯，程序员无须了解它内部的工作便可以使用这些代码。不同的编程语言提供不同形式和等级的抽象支持。

![image-8](https://pic.imgdb.cn/item/61d7bff42ab3f51d91d49250.png)

在处理器里，指令集架构提供了对实际处理器硬件的抽象。使用这个抽象，机器代码程序表现得就好像运行在一个一次只执行一条指令的处理器上。底层的硬件远比抽象描述的要复杂精细，它并行地执行多条指令，但又总是与那个简单有序的模型保持一致。在学习操作系统时，我们介绍了三个抽象：文件是对 I/O 设备的抽象，虚拟内存是对程序存储器的抽象，而进程是对一个正在运行的程序的抽象。虚拟机，它提供对整个计算机的抽象，包括操作系统、处理器和程序。


小结
--
计算机系统是由硬件和系统软件组成的，它们共同协作以运行应用程序。计算机内部的信息被表示为一组组的位，它们依据上下文有不同的解释方式。程序被其他程序翻译成不同的形式，开始时是ASCII 文本，然后被编译器和链接器翻译成二进制可执行文件。

处理器读取并解释存放在主存里的二进制指令。因为计算机花费了大量的时间在内存、I/O 设备和CPU 寄存器之间复制数据，所以将系统中的存储设备划分成层次结构层的硬件高速缓存存储器、DRAM 主存和磁盘存储器。在层次模型中，位于更高层的存储设备比低层的存储设备要更快，单位比特造价也更高。层次结构中较高层次的存储设备可以作为较低层次设备的高速缓存。通过理解和运用这种存储层次结构的知识，程序员可以优化 C 程序的性能。

操作系统内核是应用程序和硬件之间的媒介。它提供三个基本的抽象：1)文件是对 I/O 设备的抽攀；2)虚拟内存是对主存和磁盘的抽象；3)进程是处理器、主存和 I/O 设备的抽象。
最后，网络提供了计算机系统之间通信的手段。从特殊系统的角度来看，网络就是一种 I/O 设备。





