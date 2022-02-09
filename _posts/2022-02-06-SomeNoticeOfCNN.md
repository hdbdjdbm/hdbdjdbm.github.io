---
layout:     post
title:      神经网络的十万个为什么
subtitle:   持续更新
date:       2022-02-06
author:     hdbdjdbm
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - 机器学习
---

## 为什么随机梯度下降方法能够收敛?

(挖一个坑，之后填补)

1.随机梯度的期望是梯度本身，而随机梯度法的收敛性分析也多数是在期望意义下研究的。

2.随机梯度下降为了确保收敛，相比于同等条件下的梯度下降，需要采用更小的步长和更多的迭代轮数。

以目标函数强凸+光滑为例。为了达到的精度，梯度下降用 $O(1)$ 的学习率就可以确保收敛，而随机梯度需要用 $O(t^{-1})$ (t为迭代轮数)的学习率才能收敛。这个区别带来的后果是前者只要 $O(log(\epsilon^{-1}))$ 轮即可收敛，而后者需要 $O(\epsilon^{-1})$ 轮才可以。在一些较弱的假设下也有类似结果。

[1] Y. Nesterov,  Introductory Lectures on Convex Programming

[2] A. Rakhlin, O. Shamir, K. Sridharan, Making Gradient Descent Optimal for Strongly Convex Stochastic Optimization

[3] Stochastic Variance Reduction for Nonconvex Optimization

[4] ROBUST STOCHASTIC APPROXIMATION APPROACH TO STOCHASTIC PROGRAMMING




## 如何理解学习率

学习率(Learning rate)决定着目标函数能否收敛到局部最小值以及何时收敛到最小值.

当学习率设置的过小时，收敛过程将变得十分缓慢。而当学习率设置的过大时，梯度可能会在最小值附近来回震荡，甚至可能无法收敛.

注意cs231n里提到一般取log之后cross-validation.

## 为什么SGD的learning rate要逐渐减小，而一般的梯度下降可以固定

如果不使用逐渐减小的步长，最终的结果肯定是发散

## 如何理解正则率

正则化的主要作用是防止过拟合，对模型添加正则化项可以限制模型的复杂度，使得模型在复杂度和性能达到平衡,添加正则化相当于参数的解空间添加了约束，限制了模型的复杂度.

$\lambda$ 过大容易造成欠拟合, 过小就过拟合

cs231n里提到过
* L2正则化可以直观理解为它对于大数值的权重向量进行严厉惩罚，倾向于更加分散的权重向量。由于输入和权重之间的乘法操作，这样就有了一个优良的特性：使网络更倾向于使用所有输入特征，而不是严重依赖输入特征中某些小部分特征。 L2惩罚倾向于更小更分散的权重向量，这就会鼓励分类器最终将所有维度上的特征都用起来，而不是强烈依赖其中少数几个维度。这样做可以提高模型的泛化能力，降低过拟合的风险。
* L1正则化则会让权重向量在最优化的过程中变得稀疏（即非常接近0）。也就是说，使用L1正则化的神经元最后使用的是它们最重要的输入数据的稀疏子集，同时对于噪音输入则几乎是不变的了。相较L1正则化，L2正则化中的权重向量大多是分散的小数字。
* 在实践中，如果不是特别关注某些明确的特征选择，一般说来L2正则化都会比L1正则化效果好

[参考](https://blog.csdn.net/liuweiyuxiang/article/details/99984288)
