---
layout:     post
title:      "Causal Relations"
subtitle:   " \"论文杂谈\""
date:       2022-04-10 12:00:00
author:     "Yanyan"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - papers
---

## Investigating Causal Relations by Econometric Models and Cross-spectral Methods(通过计量经济学模型和交叉谱方法研究因果关系)

### 论文背景

#### 作者
C.W.J.Granger
>2003年诺贝尔经济学奖得主, 把经济学变为可以定量研究的学科

#### 提出问题
* 相关不是因果，那能否把相关关系(类似正反馈过程)拆解为因果关系（因为A所以B）
* 问题：已知两个随机过程的自回归形式，如何判断两个变量是否为因果关系。

#### 解决思路
1. 使用SPECTRAL METHODS，将随机变量的相关关系 --> 随机变量的相关关系功率谱(傅里叶变化, 时域 -> 频域, 波动 -> 正弦波的叠加，频率上的表示 -> 线性 -> 自回归模型)
2. 查看Cross-spectral(随机变量的相关关系 -> 反馈过程 -> 因果关系)


#### 主要内容


##### 定义
* SPECTRAL METHODS (谱方法)
* FEEDBACK MODELS (反馈)
* CAUSALITY(因果)

##### 实例
* TWO-VARIABLE MODELS (两个变量)
* THREE-VARIABLES MODELS (三个变量)

#### 主要结论
* 在进行随机过程中的谱表示之后，进行的反馈机制可以分解为两个因果关系，其交叉谱可以看作是两个交叉谱之和，且其中每部分都与反馈过程的一个单向因果相关。
 * 瞬时因果关系(?)


### 论文detail

#### 定义
**(1) spectral reresentation**  
$X_t$ is a stationary time series with mean zero, there are two basic spectral reresentation：

(i) Cramer representation
$$X_t = \int_{-\pi}^\pi e^{it\omega}dz_x(\omega)$$

 where $$Z_x(\omega)$$ is a complex random process with uncorrelated increments.

(ii) the spectral reprentationof the covariance sequence

$$\mu_\tau^{xx} = E[X_t\bar{X}_{t - \tau}] = \int_{-\pi}^\pi e^{it\omega}dF_x(\omega)$$

**(2) 交叉谱(Cross-spectral)**   
(有点没理解)

the cross spectrum $$Cr_(\omega)$$ between $$X_t$$ and $$Y_t$$ ia a complex fuction of $\omega$ and arises both from 

$$E[dz_x(\omega)\overline{dz_y(\omega)}] = 0 = Cr(\omega)d\omega$$ 

and

$$\mu_\tau^{xy} =E[X_t\overline{Y_{t-\tau}}] =  \int_{-\pi}^\pi e^{it\omega}Cr(\omega)d\omega$$

**(3) the coherence and the phase**

the coherence:

$$C(\omega) = \frac{|Cr(\omega)|^2}{f_x(\omega)f_y(\omega)}$$  

the phase:

$$\phi(\omega) = tan^{-1}\frac{Imaginary Part Of Cr(\omega)}{Real Part Of Cr(\omega)}$$

measures the phase difference between corresponding frequency components


**(4)  causul model** 
自回归模型

causal model
$$A_0X_t = \sum_{j=1}^m A_jX_{t-j} + \epsilon_t$$

当前时刻的行为是以往的线性组合

simple causal models：$A_0$不存在

Instantaneous Causality(瞬时因果关系)  
$A_0$存在，当前这一秒决定自身

two varible case:
$$X_t + b_0Y_t= \sum_{j=1}^m a_jX_{t-j} + \sum_{j=1}^m b_jY_{t-j} + \epsilon_t^{'}$$

$$Y_t + c_0X_t= \sum_{j=1}^m c_jX_{t-j} + \sum_{j=1}^m d_jY_{t-j} + \epsilon_t^{''}$$

**(5)  Causality(因果)**  
$$\sigma^2(X|U) < \sigma^2(X|\overline{U - Y})$$

除去Y后对X进行预测/计算，误差增大，那就说Y是X的一个因


**(6)  Feedback**  
if
$$\sigma^2(X|\bar{U}) < \sigma^2(X|\overline{U - Y})$$
$$\sigma^2(Y|\bar{U}) < \sigma^2(Y|\overline{U - X})$$

we say that feedback is occurring.
>feedback is said to occur when $X_t$ is causing $Y_t$ and also $Y_t$ is causing $X_t$


<!-- **(7)  Causality Lag**   -->

#### TWO-VARIABLE MODELS

对两个变量的causul model进行Cramer representation以及变量替换。


$$X_t = \sum_{j=1}^m a_jX_{t-j} + \sum_{j=1}^m b_jY_{t-j} + \epsilon_t$$

$$Y_t = \sum_{j=1}^m c_jX_{t-j} + \sum_{j=1}^m d_jY_{t-j} + \eta_t$$

In terms of time shift operator $U-UX_t = X_{t-1}$

rewrite the equations:
$$X_t =  a(U)X_t + b(U)Y_t + \epsilon_t$$

$$Y_t = c(U)X_t + d(U)Y_t + \eta_t$$

where x(U)s are power series in U i.e.,$$a(U) = \sum_{j = 1}^{m}$$

Using the Cramer representation of the series, i.e.,
$$X_t = \int_{-\pi}^\pi e^{it\omega}dZ_x(\omega)$$

$$a(U)X_t = \int_{-\pi}^{\pi}e^{it\omega}a(e^{-i\omega})dZ_x(\omega)$$
the equation can be written:

$$\int_{-\pi}^{\pi}e^{it\omega}[(1-a(e^{-i\omega}))dZ_x(\omega) - b(e^{-i\omega}dZ_y(\omega) - dZ_\epsilon)(\omega)] = 0$$
Same for y

the cross spectrum 
$$Cr(\omega) = \frac{1}{2\pi\triangle}[(1-d)\overline{x}\sigma_\epsilon^2 + (1-\overline{a}b\sigma_\eta^2)]$$
can be written as the sum of two components  
$$Cr(\omega)=C_1(\omega) + C_2(\omega)$$

where 
$$C_1(\omega) = \frac{\sigma_\epsilon^2}{2\pi\triangle}(1-d)\bar{c}$$

and 
$$C_2(\omega) = \frac{\sigma_\eta^2}{2\pi\triangle}(1-\bar{a})b$$

$$\triangle = |(1-a)(1-d) - bc|^2$$

如果$Y_t$没有导致$X_t$，那么 b = 0, $C_2(\omega)$ vanishes, vice versa.

>So the cross spectrum can be decomposed into the sum of two components - one which depends on the causality of X  by Y and the other on the causality of Y by X.

#### consider a special case:
$$X_t = bY_{t-1} + \epsilon_t$$
$$Y_t = cX_{t-2} + \eta_t$$


### To-do List

- 手动推导一下二变量情况
- 定义的细节部分rewrite
- 时间序列的自回归模型(VAR)