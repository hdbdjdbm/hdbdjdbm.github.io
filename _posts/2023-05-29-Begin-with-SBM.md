---
layout:     post
title:      "Begin with SBM"
subtitle:   " \"a survey of SBM\""
date:       2023-05-29 12:00:00
author:     "Yanyan"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - papers
---

![frmaework](https://pic.imgdb.cn/item/64757982f024cca1730ba0d5.jpg)

As seen in the picture, this survey(need many todo) is divided into four parts. The first part is the introduction to SBM and some extension models. The second part is the random fluctuation to community detection and why we need Bayesian block modeling. The third part is about model selection, telling us how we compare different models. The fourth part talks about the inference methods for SBM.


**Q1**: How does entropy make sense in community detection? 

**Come from**: in Bayesian stochastic block modeling[1], all the deductions are under the framework of maximum entropy. But does it always work? Cause when we choose the prior we still change the distribution of it. In real situation, community has ‘people make’, it should not be uninformation. 

**Remedy**: We can turn to the statistics physics area. How's it going well in statistical physics(cause the whole idea in statistics is entropy)? Especially with some restrictions. Refer to related paper[2], is there any way to quantify this?


**Q2**:How information theory(especially minimum descriptive length) helps to explain the detection community? 

**Come from**: Inspired by neural network of machine learning, information theory help to explain the information that every layer. Can we regard community detection as a dynamic process, using information theory in model selection and look at if there is any good way to do it faster? 

**Other questions**: Can the minimum descriptive length method be a criterion? Can it evaluate biases? ( some papers have mentioned, have more reading to support this idea)[3],[4]  

### framework

A complex network is usually seen in the real world, the website network, social network, etc. The most popular topic in complex networks is community detection, cause we always want to know the latent structure in the complex network, therefore we can understand our network better. Generally, there are two ways to do community detection, descriptive method and using generative models like SBM. Let's first talk about SBM.

the basic process is 

(1)build the stochastic blockmodel 

(2)Inference the parameters to fit the data 

(3)evaluate the model using some criteria

  
However, in application to real data, neither the group memberships nor the block matrix is observed or given. Subsequently, some problems arise.


(1) can SBM describe a real-world network? Take a social network as a simple example, for a class of people, every node in the same group is similar? Does one student only belongs to one group? Should this student also belong to other groups? Or let's back a step, can a student belongs to a group probabilistically? How about the resolution problem?  

(2) why do we need Bayesian? When we talk about stochastic we also have the randomness in our mind. What we want is to know which node belongs to which group, we neither want to let the network be too 'complex' nor want the network to be too 'simple'(unless they are). So here come two questions, how should we prevent randomness, and how we evaluate the biaes?

(3) how to say what a 'good' model is, [5]discuss the three criteria that can help us to compare different models, however, it still has a gap between the descriptive model and the generative model. [4] proposed a way to compare them on the same level.


(4) do the inference. Here the most frequent way is the MentoCarlo Markov chain and another is variance inference.



## stochastic block models and extension

As we mentioned before, can SBM describe a real-world network? Take a social network as a simple example, for a class of people, every node in the same group is similar? This deduce DC-SBM, to describe a 'dominant sheep' phenomenon. And does one student only belongs to one group? Should this student also belong to other groups? This deduces the overlapping SBM. Or let's back a step, can a student belongs to a group probabilistically? This is called mixed membership SBM. Then let's look at this social network from a bigger view, in the school network, can SBM distinguish all groups? How about the resolution problem? (when our reality community number is large but we only fit a few communities). The remedy to this problem is using a hierarchy model named nested SBM.[6]


SBM is a stochastic block model for short. Stochastic means **Stochastic equivalence**, which describes two nodes in one group having the same probability of connecting to another group. Block means when we generate the network, we already know the group member and group number.

  

Firstly, we denote some symbols. 

Consider a graph $G = (\mathcal{N},\mathcal{E})$,where $\mathcal{N}$ is the node set of size $n := |\mathcal{N}|$ and $\mathcal{E}$ is the edge list of size $M := |\mathcal{E}|$. Dyad is a pair of nodes. $Y$ denote the $n$ x $n$ adjacency matrix. If node $p$ and node $q$ have an edge between them, then $Y_{pq} = Y_{qp} = 1$. $K$ means the number of groups on the network. $Z_p$ is a $K$-vector of that all elements are 0 except exactly one takes the value 1 that represents the group mode $p$ belongs to($K$ x 1 vector). $Z:=(Z_1 Z_2 ... Z_n)^T$ ia a $n$ x $K$ matrix. $Z_{pi}$ indicates wether the nodes $p$ belong to $i$ group or not. $N$ is the $K$ x 1 vector that $N_i$ indicates the number of nodes in $i$ group. $E$ is a $K$ x $K$ matrix, $E_{ij}$ represents the number of edges between group $i$ and group $j$. $C$ is a $K$ x $K$ matrix and $C_{ij}$ represents the probability of occurrence of an edge between a node in group $i$ and a node in group $j$.

  

$Y_{pq}$ follows the with success probability $Z_p^TCZ_q$. The total number of edges between any two blocks $i$ and $j$ is a Binomial distributed random variable with a mean equal to the product of $C_{ij}$ and the number of dyads available(in binomial distribution the expectation is $np$).

  

Now we want to construct our model, do not forget our purpose: we want to have the adjacency matrix, so we need to know two inputs： first one is which nodes belong to which groups and the probability of the edges.

  

So given $Z$ and $C$, the likelihood can be written as 

$$ 

\pi(Y|Z,C) = \prod_{p<q}^n \pi(Y_{pq}|Z,C) 

= \prod_{p<q}^{n}[(Z_p^TCZ_q)^{Y_{pq}}(1-Z_p^TCZ_q)^{(1-Y_{pq})}] \tag{1} 

$$

  

If $G$ is directed, (question, a review of use $N_iN_j/2$, I think is $N_iN_j$)

  

In a real-world situation, usually, Z and C are unknown, so assumptions have to be made before modeling and inference. We assume that $K$ x 1 vector, $\theta = (\theta_1 ... \theta_k)$ and $\sum_{i=1}^K\theta_i = 1$, and $Pr(Z_{pi} = 1) = \theta_i$. Then the latent group $Z_p$ follows the multinomial distribution with probabilities $\theta$, which means

  

$$ 

\pi(Z|\theta) = \prod_{p=1}^nZ_p^T\theta = \prod_{p=1}^n\theta^TZ_p = \prod_{i=1}^K\theta_i^{N_i}\tag{2} 

$$

  

Here we need to notice that K is the number of the group.

Poisson and degree-corrected graph

Let's consider this situation: in the limit of a large sparse graph where the edge probability equals the expected number of edges, this version of the SBM, called the Poisson SBM, is asymptotically equivalent to the Bernoulli counterpart.

  

Karrer and Newman (2011) also worked with (undirected) valued graphs, but arguably in a more natural way. they redefined $Y_{pq}$ as the number of edges for the dyad $(p,q)$ following a Poisson distribution, $C_{ij}$ the expected number of edges form a node in group i and a node in group j.

  

The density of $Y_{pq}$ is

  

$$ 

\pi(Y_{pq}|Z,C) = (Y_{pq}!)^{-1}exp(-Z_p^TCZ_q)(Z_p^TCZ_q)^{Y_{pq}} \tag{3} 

$$ 

here, we put the Poisson distribution: 

$$ 

P(X=k)=\frac{λ^k}{k!}e^{−λ},k=0,1,⋯ 

$$ 

we can see that the expected value of $\lambda$ is $Z_pCZ_q$ 

  

But this kind of model still has a problem, is any node in the same community similar? Is there any dominant node in there? So based on the Poisson SBM, we introduce a parameter $\phi_p$ 

constrained to $\sum_{p=1}^n \phi_p 1{Z_{pi}= 1} = 1$ for every group $i$, so that the expected number of edges for the dyad $(p, q)$ is now $\phi_p\phi_qZ_p^TCZ_q$ . The density of $Y_{pq}$ becomes

  

$$ 

\pi(Y_{pq}|Z,C,\phi) = (Y_{pq}!)^{-1}exp(-\phi_p\phi_qZ_p^TCZ_q)(\phi_p\phi_qZ_p^TCZ_q)^{Y_{pq}}\tag{4} 

$$ 

The parameters $\phi_p$ and $C_{ij}$ have natural interpretations as their maximum likelihood estimates (MLEs) are the ratio of $p$’s degree to the sum of degrees in $p$’s group, and the total number of edges between groups $i$ and $j$, respectively.

  

Now let's go back to equation (2), we assume that $\theta$ arises from Dirichlet($(α1K)$ distribution, of which the parameter $\alpha$ comes from a $Gamma(a,b)$ prior.

  

$$ 

\pi(Y,Z|\alpha,\beta,\gamma) = \pi(Y|Z,\beta,\gamma) \times \pi(Z,\alpha) 

$$ 

$$ 

= \int\pi(Y|Z,C)\pi(C|\beta,\gamma)dC \times \int\pi(Z|\theta)\pi(\theta|\gamma)d\theta 

$$ 

### DC-SBM

DC-SBM should not be seen as an indication of the inferiority of the original SBM, as it captures different underlying structures that follow stochastic equivalence.

### Microcanonical SBM

It can be derived from modifying the Poisson SBM, we can see from (), for $1 ≤ i ≤ j ≤ K$ and conditional on $Z$ and an extra parameter $λ$, $C_{ij}$ is assumed to follow the Exponential distribution with the rate $N_iN_j/λ$. This assumption replaces that according to (4). By doing so, $C$ can be integrated out from the product of (8) and the exponential density of C.


The microcanonical is because of the hard constraints imposed, as $Y$ and $Z$ together fix the value of $E$

### Mixed membership SBM

In the mixed membership stochastic block model (MMSBM) by Airoldi et al. (2008), for each node $p$, the latent variable $Z_p$, which contains exactly one 1, is replaced by a membership vector, also of length $K$, denoted by $θ_p$. The elements of $θ_p$, which represent weights or probabilities in the groups, have to be non-negative and sum to 1, and each node can belong to different groups when interacting with different nodes.

### Overlapping SBM

But in real-world networks, each overlapping node typically belongs to 2 or 3 groups. Furthermore, the proportion of overlapping nodes is relatively small for real-world networks, usually less than 30%.

### resolution problem and nested SBM

When we deciding the number of communities, we will face the resolution problem. The resolution problem is when our real community is large but we only fit a few communities, [xxx] proposes the maximum community is $\sqrt{N}$ when we fit the data. The remedy to this problem is using a hierarchy model named the nested stochastic block model.

  

## Bayesian stochastic blockmodel


Why do we need to use Bayesian as a tool? Let's first look into an example. 

When we generate an Erd˝os-R´enyi random graph where each possible edge can occur with the same probability, we find that if we change the order of the nodes, we can find the different blocks of the random graph, "a random graph can contain a wealth of structure before it becomes astronomically large — especially if we search for it".[4] the remedy to it is to think probabilistically, statistical evidence available in the data, and also use the prior information. Here we connect entropy and information theory to construct the model and deal with the process.


Picture 1[]

  

As we mentioned above, we already have the forward path, the question is, how we fit our model into data. On the one hand, we can use the likelihood we deduce, using the EM algorithm to get the parameter. But the problem is, how to avoid the fluctuation we mentioned above, and how to use statistical evidence to think probabilistically. Naturally, we think of the Bayesian formula: 

$$ 

P(b|A) = \frac{P(A|b)P(b)}{P(A)} 

$$

  

In a Bayesian situation, $b$ indicates the partition of the network, and $A$ indicates the graph.

  

$P(A|b)$ is the generating probability of the SBM, $P(b)$ can be seen as the prior of the partition. $P(A) = \sum_b P(A|b)P(b)$ is the evidence, and $p(b|A)$ called the posterior. So if we want to combine the statistical evidence, we need to solve these three equations one by one.

### The equation of $P(A|b)$

$P(A|b)$ is what we called likelihood, we also deduce it in the first section. But here, we want to deduce it from the entropy aspect.

### The equation of $P(b)$

We want to combine our prior information to deal with this system, but the question is, how should we decide the prior? The safest solution is not to put any bias on it, which in information theory is called 'uninformation'. Here, we want to introduce the maximum entropy model, it deduce the distribution when we only know the some strained.  

## model selection and criteria

After we modify our model, we face 2 questions. The first one is, how we decided our model between model and model B. The second question is how we decided the community number k in our final model.

Here, community detection is our background when we compare the different methods. As we mention before, I categorize the model into three classes, (1)inferential (2)descriptive (3) data majority model. (use SBM generating model and machine learning method to handle it).

Let's first dive into the first class: inferential

### Bayesian model selection

let's start with a simple example of rolling the coin, suppose we throw the coin $n = 200$ times, in which we have $x = 155$ times head.

  

Suppose we have two models:

  

$M_1$: the probability coin head is $0.5$ 

$M_2$: the probability coin head is $\theta \in (0,1)$ , the the prior of $\theta$ is uniform distribution $Pr(\theta|M_2) = Uniform[0,1]$

  

Which model should you select?

  

From the frequency aspect, we may choose the second one, and use the maximum likelihood we will get the parameter $\bar{\theta} = \frac{115}{200}$

  

but from the Bayesian aspect, 

$$ 

Pr(M_i|D) = \frac{Pr(D|M_i)Pr(M_i)}{\sum_iPr(D|M_i)Pr(M_i)} 

$$

  

$Pr(M_i)$ is prior distribution of model, $Pr(D|M_i)$ is the evidence. 

In our example, we have assumed a uniform prior of models, $Pr(M_1) = Pr(M_2) = 0.5$, let's compare the posterior of two models:

  

$$ 

\frac{Pr(M_1|D)}{Pr(M_2|D)} = \frac{Pr(D|M_1)Pr(M_1)}{Pr(D|M_2)Pr(M_2)} 

$$ 

Where 

$$ 

Pr(D|M_1) = Pr(x,n,0.5) = Binomail(200,115,0.5) = 0.00595 

$$ 

And 

$$ 

\begin{aligned} 

Pr(D|M_2) = Pr(x|n,M_2) = \int Pr(x|n,\theta,M_2)Pr(\theta|M_2)d\theta =\int_0^1 \tbinom{200}{115}\theta^{115}(1-\theta)^{200-115} = 0.0049 

\end{aligned} 

$$ 

So the 

$$ 

\frac{Pr(M_1|D)}{Pr(M_2|D)} = 1.2 

$$ 

This means under our Bayesian selection frame we prefer the $M_1$ model.The reason is Bayesian model selection prefers an easier model than the complex model, and it's consistent(一致性) with Occam's razor.

In our situation,$M_1$ can only explain the dataset which half head, however, $M_2$ can explain almost every kind of dataset, but its trade-off is for a dataset that has half head, the prior will decrease. But in our case, the prior we assume the prior is the same.

In[5], the author deduces the formula of the Bayesian model selection method. Here we introduce it:

### Bayesian information criteria

In the Bayesian model selection, every time we can only compare the two models. we want to derive the formula that when we face different models we can get the value and compare them. Then we get the Bayesian information criteria.
  

$$ 

BIC(M_i) = -2\ln P(Y|M_i,\hat\prod_i) + |\prod_i|\ln(n) 

$$

where $|\prod_i|$ is the degree of freedom of the model $M_i$ with a parameter set $|\prod_i|$ , and $n$ is number of i.i.d. samples in the data.

### minimum description length  

--todo

## inferential method

We mainly have two ways to do the inference. The first one is sampling from the data, and the second is the do the variance inference.

### MCMC
--todo
### variance inference 
--todo
### nonbayesian parameters method
--todo