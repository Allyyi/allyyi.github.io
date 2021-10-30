---
# Required front matter
layout: post # Posts should use the post layout
title: Attention, please! # Post title
date: 2021-05-12 # Publish date in YYYY-MM-DD format

# Recommended front matter
tags: NLP # A list of tags
splash_img_source: # Splash image source, high resolution images with an aspect ratio close to 4:3 recommended
splash_img_caption: # Splash image caption

# Optional front matter
updated: # Updated date in YYYY-MM-DD format
author: Xin Yi
name: Some Guest Author # Author name, if not provided defaults to site.author.name
homepage: # Author link, if not provided defaults to site.author.homepage
pin: false # true if this post must be pinned on top of the page, default is false.
listed: true # false if this post must NOT be included on the posts page, sitemap, and any of the tag pages, default is true
index: true # When false, <meta name="robots" content="noindex"> is added to the page, default is true
---

- toc
{:toc}

The attention model, initially born for the Machine Translation problem, is inspired by the human biological system. When we look at a picture or read a sentence, we tend to focus on the key parts that best represent the main information. For example, given a sentence like ”I love wearing blue jeans” when we encounter the word “wearing”, we expect to see cloth for the next, so the relevance between “wearing” and “jeans” is high while relevance between “wearing” and “blue” is low. This relevance capturing ability is done by paying attention to certain parts of the input that helps to perform the task. This seemingly simple idea has been applied to Natual Language Processing, Computer Vision, and Graph Neural Network and has become an essential part of network architecture.

# What is Attention Model

## Why attention

Before attention, most machine translation models are composed of an encoder(reads and compresses a source sentence into a fixed-length context vector) and a decoder(outputs translation with decoded vector). This fixed-length context vector fails in remembering long sentences because the beginning part has been forgotten by the time the whole sentence is compressed.

![image-20210714211413935](../../../assets/img/img_for_attention/image-20210714211413935.png)

<center>Fig1: Encoder-Decoder model of Seq2Seq</center>

However, in attention model, the context vector is a weighted sum over all hidden layer of the encoder, which tackle the problem of forgetting. The encoder is a bidirectional RNN while the decoder is an RNN, taking the previous hidden layer and dynamic context vector as input.

## Definition

![image-20210714212955056](../../../assets/img/img_for_attention/image-20210714212955056.png)

<center>Fig2: Attention Model</center>

The probability of each target word $$y_i$$ is 

$$p(y_i|y_1,...,y_{i-1},\mathbf x) = g(y_{i-1},s_i,c_i)$$

where $$s_i$$ is the hidden state for decoder's RNN in time i

$$s_i = f(s_{i-1},y_{i-1},c_i)$$

The context vecotor $$c_i$$ is computed as a weighted sum of annotations  $$h_j$$

$$c_i=\sum_{j=1}^{T}\alpha_{ij}h_j$$

The annotation $$h_j$$ consists of concatenation of forward hidden state and backward hidden state, which helps the model capture preceding and following words in context.

$$h_j = [\overrightarrow {h^T_j}; \overleftarrow {h^T_j}]^T$$

The weight of each annotation $$\alpha_{ij}$$ is computed by

$$a_{ij}=\frac{exp(e_{ij})}{\sum_{k=1}^{T} exp(e_{ik})}$$

where

$$e_{ij}=a(s_{i-1},h_j)$$

$$\alpha_{ij}$$ , or its associated energy $$e_{ij}$$  (the alignment score) reflects the importance of annotation $$h_j$$ with respect to the previous hidden state $$s_{i-1}$$ in deciding the next state $$s_i$$. Since $$y_i$$ is decided by $$s_i$$ and $$h_j$$ is decided by $$x_j$$, $$\alpha_{i,j}$$ reflects how well two words $$y_i$$ and $$x_j$$ align. (Remember the relevence we mentioned above?)

The alignment model $$a$$ is a feed forward neural network with a single hidden layer and jointly trained with other parts of the network. 

$$a(s_{i-1},h_j)=v^T_a \tanh(W_{a}s_{i-1}+ U_ah_j)$$

$$v^T_a, W_a, U_a$$ are learnable parameters.

For more detailed algorithm please refer to [Bahdanau et al.,2015](https://arxiv.org/pdf/1409.0473.pdf)

# The Attention Family

The generalized attention model can be defined as below:

Given a query $$q$$ and a set of key-value pairs $$(K,V)$$, attention can be generalized to compute a weighted sum of values depending on the query and corresponding keys. The query determines which values to focus on by increasing the $$\alpha_{ij}$$. In [Bahdanau et al.,2015](https://arxiv.org/pdf/1409.0473.pdf), key is encoder's hidden layer states $$h_i$$ and query is decoder hidden state $$s_{i-1}$$, and there's no difference between K and V,  $$k_i = h_i = v_i $$. Model  A

$$A(q,K,V)=\sum_ip(a(k_i,q))*v_i$$

$$a$$ is alignment function(or called score function). The design varies in different models.

 $$p$$ is distibution function, mapping alignment function score to attention weights.

## Self-attention and its variation

Denoting a sequence of n values $$(x1, x2, ..., x_n)$$ by $$X \in \mathbb R^{n\times d_x}$$, the purpose of self-attention is capturing the dependency between all n elements after encoding. This process is achieved by the definition of 3 learnable weight matrices: $$W^Q\in \mathbb R^{d_x\times d_q}$$, $$W^K\in \mathbb R^{d_x\times d_k}$$, $$W^V\in \mathbb R^{d_x\times d_v}$$

$$Q=XW^Q, \quad K=XW^K, \quad V=XW^V $$ 

The output in self-attention layer is expressed as follow using **dot attention**

$$Z = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_q}})V,~ Z\in \mathbb R^{n\times d_v}$$

Note that the dimension of query and key has to be the same, which guaranteens $$QK^T$$ operation is valid. The result of softmax is a $$n\times n$$ matrix called **attention map**. In the [long short-term memory network](https://arxiv.org/pdf/1601.06733.pdf) paper, self-attention is applied to learn how much the current word attends to the word in previous part of sentence. 

![image-20210715205600717](../../../assets/img/img_for_attention/image-20210715205600717.png)

<center>Fig 3: Example of self-attention. Bold lines indicate higher attention score. </center>

Self-attention layer applies to every element, however, when we apply self-attention to a decoder to generate the following words, these words are not yet generated so it dosen't make sense to attend to these empty positions. To solve this problem, we mutiply the score matrix by doing element-wise product (Hadamard product) with a mask $$M \in \mathbb R^{n \times n}$$ (lower triangular matrix), the attention map of future elements are set to be 0.

$$\mathrm{Attention}(K,Q,V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_q}} \circ M)V$$

 You can find more details in [Transformer](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). 

## Multi-head attention

Instead of performing  a single attention function to K, Q, V, the multi-head attention learns different linear projections from K, Q, V to $$d_k, d_k, d_v$$ dimensions for h times. $$d_k=d_{model}/h$$. Then the output of attentions are concatenated again and projected to the final value. Each attention is performed in parallel. 

$$\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(head_1,head_2,...,head_n)W^Q$$



where

$$head_i = \mathrm{Attention}(QW^Q_i, KW^K_i,VW^V_i)$$



where $$W_i^Q \in \mathbb R^{d_{model} \times d_k}, W_i^ K\in \mathbb R^{d_{model} \times d_k},W_i^V \in \mathbb R^{d_{model} \times d_V}, W_i^O \in \mathbb R^{d_{hd_v} \times d_{model}}$$

<img src="../../../assets/img/img_for_attention/image-20210715213715113.png" alt="image-20210715213715113" style="zoom:80%;" />

<center>
    Fig4: Multi-head Attention
</center>



Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. Intuitively, you can think of different attention as head paying attention from unique linear view.

## Attention different in positions

### Soft & Hard attention

The attention introduced by [Bahdanau et al.,2015](https://arxiv.org/pdf/1409.0473.pdf) is also known as **soft** attention, as its name implies, it uses a weighted average of all hidden states of the input sequence to build the context vector. Soft attention takes the weighted sum over all positions. It has nice differentiability but is computationally expensive. 

In contrast, [Xu et al. 2015](https://arxiv.org/pdf/1502.03044.pdf) proposed **hard** attention, which selects one position (in the paper, it's a patch of image) to attend to at a time by sampling. This saves computational cost in sacrifice of differentiability. Variational learning methods and policy gradient methods in reinforcement learning are introduced to overcome this problem.

![image-20210716210315324](../../../assets/img/img_for_attention/image-20210716210315324.png)

<center>
    Fig:5 Soft Attention vs Hard Attention
</center>



### Global & Local attention

In [Luong, et al., 2015](https://arxiv.org/pdf/1508.04025.pdf), the author proposed global attention and local attention. Global attention is very similar to soft attention. A context vector is computed as a weighted average over all source states.

<img src="../../../assets/img/img_for_attention/image-20210716134428586.png" alt="image-20210716134428586" style="zoom:70%;" />

<center>
    Fig6: Global Attention
</center>


The global attention has a drawback that's when the input sequence is long(a paragraph or document), it has to attend to all words in the source input for each target word. It's too computationally expensive and sometimes even unnecessary. To address this problem, the author proposed the local attention. It first predicts a single aligned position $p_t$ for the current target word. Then, a window centered around $$p_t$$ is used to compute the context vector. The weights $$a_t$$ are inferred from hidden layers in the windows and current target state.  It's more like a combination of soft and hard attention. It avoids expensive computation and is locally differentiable. Because of the fixed size of window, the alignment vector $$a_t$$ has fixed dimension. Window is defined by $$[p_t-D, p_t+D]$$, then $$a_t \in \mathbb R^{2D+1}$$  .

The position of time t $$p_t$$ can be chosen with two methods below:

- **Monotonic** aligment(local-m): simply assume $$p_t = t$$.

- **Predictive** aligment(local-p): S is the length of source input, $$v_p, W_p$$ are learnable parameters.  

   $$p_t = S\cdot \mathrm{sigmoid}(v_p^Ttanh(W_ph_t))$$

The aligment score is defined as follows instead of softmax.

$$a_t(s)=\mathrm{align}(h_t,\overline h_s)exp(-\frac{(s-p_t)^2}{2\sigma^2})$$

<img src="../../../assets/img/img_for_attention/image-20210716143723839.png" alt="image-20210716143723839" style="zoom:60%;" />

<center>
    Fig7: Local Attention
</center>


