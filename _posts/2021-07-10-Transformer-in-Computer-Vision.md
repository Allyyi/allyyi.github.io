---
# Required front matter
layout: post # Posts should use the post layout
title: Transformer in Computer Vision Ⅰ # Post title
date: 2021-07-10 # Publish date in YYYY-MM-DD format

# Recommended front matter
tags: CV tag4 # A list of tags
splash_img_source: 
splash_img_caption:

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
The transformer archietecture has achieved state of the art in many NLP tasks and inspired many researchers to adopt it in computer vision. Nowadays, the attention-based transformer module is challenging the dominating position of convolution. In this post, I will share some of my understanding in transformer and its progresses made in computer vision. 

## What is Transformer

Attention mechanism address the problem of forgetting in long seqeunce by assigning weight between Quest and Key, which I have introduced in this [post](https://allyyi.github.io/yixin.github.io//2021/05/12/Attention,-please!.html). However, it still used as an supplimentary part in recursive neural network architecture. In recursive model, the state i can not be generated until all previous states are done. The inability of parralel computing brings huge inefficiency. An intuitive idea is that, since the attention needs the entire seqeunce to be computed, can we build a model that purely rely on the attention mechanism so we can aggregate global information as well as do parralel computing? The answer is yes! And that is the reason why the Transformer is borned for. Now, let's take a closer look at Transformer.

The model architecture is an encoder-decoder structure, using a stack of self-attention and point-wise fully connected layers in each block.

<img src="../../../assets/img/img_for_transformer/image-20210718201102009.png" alt="image-20210718201102009" style="zoom:80%;" />

**Encoder:**

> - Composed of N=6 identical layers
> - Each layer has two sub-layer: Multi-head attention layer and feed-forward layer. 
> - Residual connection is applied around each sub-layer, followed by layer normalization. That is, the output of each sub-layer is $$\textrm{Layernorm}(x+\textrm{Sublayer}(x))$$
> - The output dimension $$d_{model}=512$$

**Decoder:**

> - Composed of N=6 identical layers
> - Each layer has three sub-layers: Masked multi-head attention layer in addition to previous two sub-layers.
> - Residual connection and layernorm to each sub-layer.

Here are some noticable details:

- **Masked self attention**: The output is generated in an auto-regressive manner, so we need to mask out the places that haven't been generated so far.
- **Positional encoding**: The attention itself doesn't preserve any positional information. In transformer, there's no reccursive or convolution, so we need to add an additive vector to represent the sequential order.
- **Scaled-dot self-attention**: When dimension of query and keys $$d_k$$ is large, the dot products of $$QK^T$$ grow large in magnitude, pushing softmax to regions where gradient is extremely small. Scale it by $$1/\sqrt{d_k}$$ can relieve the problem of gradient vanishing.

Sefl-attention based Tranformer model generally operates in two-stage learning mechanism.

- Pre-training on a large-scale dataset in supervised or unsupervised manner.
- Adapt pre-trained weights to downstream tasks using small datasets.

Since manually acquring labels for large datasets is costive, self-supervised learning has been very effectively used pre-trained stage.

## Vision Transformer

### Motivation

There has also been a lot of interest in combining CNNs with self-attention mechanism by augmenting feature maps for image classification or by further processing the output of CNNs. In another line of work, local multi-head self-attention blocks are modified to completely replace convolution. Many of these work demonstrate promising results in computer vision, but are trained on the standard ImageNet dataset and cannot be scaled efficiently on modern hardware accelerators. In large-scale image recognition, classic ResNet-like architecture are still the mainstream. Can image recognition get rid of dependency on convolution? The ViT gives its answer.

### Model architecture

The Vision Transformer(ViT) is proposed in [*An Image is worth 16x16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/pdf/2010.11929.pdf). Just as the paper's title implies, the ViT treated an image as sequential input like sentences in NLP task. The image is first splitted to fixed-sized patches and the patches are flattened, projected to a fixed latent vector dimension D and then fed to the Transformer along with their own positional embeddings. The ViT only use a pure transformer encoder. Its architecture is showned as below.![image-20211120151425565](../../../assets/img/img_for_transformer/image-20211120151425565.png)

Similar to BERT's [class] token, a learnable embedding is prepended to the sequence of embedded patches($$z_0^0 = x_{class}$$) and its state at the output($$z_L^0$$) serves as image representation. $$z_L^0$$, after a layer norm operation, is then fed to MLP head, which contains one hidden layer at pre-training stage and a single linear at fine tuning stage.

Different from the self-supervised manner in NLP, the ViT uses supervised manner in pre-training stage. The authors tried self-supervised method in ViT, even though the result is better than training from scratch, it's still lower than supervised manner. Self-supervised method in ViT's pretraining is worthy of further research.

### Comparison between ViT and CNN

<img src="../../../assets/img/img_for_transformer/image-20211121124412957.png" alt="image-20211121124412957" style="zoom:50%;" />

Here is a plot of ImageNet Top1 accuracy with different pre-training dataset for different model. BiT(Big Transfer) performs supervised transfer learning with large ResNets. From plot, we can see the CNN-based method BiT is outperforming all sizes of ViT model on a relatively small ImageNet dataset. However, as the pretraining datasets grows larger, the ViT model overtakes. Why this happens? Inductive bias is a possible reason.

**Inductive bias**

For convolution, locality, 2D neighborhood structure and translation equivariance are intergrated in each layer.

For ViT, in each block, only MLP has locality and translation equivariance, and self-attention is able to extract global information while all spatial relations between patches need to be learnt from scratch.

From analysis above, we can see that ViT has much less image-specific inductive bias than CNNs, which means, in smaller datasets, the image-specific inductive bias can help to reduce the parameter searching space so that the model is able to converge faster and achieve higher accuracy than less-inductive-bias model. However, when it comes to large dataset, the less-inductive-bias model has less constrained parameter searching space, which bringd a more powerful representative ability thus a better results.

**Similarity**

Apart from the difference in inductive bias, I think there is still something common between CNN and ViT. In CNN, a convolution layer usually has multiple channels, aiming to capture different features for one image/feature map in each channel. In ViT, the multi-head attention attend to one input with different head in their own learnt sub-spaces. Both of them are trying to capture different features from various viewpoint.

### Conclusions

ViT's contribution includes but not limited to:

- Reliance on convolution in image task is not necessary.
- Large scale training data trumps inductive bias.
- It provides a possible way to combine multi-modality data training with a uniform model.

However, ViT also has the following drawbacks:

- It requires large corpus data to achieve a competative reuslt.
- It has too much parameters. Parameters size of a ViT-Base is 86M and 623M on ViT huge.
- Attention is computed along patches and theres no information interaction within patches.

There's still many unsolved problem in Transformer for Vision, and many researchers dive deeper in various directions to unravel the mystries of ViT. In next part, I will cover some interesting follow-up works.

## Towards deeper

### DeepViT

Recall the development CNN based neural network, from AlexNet -> VGG -> ResNet, depth scaling is deemed to be an effective way to improve the model performance. So it's a very natural idea to make ViT model deeper. However, in the paper [*DeepViT: Towards deeper Vision Transformer*](https://arxiv.org/abs/2103.11886), the author finds that simply make the network deeper by adding more Transformer blocks is not working as the following figure shows.

<img src="../../../assets/img/img_for_transformer/image-20211121152403196.png" alt="image-20211121152403196" style="zoom:67%;" />

 What might be the reason? Before we jump to the reason, let me introduce a similarity measurement of attention map. $$M^{p,q}$$ is the cosine similarity matrix between the attention map of layer p and q(layer is equals expression of block). Each element $$M^{p,q}_{h,t}$$ measures the similarity of attention for head h and token t. When  $$M^{p,q}_{h,t}$$ equals to one it means token t plays excatly the same role for self-attention in layer p and q.

$$M^{p,q}_{h,t}=\frac{ {A_{h,:,t}^p}^T {A_{h,:,t}^q} }{||{A_{h,:,t}^p}^T|| ||A_{h,:,t}^q||}$$

By investigating cross layer attention, we can find that as the network goes deeper, the attention map similarity ratio across layers becomes higher and this phenomenon is named as **attention collapse**. In the deeper blocks, the attention maps are highly similar with each other which means the features they learned are pretty much the same  and they barely learn something new under the basis of shallower layers. You can find the evidence in figure (a) and (b). However, in figure (c), the similarity of different heads within the same blocks are lower than 30%, indicating sufficient diversity.

![image-20211121154226429](../../../assets/img/img_for_transformer/image-20211121154226429.png)

So, how can we counteract the effect of attention collapse?

- **Increase dimension**. Higher dimension can holds richer information, but the parameter scale is large so we also face higher risk of overfitting as well as higher computation cost.

- **Re-attention**. 

  Based on the observation in Fig. (c), the author proposes to establish cross-head communication to re-generate the attention maps.

  $$\mathrm{\mbox{Re-Attention}(Q,K,V) = Norm(\Theta^T(Softmax(\frac{QK^T}{\sqrt(d)})))V}$$

  where $$\Theta \in R^{H\times H}$$ and end-to-end learnable. From the following plot, we can find that the feature map similarity between blocks drops signicantly in deeper blocks. This trick is very flexible and the performance gains significantly for ViT-32B.

  <center><img src="../../../assets/img/img_for_transformer/image-20211121193024993.png" alt="image-20211121193024993" style="zoom: 60%;" /></center>
