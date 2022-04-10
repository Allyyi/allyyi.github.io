---
# Required front matter
layout: post # Posts should use the post layout
title: Transformer in Computer Vision Ⅱ # Post title
date: 2021-07-15 # Publish date in YYYY-MM-DD format

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
listed: false # false if this post must NOT be included on the posts page, sitemap, and any of the tag pages, default is true
index: true # When false, <meta name="robots" content="noindex"> is added to the page, default is true
---

## Towards more data-efficient

### Data-efficient image Transformer(DeiT)

The training of vanilla ViT took a large and private dataset JFT-300(contains 300 million images), and do not generalize well when trained on insufficient amount of data. Under this situation, the researchers have proposed a new training strategy based on **distillation** in [Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2012.12877v2.pdf).

## Towards lighter

### DeiT

In the paper [*Training data-efficient image transformers & distillation through attention*](https://arxiv.org/pdf/2012.12877v2.pdf), the author 

## Swin Transformer

