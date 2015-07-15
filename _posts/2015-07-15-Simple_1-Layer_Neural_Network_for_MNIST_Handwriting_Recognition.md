---
layout: post
title: Simple 1-Layer Neural Network for MNIST Handwriting Recognition
---

In this post I'll explore how to use a very simple 1-layer neural network to recognize the handwritten digits in the MNIST image files.

![_config.yml]({{ site.baseurl }}/images/mnist-1lnn-logo.jpg)

In my previous blog post I gave a brief introduction [what neural networks are and how they work](../What_is_a_Neural_Network/).
Everything was theory, so let's apply this know-how by writing some code.

Our first challenge will be image recognition. 
For a computer an image is just a collection of pixels with different colors.
So, whatever is on them is very hard for a computer to identify.

State-of-the-art neural networks nowadays are already able identify faces, or prescribe the content of a photo.
We'll start, much much simpler, with recognizing handwritten digits stored as images.
(A possible use case for this is automatically recognizing handwritten ZIP codes in the mail.)

The *Gold-standard* in this area is called [MNIST](http://yann.lecun.com/exdb/mnist/), maintained by one of the leading experts in machine learning, [Yann Lecun](http://yann.lecun.com).
It holds 60,000 standardized images of handwritten digits to *train* our neural network (*training set*), and another 10,000 to *test* it (*testing set*).

```
mnist-1lnn/data/train-images-idx3-ubyte
mnist-1lnn/data/t10k-images-idx3-ubyte
```

Since we're using a *supervised learning* method we must know the *correct* content of each image. 
For the developer's convenience these are provided in the accompanying *label* files:

```
mnist-1lnn/data/train-labels-idx1-ubyte
mnist-1lnn/data/t10k-labels-idx1-ubyte
```

Each MNIST image has a size of 28*28 pixels or a total of 784 pixels. 
Each pixel is a number between 0-255 indicating its density which, however, we'll ignore.
To keep things simple, we'll regard each pixel in an image either as ON (1) or OFF (0).
That means we neither consider colors nor stroke strength. 

![_config.yml]({{ site.baseurl }}/images/mnist-image.svg)


