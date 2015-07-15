---
layout: post
title: Simple 1-Layer Neural Network for MNIST Handwriting Recognition
---

In this post I'll explore how to use a very simple 1-layer neural network to recognize the handwritten digits in the MNIST image files.
If you want to jump straight to the code, you can find the [code documentation here](https://rawgit.com/mmlind/mnist-1lnn/master/doc/html/index.html) and the [Github project page here](https://github.com/mmlind/mnist-1lnn/).

![_config.yml]({{ site.baseurl }}/images/mnist-1lnn-logo.jpg)

In my previous blog post I gave a brief introduction [what neural networks are and how they work](../What_is_a_Neural_Network/).
Everything was theory, so let's apply this know-how by writing some code.

Our first challenge will be image recognition. 
For a computer an image is just a collection of pixels with different colors.
So, whatever is on them is very hard for a computer to identify.

State-of-the-art neural networks nowadays are already able identify faces, or prescribe the content of a photo.
We'll start, much much simpler, with recognizing handwritten digits stored as images.
(A possible use case for this is automatically recognizing handwritten ZIP codes in the mail.)

## MNIST

The *Gold-standard* in this area is called [MNIST](http://yann.lecun.com/exdb/mnist/), maintained by one of the nowadays most-cited experts in machine learning, [Yann Lecun](http://yann.lecun.com).
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

![_config.yml]({{ site.baseurl }}/images/mnist-image.svg)

To keep things simple, we'll regard each pixel in an image either as ON (1) or OFF (0).
That means we neither consider colors nor stroke strength. 

## Design the Neural Network

Next, we need to consider how to design our neural network. 
Since it will only have 1 layer, in addition to its output layer, this will be pretty straight forward.

Let's start at the end, the output layer. 
When designing a neural network you normally want your output expressed as values between 0 and 1.
Thus, for our problem of recognizing handwritten digits, instead of defining only one output with a value 0-9, we design a vector of 10 output values where each value is 0 except that of the target number.
So, a target value of "0" would be expressed as

```
{1,0,0,0,0,0,0,0,0,0}
```

a target value of "1" would be expressed as 

```
{0,1,0,0,0,0,0,0,0,0}
```

and a target value of "9" would be expressed as 

```
{0,0,0,0,0,0,0,0,0,9}
```

Since each image has 28 * 28 pixels we design out input layer as 724 cells where each cell has 10 forward connections, one connection to each cell in the output vector.
This gives us a total of 28 * 28 * 10 connections, and a network structure as follows:

![_config.yml]({{ site.baseurl }}/images/1lnn.svg)

Enough theoretical preparation, let's start coding.

## Start Coding

First, we create an object (or a `struct`in C) called `MNIST_Image` containing the 28*28 pixels which we read from the MNIST file.

```
struct MNIST_Image{
    uint8_t pixel[28*28];
};
```


![_config.yml]({{ site.baseurl }}/images/mnist_numbers.png)







