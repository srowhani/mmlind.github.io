---
layout: post
title: Simple 1-Layer Neural Network for MNIST Handwriting Recognition
---

In this post I'll explore how to use a very simple 1-layer neural network to recognize the handwritten digits in the MNIST image files.

![_config.yml]({{ site.baseurl }}/images/mnist-1lnn-logo.jpg)

In my previous blog post I gave a brief introduction [what neural networks are and how they work](../What_is_a_Neural_Network/).
In this post, I will apply these ideas to the problem of image recognition and write the code for a simple 1-layer neural network recognizing images of handwritten digits.

## Image Recognition

For a computer an image is just a collection of pixels with different colors.
Whatever is actually in the picture is very very hard for a computer to identify.

Yet, state-of-the-art neural networks are already able to identify faces, or prescribe the content of a photo.
In this post, I'll do something much much simpler, which is recognizing handwritten digits in images.
(A common practical use case for this problem is the automatic classification of handwritten ZIP codes in the mail.)

## The MNIST Database

The *Gold-standard* in machine learning for handwritten digits is called the [MNIST database](http://yann.lecun.com/exdb/mnist/), maintained by one of the most-cited experts in machine learning, [Yann Lecun](http://yann.lecun.com), who is also leading the machine learning endeavours of Facebook.
The MNIST database contains 70,000 standardized images of handwritten digits. 
It consists of 4 files:

```
(1) A training set of 60,000 images:
*mnist-1lnn/data/train-images-idx3-ubyte*

(2) The labels (correct answers) for the training set:
*mnist-1lnn/data/train-labels-idx1-ubyte*

(3) A testing set of 10,000 images:
*mnist-1lnn/data/t10k-images-idx3-ubyte*

(4) The labels (correct answers) for the testing set:
*mnist-1lnn/data/t10k-labels-idx1-ubyte*
```

The idea is to *train* the neural network first using the *training set*, and then to switch off training and *test* the effectiveness of the trained network using the *testing set*.

Using prior known correct answers to train a network is called *supervised learning* which is exactly what we're doing in this excercise.

![_config.yml]({{ site.baseurl }}/images/mnist-image.svg)

Each MNIST image has a size of 28*28 pixels or a total of 784 pixels. 
Each pixel is a number between 0-255 indicating its density which, however, we'll ignore.
To keep things simple, I treat each pixel in an image either as ON (1) or OFF (0).
That means I don't consider colors and stroke strength. 

The code representing a MNIST image is simply an object (`struct`) called `MNIST_Image` containing the 28*28 pixels:

```
typedef struct MNIST_Image MNIST_Image;

struct MNIST_Image{
    uint8_t pixel[28*28];
};
```

For the MNIST label we don't use a `struct` but simply an 8 bit integer and call it `MNIST_Label`:

```
typedef uint8_t MNIST_Label;
```

When opening the files you first and only once need to read each file's header in order to move the read pointer to the position of the first image.
The *file header* contains information such as number of images in the file and the respective width and height of each image. 
Since the content of the headers is not critical for our network's function I don't go into further details here. 
Yet, I briefly want to highlight that the numbers in the header are stored in reversed byte order and therefore need to be reversed back in order to use them.
For more details on this you can check the [MNIST homepage](http://yann.lecun.com/exdb/mnist/) or [my project code](https://github.com/mmlind/mnist-1lnn/). 

## Design the Neural Network

Enough preparation, let's start coding the neural network.

### Pixel Input

As outlined above each MNIST image has 28*28 pixels and each pixel is represented as either a "1" (ON/BLACK) or a "0" (OFF/WHITE).
The 28*28 matrix is converted into a simple 784 one dimensional input vector:

![_config.yml]({{ site.baseurl }}/images/1lnn_input.svg)

### The Neural Network Cell

In my last blog post I explained what the basic unit of a neural network, the *perceptron* or in our code `cell` looks like.
A *perceptron* is a cell

![_config.yml]({{ site.baseurl }}/images/perceptron.svg)

For recognizing a MNIST image each cell (node) needs to be linked to 724 pixels and each link (=connection) has a [0-1] weight.
Each of the 724 input values is either 0 or 1.


![_config.yml]({{ site.baseurl }}/images/1lnn_nnlayer.svg)

The corresponding code looks like this:

```
struct Cell{
    double input [28*28];
    double weight[28*28];
    double output;
};
```

In order to be able to recognize 10 different digits [0-9] we need to define 10 such cells. 
These 10 cells form the *intelligent* network layer in our 1-layer neural network.

```
struct Layer{
    Cell cell[10];
};
```

Since each image has 28 * 28 pixels we design our input layer as 724 cells where each cell has 10 forward connections, one connection to each cell in the output vector.
This gives us a total of 28 * 28 * 10 connections, and a network structure as follows:

![_config.yml]({{ site.baseurl }}/images/1lnn.svg)


## The Target Output Vector

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
{0,0,0,0,0,0,0,0,0,1}
```

The code for this looks like this:

```
struct Vector{
    int val[10];
};

Vector targetOutput;

```



### Initialize Neural Network Layer

At start we need to reset or initialize our network layer by doing 3 things for each cell:

```
1. Set all cell inputs to 0
2. Set all cell weights to a random value 0-1
3. Set the cell output to 0 
```

The respective code can look like this:

```
void initLayer(Layer *l){
    
    for (int o=0; o<10; o++){
        
        for (int i=0; i<(28*28); i++){
            l->cell[o].input[i]=0;
            l->cell[o].weight[i]=rand()/(double)(RAND_MAX);
        }
        
        l->cell[o].output = 0;
    }
}
```

## Feed image into the network

Next we loop through all 60,000 images and do the following for each image:

```
(1) Load the MNIST image and its corresponding label from the MNIST files
(2) Define a target output vector based on correct label
(3) Loop through all cells in the layer
    (a) Set the cell's input according to the MNIST image pixels
    (b) Calculate the cell's output by summing all weighted inputs
    (c) Update the cell's weights based on the comparison with target
(4) Get the layer's prediction
```

![_config.yml]({{ site.baseurl }}/images/1lnn_input.svg)
![_config.yml]({{ site.baseurl }}/images/1lnn_nnlayer.svg)
![_config.yml]({{ site.baseurl }}/images/1lnn.svg)

![_config.yml]({{ site.baseurl }}/images/1lnn_full.svg)

### Load MNIST image and label

```
	MNIST_Image img = getImage(imageFile);
	MNIST_Label lbl = getLabel(labelFile);

```

### Define target output vector

```
    Vector targetOutput;
    targetOutput = getTargetOutput(lbl);
```

### Loop through all cells

```
	for (int i=0; i < 10; i++){
        trainCell(&l->cell[i], &img, targetOutput.val[i]);
    }

```

### Train each cell

```
void trainCell(Cell *c, MNIST_Image *img, int target){
    
    setCellInput(c, img);
    calcCellOutput(c);
    
    double err = getCellError(c, target);
    updateCellWeights(c, err);
}
```

### Set a cell's Input

```
void setCellInput(Cell *c, MNIST_Image *img){
    
    for (int i=0; i<(28*28); i++){
        c->input[i] = img->pixel[i] ? 1 : 0;
    }
}
```

### Calculate the cell's actual output

```
void calcCellOutput(Cell *c){
    
    c->output=0;
    
    for (int i=0; i<(28*28); i++){
        c->output += c->input[i] * c->weight[i];
    }
    
    c->output /= (28*28);             // normalize output (0-1)
}
```

### Calculate a cell's error

```
double getCellError(Cell *c, int target){

    double err = target - c->output;

    return err;
}
```

### Update cell's weights

```
void updateCellWeights(Cell *c, double err){
    
    for (int i=0; i<(28*28); i++){
        c->weight[i] += LEARNING_RATE * c->input[i] * err;
    }
}
```

You can find all the fully running code for the above excercise on the [Github project page](https://github.com/mmlind/mnist-1lnn/), including [code documentation](https://rawgit.com/mmlind/mnist-1lnn/master/doc/html/index.html).

![_config.yml]({{ site.baseurl }}/images/mnist_numbers.png)







