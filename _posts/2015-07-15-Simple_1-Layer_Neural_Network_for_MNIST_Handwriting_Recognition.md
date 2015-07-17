---
layout: post
title: Simple 1-Layer Neural Network for MNIST Handwriting Recognition
---

In this post I'll explore how to use a very simple 1-layer neural network to recognize the handwritten digits in the MNIST image files.

![_config.yml]({{ site.baseurl }}/images/mnist-1lnn-logo.jpg)

In my previous blog post I gave a brief introduction [what neural networks are and how they work](../What_is_a_Neural_Network/).
In this post I will apply these ideas to the problem of image recognition and write the code for a simple 1-layer neural network recognizing images of handwritten digits.

## Image Recognition

For the computer an image is just a collection of pixels with different colors.
Whatever is actually in the picture is very very hard for a computer to identify.

Yet, state-of-the-art neural networks are already able to identify faces, or prescribe the content of a photo.
In this post, I show how this can be done in principal, but using something much simpler to start with: recognizing handwritten digits in images.
(A common practical use case for this problem is the automatic classification of handwritten ZIP codes in the mail.)

### The MNIST Database

The *Gold-standard* in machine learning for handwritten digits is called the [MNIST database](http://yann.lecun.com/exdb/mnist/), maintained by one of the most-cited experts in machine learning, [Yann Lecun](http://yann.lecun.com), who is also leading the machine learning endeavours of Facebook.
The MNIST database contains 70,000 standardized images of handwritten digits and consists of 4 files:

(1) A training set of 60,000 images:

```
mnist-1lnn/data/train-images-idx3-ubyte
```

(2) The labels (correct answers) for the training set:

```
mnist-1lnn/data/train-labels-idx1-ubyte
```

(3) A testing set of 10,000 images:

```
mnist-1lnn/data/t10k-images-idx3-ubyte
```

(4) The labels (correct answers) for the testing set:

```
mnist-1lnn/data/t10k-labels-idx1-ubyte
```

The idea is to *train* the neural network first using the *training set*, and then to switch off training and *test* the effectiveness of the trained network using the *testing set*.

Using prior known correct answers to train a network is called *supervised learning* which is exactly what we're doing in this excercise.

Each MNIST image has a size of 28*28 pixels or a total of 784 pixels. 
Each pixel is a number between 0-255 indicating its density which, however, we'll ignore.
To keep things simple, I treat each pixel in an image either as ON (1) or OFF (0).
That means I don't consider colors and stroke strength. 

![_config.yml]({{ site.baseurl }}/images/mnist-image.svg)

To model a MNIST image in code I simply use an object (`struct`) called `MNIST_Image` containing the 28*28 pixels:

```c
typedef struct MNIST_Image MNIST_Image;	

struct MNIST_Image{
    uint8_t pixel[28*28];
};
```

For the *MNIST label* aa simple 8-bit integer will do:

```c
typedef uint8_t MNIST_Label;
```

<sidenote> When opening the files you first and only once need to read each file's header in order to move the read pointer to the position of the first image.
The *file header* contains information such as number of images in the file and the respective width and height of each image. 
Since the content of the headers is not critical for our network's function I don't go into further details here. 
Yet, I briefly want to highlight that the numbers in the header are stored in reversed byte order and therefore need to be reversed back in order to use them.
For more details on this you can check the [MNIST homepage](http://yann.lecun.com/exdb/mnist/) or [my project code](https://github.com/mmlind/mnist-1lnn/). </sidenote>

## Designing the Neural Network

Before we start coding the neural network, we need to consider its design. 
I.e. every neural network's structure is likely somewhat different, always trying to best suit the particular problem to be solved.

### The Input 

As outlined above each MNIST image has 28x28 pixels and each pixel is represented as either a "1" (ON/BLACK) or a "0" (OFF/WHITE).
The 28x28 matrix is converted into a simple 784 one dimensional input vector containg 0s and 1s:

![_config.yml]({{ site.baseurl }}/images/1lnn_input.svg)

### The Core 

In [my last blog post](../What_is_a_Neural_Network/) I explained what the basic unit of a neural network, the *perceptron*, looks like.
It consists of a node with at least 2 inputs and 2 weights, and one output value.

![_config.yml]({{ site.baseurl }}/images/perceptron.svg)

For recognizing a MNIST image a slightly bigger *perceptron* is needed, one with 724 inputs (one for each of the 28 * 28 pixel) and each link (=connection) has a [0-1] weight.


![_config.yml]({{ site.baseurl }}/images/1lnn_nnlayer.svg)


In our code the *perceptron* is modeled as a `cell`:

```c
struct Cell{
    double input [28*28];
    double weight[28*28];
    double output;
};
```

### The Network Layer

A neural *network* normally consists of more than just 1 cell. 
In order to be able to recognize 10 different handwritten digits [0-9] we need to define 10 such cells. 
Combined these 10 cells form the *intelligent* network layer in our 1-layer neural network.

![_config.yml]({{ site.baseurl }}/images/1lnn.svg)

Using 10 cells with 28 * 28 input values (0,1) and 28 * 28 respective weights (0-1) the network layer can be defined as follows:


```c
struct Layer{
    Cell cell[10];
};
```

### The Output

The *output* of a neural network is the value we expect it to *predict*. 
When designing a neural network you want your output expressed a value between 0 and 1.
Thus, for our problem of recognizing handwritten digits, instead of defining a single output with a value 0-9, we design a vector of 10 outputs.
Each output value is either 0 or 1. 
The _index_ of the one output that is switched ON ("1") represents the network's *prediction*.

Example: If the MNIST image contains the digit "1" this "1" becomes the network's *target value* and is modeled as an output vector of

```
{0,1,0,0,0,0,0,0,0,0}
```

(Remember, the *index* in coding always starts at 0 not at 1.) A target value of "9" would be modeled as an output vector of	

```
{0,0,0,0,0,0,0,0,0,1}
```

The code for modeling the output vector is:

```c
struct Vector{
    int val[10];
};

```

Adding all the above pieces together we'll end up with a network design like this:

![_config.yml]({{ site.baseurl }}/images/1lnn_full.svg)

Once we've finished designing our network structure we can (almost) start *training* the network by feeding the MNIST images into it.


## Initialize Neural Network Layer

Before we start training our network we need to reset or initialize our network layer. 
In particular, all 3 components of a `cell` (*perceptron*) need to be reset:

```
1. Input set to 0
2. Weights set to random value 0-1
3. Output set to 0 
```

The respective code can look like this:

```c
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

## Train the Network

The nextwork is trained by looping through all 60,000 images and letting the network *predict* each image's digit.
Its prediction is then compared with the correct answers (given in the label files) and cell weights are adjusted according to the difference between the two (the *error*).

The training algorithm looks like this:	

```
(1) Load a MNIST image and its corresponding label from the database
(2) Define the target output vector for this specific label
(3) Loop through all 10 cells in the layer and:
    (a) Set the cell's inputs according to the MNIST image pixels
    (b) Calculate the cell's output by summing all weighted inputs
    (c) Calculate the difference between actual and desired output
    (c) Update the cell's weights based on this difference (the error)
(4) Get the layer's prediction using the index of the highest output 
(5) Define the network's success rate by comparing prediction and label
(6) Go to (1) for processing the next image
```

Now, let's go though the code for each of the 4 steps above:

### Load MNIST image and label

To load a MNIST image and its corresponding label we need to call two simple functions

```c
	MNIST_Image img = getImage(imageFile);
	MNIST_Label lbl = getLabel(labelFile);
```

which can be implemented like this

```c
MNIST_Image getImage(FILE *imageFile){
    
    MNIST_Image img;
    size_t result;
    result = fread(&img, sizeof(img), 1, imageFile);
    if (result!=1) {
        printf("\nError when reading IMAGE file! Abort!\n");
        exit(1);
    }
    
    return img;
}

MNIST_Label getLabel(FILE *labelFile){
    
    MNIST_Label lbl;
    size_t result;
    result = fread(&lbl, sizeof(lbl), 1, labelFile);
    if (result!=1) {
        printf("\nError when reading LABEL file! Abort!\n");
        exit(1);
    }
    
    return lbl;
}

```

### Define target output vector

Next we want to define an output vector of the type {0,0,0,0,0,0,1,0,0,0} for the particular *label* that we just loaded from the database.
The function we need looks like this

```c
    Vector targetOutput;
    targetOutput = getTargetOutput(lbl);
```

and is implemented like this

```c
Vector getTargetOutput(int lbl){
    Vector v;
    for (int i=0; i<10; i++){
        v.val[i] = (i==lbl) ? 1 : 0;
    }
    return v;
}
```

### Loop through all cells

While the output vector above defines our *target* our *desired output* we now calculate the network's actual output, which will become its *prediction*, in order to compare the two.
Each of the 10 cells in our neural network layer represents one of the digits 0-9.
So the output of cell 1 represents, figuratively speaking, the probability that the image that is processed, contains a 1.
And the output of cell 2 gives the probability that this image is actually a 2.
So what we need to do is to calculate all 10 outputs and check which cell's output is the highest.

So we'll loop through all 10 cells and train them on this particular image:

```c
	// Layer l
	// Image *img
	// Vector targetOutput
	
	for (int i=0; i < 10; i++){
        trainCell(&l->cell[i], &img, targetOutput.val[i]);
    }

```

## Train each cell

As outlined in the algorithm above our `trainCell` consists of 4 steps: 

```c
void trainCell(Cell *c, MNIST_Image *img, int target){
    
    setCellInput(c, img);
    calcCellOutput(c);
    
    double err = getCellError(c, target);
    updateCellWeights(c, err);
}
```

### Set a cell's Input

First, we need to set the cell's input to match the current image's pixels.
I.e. our cell's input values should be the same 724 long sequence of 0s and 1s as the images's pixels.
For simplicity I'll only consider whether a pixel is ON or OFF, i.e. 0 or 1, and neglect a pixel's density.

```c
void setCellInput(Cell *c, MNIST_Image *img){
    
    for (int i=0; i<(28*28); i++){
        c->input[i] = img->pixel[i] ? 1 : 0;
    }
}
```

### Calculate the cell's actual output

Next, we calculate the cell's output, i.e. its *guess* whether this image represents e.g. a "1".
This is done simply by summing the product of all 724 inputs multiplied by their weights:

`
output = sum (input * weight)
`

The code to do this looks like this

```c
void calcCellOutput(Cell *c){
    
    c->output=0;
    
    for (int i=0; i<(28*28); i++){
        c->output += c->input[i] * c->weight[i];
    }
    
    c->output /= (28*28);             // normalize output (0-1)
}
```

Please note that at the end we need to *normalize* our output, i.e. enforce a value between 0 and 1, to be able to compare and rank our 10 outputs against each other.


### Calculate the cell's error

Next, the calculated cell output is compared to our desired output. 
Let's say our image shows a "6", then our target output vector would look like this `{0,0,0,0,0,0,1,0,0,0}`.
I.e. the value at the index position 6 (remember, counting starts from 0) is a "1" while all others are "0".
So the *desired* or *target* output is either a "0" for a "1". 

```c
double getCellError(Cell *c, int target){

    double err = target - c->output;

    return err;
}
```

### Update the cell's weights

In *supervised learning* we train the network by letting it know how far off its previous *guess* or *prediction* was.
The network then slightly adjusts its weights in order to reduce the *error*, i.e. in order to allow its next *guess* or *prediction* to move close the *target*.
The size of the incremental change is given not the network in the form of a constant which is normally called the `LEARNING_RATE`. 

```c
void updateCellWeights(Cell *c, double err){
	int LEARNING_RATE = 0.05;
    for (int i=0; i<(28*28); i++){
        c->weight[i] += LEARNING_RATE * c->input[i] * err;
    }
}
```





You can find all the fully running code for the above excercise on the [Github project page](https://github.com/mmlind/mnist-1lnn/), including [code documentation](https://rawgit.com/mmlind/mnist-1lnn/master/doc/html/index.html).

![_config.yml]({{ site.baseurl }}/images/mnist_numbers.png)







