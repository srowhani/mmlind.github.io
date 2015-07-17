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


## The MNIST Database

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

To model a MNIST image in code I use an object (`struct`) called `MNIST_Image` containing the 28*28 pixels:

```c
typedef struct MNIST_Image MNIST_Image;	

struct MNIST_Image{
    uint8_t pixel[28*28];
};
```

For the *MNIST label* a simple 8-bit integer will do:

```c
typedef uint8_t MNIST_Label;
```

<sidenote> When opening the files you first and only once need to read each file's header in order to move the read pointer to the position of the first image.
The *file header* contains information such as number of images in the file and the respective width and height of each image.

```c
struct MNIST_ImageFileHeader{
    uint32_t magicNumber;
    uint32_t maxImages;
    uint32_t imgWidth;
    uint32_t imgHeight;
};

struct MNIST_LabelFileHeader{
    uint32_t magicNumber;
    uint32_t maxLabels;
};
```
 
Since the content of the headers is not critical for our network's function I don't go into further details here. 
Yet, I briefly want to highlight that the values (maxImages, imgWidth,imgHeight, etc.) in the header are stored in reversed byte order and therefore need to be reversed back in order to use them.
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

For recognizing a MNIST image a slightly bigger *perceptron* is needed, one with 724 inputs (one for each of the 28 * 28 pixel) and each input connection has a [0-1] weight.


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

The *output* of a neural network is the network's *classification* of the input. 
When designing a neural network you want your output expressed a value between 0 and 1.
Thus, for our problem of recognizing handwritten digits, instead of defining a single output with a value 0-9, we design a vector of 10 outputs.
Each output value is either 0 or 1. 
The _index_ of the one output that is switched ON ("1") represents the network's *classification*.

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

### The 1-Layer MNIST Neural Network's Design

Adding all the above pieces together we'll end up with a network design like this:

![_config.yml]({{ site.baseurl }}/images/1lnn_full.svg)

Once we've finished designing our network structure we can (almost) start *training* the network by feeding the MNIST images into it.


## Initialize the Network

Before we start training our network we need to reset or initialize all values in the layer. 
All 3 components of a `cell` (*perceptron*) need to be reset:

```
1. Set all inputs to 0
2. Set all weights to a random value 0-1
3. Set output to 0 
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

The network is trained by looping through all 60,000 images and letting the network *classify* each image's digit.
Its classification is then compared with the correct answer (given in the label file) and cell weights are adjusted according to the difference between the two (the *error*).

The training algorithm looks like this:	

```
1. Load a MNIST image and its corresponding label from the database
  
2. Define the target output vector for this specific label
  
3. Loop through all 10 cells in the layer and:
   1. Set the cell's inputs according to the MNIST image pixels
   2. Calculate the cell's output by summing all weighted inputs
   3. Calculate the difference between actual and desired output
   4. Update the cell's weights based on this difference (the error)
  
4. Get the layer's classification using the index of the highest output 
  
5. Define the network's success rate by comparing classification and label
  
6. Go to (1) for processing the next image
```

Now, let's go though the code for each of the steps above:


### Load MNIST image and label

To load a MNIST image and its corresponding label we need to call two simple functions

```c
	MNIST_Image img = getImage(imageFile);
	MNIST_Label lbl = getLabel(labelFile);
```

which can be implemented like this:

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

Next we want to define an output vector of the type 

`
{0,0,0,0,0,0,1,0,0,0}
` 

for the particular *label* that we just loaded from the database.
The function we need looks like this:

```c
Vector targetOutput;
targetOutput = getTargetOutput(lbl);
```

and is implemented like this:

```c
Vector getTargetOutput(int lbl){
    Vector v;
    for (int i=0; i<10; i++){
        v.val[i] = (i==lbl) ? 1 : 0;
    }
    return v;
}
```


### Looping through all cells

While the output vector above defines our *target* or *desired output* we now calculate the network's *actual* output and compare the two.

Each of the 10 cells in our neural network layer represents one of the digits 0-9.
So the output of cell 1 represents, figuratively speaking, the probability that the image that is processed, contains a 1.
And the output of cell 2 gives the probability that this image is actually a 2.
So what we need to do is to calculate all 10 outputs and check which cell's output is the highest.

So we'll loop through all 10 cells and train them on the current image:

```c
	// Layer l
	// Image *img
	// Vector targetOutput
	
	for (int i=0; i < 10; i++){
        trainCell(&l->cell[i], &img, targetOutput.val[i]);
    }

```


## Train each cell

As outlined in the algorithm above our `trainCell` function needs 4 steps: 

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

```
output = sum (input * weight)
```

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

Please note that at the end we need to *normalize* our output, i.e. enforce a value between 0 and 1, to be able to compare it to our target output (which will always be either 0 or 1).


### Calculate the cell's error

Next, the calculated cell output is compared to our desired output. 
Let's say our image shows a "6", then our target output vector would look like this 

```
{0,0,0,0,0,0,1,0,0,0}
```

I.e. the value at the index position 6 (remember, counting starts from 0) is a "1" while all others are "0".
So the *desired* or *target* output is either a "0" for a "1". 

```c
double getCellError(Cell *c, int target){

    double err = target - c->output;

    return err;
}
```


### Update the cell's weights

In *supervised learning* we train the network by letting it know how far off its previous *guess* or *classification* was.
The network then slightly adjusts its weights in order to reduce the *error*, i.e. in order to allow its next *guess* or *classification* to move closer to the *target*.
The size of the incremental change is given to the network in the form of a constant which is normally called the `LEARNING_RATE`. 

```c
void updateCellWeights(Cell *c, double err){
	int LEARNING_RATE = 0.05;
    for (int i=0; i<(28*28); i++){
        c->weight[i] += LEARNING_RATE * c->input[i] * err;
    }
}
```


## Get the Network's Classification

After we looped through and trained all 10 cells on the current image we can get the network's *guess* or *classification* by comparing the output values of all 10 cells.
In the code I decided to use the term *prediction* rather than *classification*. 
The corresponding function

```c
int predictedNum = getLayerPrediction(l);
```

can be implemented like this:

```c
int getLayerPrediction(Layer *l){
    
    double maxOut = 0;
    int maxInd = 0;
    
    for (int i=0; i<10; i++){
        
        if (l->cell[i].output > maxOut){
            maxOut = l->cell[i].output;
            maxInd = i;
        }
    }
    
    return maxInd; 
}
```

What we're doing here is simply returning the index of the cell with the highest output.
The logic behind is simple: since the correct answer, provided via the target output vector, is always represented as a "1", therefore the closer the output value is to "1" the more likely this cell models the right answer.


## Calculate Success Rate

The network's success rate is defined as the ratio of correct answers to the total number of attempts. 
For before we'll move on to train the network on the next image we update an error counter to keep track of how many digits "*we missed*" (classified incorrectly).

```c
if (predictedNum!=lbl) errCount++;
```

After running all of the 60,000 images we can calculate the layer's success rate which, for training and using my example code, is around 83%.

```c
// NUMBER_OF_IMAGES = 60000 for training or 10000 for testing
successRate = errCount / NUMBER_OF_IMAGES * 100;
```

Done. That was pretty much it. Only thing remaining is testing.


## Test the Network

After the network has been *trained* on 60,000 images we'll *test* it using another 10,000 images.
The testing process is exactly the same as the training process, the only difference being we switch off *learning*, i.e. we do NOT update the weights but maintain their values deducted from training.

![_config.yml]({{ site.baseurl }}/images/mnist-1lnn-screenshot.png)

Our simple 1-layer neural network's success rate in the testing set is 85%.
This value is embarrassingly low when comparing it to state of the art networks achieving a success rate of up to 99.97%. 
Given the simple algorithm of this exercise, however, this is no surprise and close to the 88% achieved by Yann Lecun using a similar 1-layer network approach.


---

## Code & Documentation

You can find all the code for this exercise on my [Github project page](https://github.com/mmlind/mnist-1lnn/), including [code documentation](https://rawgit.com/mmlind/mnist-1lnn/master/doc/html/index.html).

When I run it on my 2010 MacBook Pro it takes about 19 seconds to process all 70,000 images.
And the only reason why it is so *slow* is that I'm rendering each image (using "." and "X") in the console while processing.
Once I switch that off the program runs less than 10 seconds ... which is why I love C. :-)

Happy Hacking!

![_config.yml]({{ site.baseurl }}/images/mnist_numbers.png)

