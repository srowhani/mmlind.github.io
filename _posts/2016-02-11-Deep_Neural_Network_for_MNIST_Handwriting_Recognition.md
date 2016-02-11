---
layout: post
title: Deep Neural Network for MNIST Handwriting Recognition
---

I finally got my hands dirty in the recent hype around deep learning by building my own deep neural network. It can consist of any variable number of layers and now also supports convolutional layers which empirically perform best at image recognition problems. Its architecture is kept generic, extendable and tries to mimic its biological parent, the brain. 

![_config.yml]({{ site.baseurl }}/images/dnn_mnist-logo.png)

In a previous blog post I wrote about a simple [3-Layer neural network for MNIST handwriting recognition](../Simple_3-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/) that I built. It's architecture was fixed, always using a 3-layer structure with exactly 1 hidden layer. And it only supported normal, fully connected feed forward layers. 

To achieve better results in image recognition tasks such as MNIST deeper networks are needed. 
And they need to be capable of running convolutional layers.
Hence, as a next step on my journey towards *coding AI wisdom* I set out to add these two features to my C network. 

So, let me introduce to you in more detail how the network works, starting off with its overall architecture.

![_config.yml]({{ site.baseurl }}/images/dnn_convolutional_net.png)

## Network Architecture

The introduction of convolutional layers heavily influenced the design of the network. 

### Convolutional Networks

Convolutional layers are a very different beast than normal fully connected layers. 
Instead of each node in a layer connecting to all nodes in the previous layer, it connects to only some of them.
The selection of which nodes it connects to is defined by a quadratic filter (or kernel window) that is moved over the previous layer.
Thereby, the geolocation of a specific node, i.e. the horizontal and vertical proximity to its neighbors, is taken into account which is crucial for image recognition.

The second major difference of convolutional layers is the fact that weights are shared between nodes. 
This siginificantly reduces the network's complexity, i.e. number of weights or parameters that need to be trained and memory requirements.

This post assumes that you understand the basic functionality of a convolutional network. 
If you don't I strongly recommend first reading Stanford's [CS231n Convolutional Neural Networks for Visual Recognition
](http://cs231n.github.io/convolutional-networks/) by Andrej Karpathy or, more hardcore, Yann Lecun's [LeNet-5 Convolutional Neural Networks](http://yann.lecun.com/exdb/lenet/).

So, let's jump into the overall design of the network. 

### Data Model

Previously, my network structure consisted of layers, nodes and weights. Now, I added 2 additional concepts: *columns* and *connections*. Let me explain why.

![_config.yml]({{ site.baseurl }}/images/dnn_network_struct_design.png)

### Columns

In a layer of nodes, these nodes are aligned flat in a 1-dimensional vector. 
A MNIST image, for example, has 28 x 28 pixels and the respective network layer thus consists of 784 nodes that are all aligned in a single row.

In image recognition, though, the exact location of a pixel inside that image matters. Convolutional networks or layers therefore build connections to neighboring nodes that are all situated within a defined region, the so called *filter* or *kernel window*.

Example: A convolutional network node may want to create connections to a 3 x 3 area (*filter*) of 9 neighboring nodes (pixels) located at the top left of the image. If all pixels of the image are numbered from 0..783 than this area can be easily calculated as [0,1,2,28,29,30,56,57,58].
In the respect, we essentially treat the 1-dimensional vector as 2-dimensional using simple algebra. 

So far, so good. The critical point, though, that isrequiring a design change is that images are not 2-dimensional but 3-dimensional. 
What? Yes. Convolutional networks treat their input, be it an incoming image or the output from a previous layer, as 3-dimensional.

For the input image, the 3rd dimension is color. In a simple RGB image, each pixel is actually composed of 3 values: a red value, a green value and a blue value. Hence, a 3rd dimension is required which in the case of an RGB image is 3 levels deep.

Adding the 3rd dimension of network nodes into the general network architecture, I faced 2 design options: feature maps or columns.

Intuitively, I first wanted to slice the image (or the input of a previous layer for that matter) __horizontally__ which creates what the convolutional network theory sometimes refers to as *feature maps*. So in the example of an RGB image, instead of having a single 1-dimensional [0..783] vector of 784 nodes, you'd have 3 *feature maps* and each would consist of 784 nodes. 

Alternatively, if you slice the image (or the input of a previous layer for that matter) __vertically__ you end up with *columns*. 
The network layer has 784 columns with each column consisting of 3 nodes.

Conceptually, both designs are equivally acceptable and feasible to implement. 
The fact that I ended up chosing the latter may have been related to my previous study of Hierarchical Temporal Memory (HTM) theory where columns are a intrinsic element. 
This design overall also more closely resembles the design of the brain.

```c
struct Column{
    ByteSize size;              // actual memory size of this column in run-time
    ByteSize nodeSize;          // actual memory size of this column's nodes in run-time
    int maxConnCountPerNode;    // maximal amount of connections per node
    int nodeCount;              // number of nodes in this column
    Node nodes[];               // a variably-sized array of nodes
};
```


### Connections

The 2nd major design change is the introduction of connections. 
If you remember, in my previous network weights were attached directly to a node because each node had exactly 1 weight.
And the *target* node to which a certain node is connectd to could be easily derived from the weight's ID, i.e. the 1st weight of a each hidden layer node is applied to the 1st input layer node, the 2nd weight of each hidden layer node is applied to the 2nd input layer node, and so forth. That was simple.

For convolutional networks, things get more complicated. First, weights are shared, i.e. the same weight will be used from multiple nodes. 
Second, each node in the hidden layer is linked to very different nodes in the previous layer. 
Hence, we need to keep track of which *target nodes* (= nodes in the previous layer) a hidden layer node is connected to.

Again, looking at biology, I introduced the concept of connections (similar to *synapses*) into the model.
A connection is a simple structure storing 2 pointers: a pointer to a target node and a pointer to a certain weight.

![_config.yml]({{ site.baseurl }}/images/network_struct_design.png)

### Network Definition

To build a network one has to first define a model, i.e. how many layers it has, how may nodes inside each layer, etc.
To do so I introduced a new structure called *LayerDefinition* which holds the key parameters for design a network layer: 

layerType

node structure

activation Function

One can create a variable number of these layer definitions and then pass them together into a function which returns an array pointer for easier reference throught the rest of the code.

Once the layers are defined, you can createe the network object by simply calling

The system will automatically allocate the required memory for this type of network and return a pointer to reference the network.

###Network Initialization

Now, what actually happens inside this *createNetwork* function? It does mainly 2 things, intializing connections and setting random weights.

###Connection Wiring

Initializing connections means creating such connections between the layers, columns and nodes. 


### Shared Weights


## Network Execution


### Network Performance


### Hyper-parameter Optimization




number of layers variable (before only number nodes)


convolutional layer

 introduce connections, separate weights from nodes

 introduce columns (between layer and nodes)



initialise by looping not by copying

same logic: input, feed, learn/backprop, classify

new activation function

remarks: use simple max output for classification (not a xxxx function)














## Conclusion



---

## Code & Documentation

You can find all the code for this exercise on my [Github project page](https://github.com/mmlind/mnist-3lnn/), including full [code documentation](https://rawgit.com/mmlind/mnist-3lnn/master/doc/html/index.html).

When I run it on my 2010 MacBook Pro, using 784 input nodes, 20 hidden nodes and 10 output nodes, it takes about 19 seconds to process all 70,000 images (with the image rendering turned-off) and achieves an accuracy on the *testing set* of 91.5%.

Happy Hacking!

![_config.yml]({{ site.baseurl }}/images/mnist_logo.png)

