---
layout: post
title: Deep Neural Network for MNIST Handwriting Recognition
---

I finally got my hands dirty in the recent hype around deep learning by building my own deep neural network. It can consist of any variable number of layers and now also supports convolutional layers which empirically perform best at image recognition problems. Its architecture is kept generic, extendable and tries to mimic its biological model, the brain. 

![_config.yml]({{ site.baseurl }}/images/dnn_mnist-logo.png)

In a previous blog post I wrote about a simple [3-Layer neural network for MNIST handwriting recognition](../Simple_3-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/) that I built. It's architecture was fixed, always using a 3-layer structure with exactly 1 hidden layer. And it only supported normal fully connected layers. 

To achieve better results in image recognition tasks deeper networks are needed. 
And they need to be capable of running convolutional layers.
Hence, as a next step on my journey towards *coding AI wisdom* I set out to add these two features to my C network. 

So, let me introduce to you in more detail how the network works, starting off with its overall architecture.


## Network Architecture

More than anything else did the introduction of convolutional layers influence the design of the network. 


### Convolutional Networks

Convolutional layers are a different beast than normal fully connected layers. 
Instead of each node in a layer connecting to *all* nodes in the previous layer, it connects to only *some* of them.
The selection of which nodes it connects to is defined by a quadratic filter (or kernel window) that is moved over the previous layer.
Thereby, the geolocation of a specific node, i.e. the horizontal and vertical proximity to its neighbors, is taken into account which is crucial for image recognition.

The second major difference of convolutional layers is the fact that weights are shared between nodes. 
This siginificantly reduces the network's complexity, i.e. the number of weights or parameters that need to be trained, and its memory requirements.

![_config.yml]({{ site.baseurl }}/images/dnn_convolutional_net.png)

This post assumes that you understand the basic functionality of a convolutional network. 
If you don't I strongly recommend first reading Stanford's [CS231n Convolutional Neural Networks for Visual Recognition
](http://cs231n.github.io/convolutional-networks/) by Andrej Karpathy or, more hardcore, Yann Lecun's [LeNet-5 Convolutional Neural Networks](http://yann.lecun.com/exdb/lenet/).

So, now let's jump into the code and let's look how the network's data structures are defined. 


## The Data Model

Previously, my network structure consisted of layers, nodes and weights. Now, I added 2 additional concepts: *columns* and *connections*. 
So in total, the neural network is now based on below 5 structures: 


### The Network

The network structure itself contains the *learning rate* and information about number and location of all *weights* in this network. 
And most importantly this structure serves as a container for all other components of the network.

```c
struct Network{
    ByteSize size;                  // actual byte size of this structure in run-time
    double learningRate;            // factor by which connection weight changes are applied
    int weightCount;                // number of weights in the net's weight block
    Weight *weightsPtr;             // pointer to the start of the network's weights block
    Weight nullWeight;              // memory slot for a weight pointed to by dead connections
    int layerCount;                 // number of layers in the network
    Layer layers[];                 // array of layers (of different sizes)
};
```


### A Network Layer

Inside a network structure there is an array of layers. 
Each layer contains a reference to its own model definition (more on that below), a pointer to its weights and all its columns.

```c
struct Layer{
    int id;                         // index of this layer in the network
    ByteSize size;                  // actual byte size of this structure in run-time
    LayerDefinition *layerDef;      // pointer to the definition of this layer
    Weight *weightsPtr;             // pointer to the weights of this layer
    int columnCount;                // number of columns in this layer
    Column columns[];               // array of columns
};
```


### A Network Column

Previously, the network layer directly contained the network nodes that were all aligned flat in a 1-dimensional vector. 
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
    ByteSize size;              // actual byte size of this structure in run-time
    int maxConnCountPerNode;    // maximum number of connections per node in this layer
    int nodeCount;              // number of nodes in this column
    Node nodes[];               // array of nodes
};
```

### A Network Node

Inside a column there is a number of network nodes. 
In non-convolutional layers the number of nodes per column is always 1. 
In this case the layer has a depth of 1 and the number of columns is the same as the number of nodes.

```c
struct Node{
    ByteSize size;              // actual byte size of this structure in run-time
    Weight bias;                // value of the bias weight of this node
    double output;              // result of activation function applied to this node
    double errorSum;            // result of error back propagation applied to this node
    int backwardConnCount;      // number of connections to the previous layer
    int forwardConnCount;       // number of connections to the following layer
    Connection connections[];   // array of connections
};
```


### A Node's Connections

The 2nd major design change is the introduction of connections. 
If you remember, in my previous network weights were attached directly to a node because each node had exactly 1 weight.
And the *target* node to which a certain node is connectd to could be easily derived from the weight's ID, i.e. the 1st weight of a each hidden layer node is applied to the 1st input layer node, the 2nd weight of each hidden layer node is applied to the 2nd input layer node, and so forth. That was simple.

For convolutional networks, things get more complicated. First, weights are shared, i.e. the same weight will be used from multiple nodes. 
Second, each node in the hidden layer is linked to very different nodes in the previous layer. 
Hence, we need to keep track of which *target nodes* (= nodes in the previous layer) a hidden layer node is connected to.

Again, looking at biology, I introduced the concept of connections (similar to *synapses*) into the model.
A connection is a simple structure storing 2 pointers: a pointer to a target node and a pointer to a certain weight.

```c
struct Connection{
    Node *nodePtr;              // pointer to the target node
    Weight *weightPtr;          // pointer to a weight that is applied to this connection
};
```


### The Weights Block

The most important component of a neural network is still missing in all of the above: the weights. 
The *connection* structure only contains a pointer to a weight.
But where is the actual weight?

The answer is I put all the weights together in a *weights block* which and locate it inside the network structure after the last layer.
This design has several advantages: first, the weights are kept together with the rest of the network inside the same memory block. 
Second, the separation of weights from the nodes allows for convenient weight sharing. 

The following drawing shows how all the above components fit together into a *network* structure.

![_config.yml]({{ site.baseurl }}/images/dnn_network_struct_design.png)


### Network Definition

To build a network one has to first define a model, i.e. define how many layers the network shall have, how may nodes are inside of each layer, etc.
For this reason I introduced a new structure called *LayerDefinition* which holds the key parameters for each layer and will be attached via a pointer to the respective layer structure (see above).

```c
struct LayerDefinition{
    LayerType layerType;        // what kind of layer is this (INP,CONV,FC,OUT)
    ActFctType activationType;  // what activation function is applied
    Volume nodeMap;             // what is the width/height/depth of this layer
    int filter;                 // size of the filter window (conv layers only)
};
```

One can create a variable number of these layer definitions and then pass them together into a function which returns an array pointer for easier reference throught the rest of the code.


### Layer Types

The code currently supports the following *types* of layers:

```c

typedef enum LayerType {INPUT, CONVOLUTIONAL, FULLY_CONNECTED, OUTPUT} LayerType;

```

The idea here is that based on the type of layer certain rules are applied, for example, the connections between nodes are created differently.

Moving forward I want to expand this further to include other layer types, for example Hierarchical Temporal Memory (HTM) layers for sequentiell learning.


### Activation Function

The code currently supports below 3 activation functions:

```c

typedef enum ActFctType {SIGMOID, TANH, RELU} ActFctType;

```

Each layer can define its own activation function which is applied during feed forward (*activation*) as well as during error *back propagation*.
In theory, you can design a network using different activation functions for different layers.
In practive, however, this does not improve the network performance, rather the contrary.

It's also important to note that other parameters, most significantly the  *learning rate* depend on what *activation function* is chosen.
Hence, both of these hyper-parameters should be considered in tandem.


## Create the Network

Once the network and its layers have been defined via the above *LayerDefinition* structures you can create the network object via the *createNetwork* function

```c

Network *nn = createNetwork(numberOfLayers, layerDefs);

```

which automatically allocates the required memory for this network, initializes its internal structures (see below) and returns a pointer for reference.

```c

Network *createNetwork(int layerCount, LayerDefinition *layerDefs){
    
    // Calculate network size
    ByteSize netSize = getNetworkSize(layerCount, layerDefs);
    
    // Allocate memory block for the network
    Network *nn = (Network*)malloc(netSize);
    
    // Set network's default values
    setNetworkDefaults(nn, layerCount, layerDefs, netSize);

    // Initialize the network's layers, nodes, connections and weights
    initNetwork(nn, layerCount, layerDefs);
    
    // Init all weights -- located in the network's weights block after the last layer
    initNetworkWeights(nn);
    
    return nn;
}

```


### Calculate the Network's Memory Needs

First, we need to calculate how much memory this network will need. 
Obviously, this depends on many different parameters, such as number of layers, number of nodes per layer, the size of the filter in a convolutional layer, etc.

```c

ByteSize getNetworkSize(int layerCount, LayerDefinition *layerDefs){

    ByteSize size = sizeof(Network);
    
    for (int i=0; i<layerCount; i++){
        ByteSize lsize =getLayerSize(layerDefs+i);
        size += lsize;
    }
    
    // get size of weight memory block (located within network, after layers)
    ByteSize weightBlockSize = getNetworkWeightBlockSize(layerCount, layerDefs);
    
    // add weight block size to the network
    size += weightBlockSize;
    
    return size;
}

```

To calculate the network size, I need to calculate each layer's size which in return requires to calculate each column's size which depends on each node's size which depends on the number of connections. Wow. Sounds complicated at first, but it's actually pretty straight forward. 
Hence, I'm going to skip a review of the detailed functions here. They can be reviewed in the code directly.


## Initialize the Network 

In the above *createNetwork* function we saw that besides setting some default values for the network, we need to *initialize* the network and set random weights.

Now, what do I mean by *initializing* the network? 
Don't forget, until now the network is simply a block of memory with some unknown content inside. 
We need to insert our desired structure into this memory block so that it mirrors our network design shown above: a network that holds layers that hold columns that hold nodes that hold connections which point to other nodes and weights. Let's see how this is done.

```c

void initNetwork(Network *nn, int layerCount, LayerDefinition *layerDefs){
    
    // Init the network's layers including their backward connections
    // Backward connections point to target nodes in the PREVIOUS layer and are used during FEED FORWARD
    // (i.e. during calculating node outputs = node activation)
    for (int l=0; l<layerCount; l++) initNetworkLayer(nn, l, layerDefs);
    
    // Init the network's forward connections
    // Forward connections point to target nodes in the FOLLOWING layer that point back to this node, and
    // and are used during BACK PROPAGATION (to speed-up calculating the proportional error)
    for (int l=0; l<layerCount; l++) initNetworkForwardConnections(nn, l);
    
}

```

First, we loop through all layers and initialize it one by one. After that, we loop again to set up each layer's forward connections.
The reason why we need 2 loops here is that in order for the *initNetworkForwardConnections* to work, each layer's following layer must have been initialized already.

The background is simple: *initNetworkForwardConnections* checks which nodes in the following layer point back to a node in this layer and then creates forward connections mirroring these backward connections.
Obviously, if the following layer hasn't been setup yet, there wouldn't be any forward connections. Hence, 2 loops.

Now, let's see what actually happens inside the *initNetworkLayer* function:

```c

void initNetworkLayer(Network *nn, int layerId, LayerDefinition *layerDefs){
    
    LayerDefinition *layerDef = layerDefs + layerId;
    
    // Calculate the layer's position by moving a single byte pointer forward
    // by the total sizes of all previous layers
    uint8_t *sbptr1 = (uint8_t*) nn->layers;
    for (int l=0; l<layerId; l++) sbptr1 += getLayerSize(layerDefs+l);
    Layer *layer = (Layer*) sbptr1;
    
    // Calculate the position of this layer's weights block
    uint8_t *sbptr2 = (uint8_t*) nn->weightsPtr;
    for (int l=0; l<layerId; l++) sbptr2 += getLayerWeightBlockSize(layerDefs+l);
    Weight *w = (Weight*) sbptr2;
    
    // Set default values for this layer
    layer->id              = layerId;
    layer->layerDef        = layerDef;
    layer->weightsPtr      = w;
    layer->size            = getLayerSize(layerDef);
    layer->columnCount     = getColumnCount(layerDef);
    
    // Initialize all columns inside this layer
    initNetworkColumns(nn, layerId);
    
}

```

In order to *initialize* a layer, we first need to know where each layer starts.
Remember, since each layer has a different size, we cannot use a simple array references such as *network.layer[1]* to locate a layer.
Instead, we use a single byte pointer and point it to the start of the network's layers *nn->layers*.
From there, we move it forward by exactly the size of each layer which we had calculated before.

Then, we define for each layer some basics such as its *id*, its *weightPtr*, etc., and we add a reference to its original *LayerDefinition*.
This will become handy because througout the code we will need to access a layer's definition (remember? things like layer type, number of nodes, activation function, etc.) quite often.

With this, though, we've only setup the *head* of the layer. 
What's missing is the structure underneath it, i.e. the columns inside this layer.
So that's what's next.


```c

void initNetworkColumns(Network *nn, int layerId){
    
    Layer *layer = getNetworkLayer(nn, layerId);
    
    int backwardConnCount = getNodeBackwardConnectionCount(layer->layerDef);
    int forwardConnCount  = getNodeForwardConnectionCount(layer->layerDef);
    
    ByteSize columnSize = getColumnSize(layer->layerDef);
    
    // Init all columns attached to this layer
    for (int c=0; c<layer->columnCount; c++){
        
        // Set pointer to the respective column position (using a single byte pointer)
        uint8_t *sbptr = (uint8_t*) layer->columns;
        sbptr += c * columnSize;
        
        Column *column = (Column*) sbptr;
        
        // Set default values of a node
        column->size     = columnSize;
        column->nodeCount= layer->layerDef->nodeMap.depth;
        column->maxConnCountPerNode = backwardConnCount+forwardConnCount;
        
        // Initialize all nodes of a column
        initNetworkNodes(nn, layerId, c);
        
    }
    
}

```

Again, the same logic as above applies. 
Since the size of a column structure is variable we use a single byte pointer to determine the starting point of each column inside this layer.
Once we've done that, we continue to intitialize all of the nodes inside this column.

```c

void initNetworkNodes(Network *nn, int layerId, int columnId){
    
    Layer *thisLayer    = getNetworkLayer(nn, layerId);
    Layer *prevLayer    = getNetworkLayer(nn, layerId-1);
    Column *column      = getLayerColumn(thisLayer, columnId);
    ByteSize nodeSize   = getNodeSize(thisLayer->layerDef);
    
    uint8_t *sbptr = (uint8_t*) column->nodes;

    // Create a vector containing the ids of the target columns (conv layers only)
    Vector *filterColIds = createFilterColumnIds(thisLayer, columnId, prevLayer);
    
    // Init all nodes attached to this column
    for (int n=0; n<column->nodeCount; n++){
    
        // Set pointer to the respective node position (using a single byte pointer)
        Node *node = (Node*) sbptr;
        sbptr += nodeSize;

        // Reset node's defaults
        setNetworkNodeDefaults(thisLayer, column, node, &nn->nullWeight);
        
        // Initialize backward connections of fully-connected layer node
        if (thisLayer->layerDef->layerType==FULLY_CONNECTED || thisLayer->layerDef->layerType==OUTPUT){
            
            int nodeId = (columnId * column->nodeCount) + n;

            // @attention When calculating the weightsId, only consider backwardConnections
            int layerWeightsId = nodeId * getNodeBackwardConnectionCount(thisLayer->layerDef);
            Weight *nodeWeight = thisLayer->weightsPtr + layerWeightsId;
            
            initNetworkBackwardConnectionsFCNode(node, prevLayer, nodeWeight);
        }
        
        // Initialize backward conections of convolutional layer node
        if (thisLayer->layerDef->layerType==CONVOLUTIONAL){
            // @attention Nodes on the same level share the same weight block
            initNetworkBackwardConnectionsConvNode(node, n, thisLayer->weightsPtr, prevLayer, filterColIds, &nn->nullWeight);
        }
    
    }

    free(filterColIds);
}

```

This function is central to the network's initialization process. 
In addition to setting some default values for all nodes, it's preparing the creation of backward connections of each node to the previous layer.
The way in which these *backward connections* are constructed is obviously different for *fully connected* and for *convolutional* layers.
Hence there are 2 different sub functions, *initNetworkBackwardConnectionsFCNode* for fully connected nodes and *initNetworkBackwardConnectionsConvNode* for convolutional nodes.

Let's look at the easier of the 2 first, the *fully connected* node:


```c

void initNetworkBackwardConnectionsFCNode(Node *thisNode, Layer *prevLayer, Weight *nodeWeightPtr){
    
    int connId = 0;
    int nodeWeightId = 0;
    
    ByteSize columnSize = getColumnSize(prevLayer->layerDef);
    
    uint8_t *sbptr_column = (uint8_t*) prevLayer->columns;
    
    // loop through the columns of the previous layer
    for (int col=0; col<prevLayer->columnCount;col++){
        
        Column *column = (Column *)sbptr_column;
        
        uint8_t *sbptr_node = (uint8_t*) column->nodes;

        // loop through the nodes of the column
        for (int n=0; n<column->nodeCount; n++){

            Connection *conn = &thisNode->connections[connId];
            
            conn->weightPtr = nodeWeightPtr + nodeWeightId;
            
            conn->nodePtr = (Node *)sbptr_node;
            
            sbptr_node += getNodeSize(prevLayer->layerDef);

            connId++;
            nodeWeightId++;
            
        }
        
        sbptr_column += columnSize;
        
    }
    
}

```

The function loops through the previous layer's nodes and creates a connection from the current node to __each__ node in the previous layer (hence *__fully__ connected*).
Remember, since nodes are structured in columns, we need to loop through the columns first, and then through the nodes inside each column respectively.

So far so good. That was the easier of the two. Now, let's look at how the wiring (building connections) works for *convolutional* nodes.

As outlined above, a convolutional node is connected to a selected group of neighboring nodes, located within a quadratic region of *filter* size.
This quadratic region (the filter or kernel window) needs to be calculated.
It depends on a number of parameters and changes as the filter is moved across the target layer.

```
#### Example: 

Let's say our network consists of 3 layers, the input layer containing the MNIST image, a convolutional layer and the output layer.
We know that the MNIST image has 28*28 pixels, hence the input layer has 784 columns. Each column only has 1 node (the *depth* of the input layer is 1) therefore there are also exactly 728 nodes. And let's assume our convolutional layer has dimensions of [24 * 24 * 10], i.e. there are 24 * 24 = 576 columns and each column has 10 nodes. And let's assume we defined a *filter* of 5 for this layer.

Now we start wiring up the 1st node in the convolutional layer. We need to create connections to a region of 5 x 5 (*filter* x *filter* size) columns located at the top left of the target (in this case = input) layer. The respective ids of these columns would be: 

[   0,   1,   2,   3,   4,
   28,  29,  30,  31,  32,
   56,  57,  58,  59,  60,
   84,  85,  86,  87,  88,
  112, 113, 114, 115, 116]

Then we move on to the 2nd node in the convolutional layer. Again, we create connections to a 5 x 5 region, but the region now moved to right along with the convolutional (*source*) node itself. Hence, the target column ids of the 2nd node in the convolutional layer would be:

[   1,   2,   3,   4,   5,
   29,  30,  31,  32,  33,
   57,  58,  59,  60,  61,
   85,  86,  87,  88,  89,
  113, 114, 115, 116, 117]

```

Above calculation is achieved by the following function. I trust you'll be able to walk through it, using the logic explained in the example above.

```c

void calcFilterColumnIds(Layer *srcLayer, int srcColId, Layer *tgtLayer, Vector *filterColIds){
    
    int srcWidth  = srcLayer->layerDef->nodeMap.width;
    int tgtWidth  = tgtLayer->layerDef->nodeMap.width;
    int tgtHeight = tgtLayer->layerDef->nodeMap.height;
    
    int filter = srcLayer->layerDef->filter;
    int stride = calcStride(tgtWidth, filter, srcWidth);
    
    int startX = (srcColId % srcWidth) * stride;
    int startY = floor((double)srcColId/srcWidth) * stride;    
    int id=0;

    for (int y=0; y<filter; y++){
        
        for (int x=0; x<filter; x++){
            
            int colId = ( (startY+y) * tgtWidth) + (startX+x);
            
            // Check whether target columnId is still within the target node range
            // If NOT then assign a dummy ID ("OUT_OF_RANGE") that is later checked for
            if (
                (floor(colId / tgtWidth) > (startY+y)) ||  // filter exceeds nodeMap on the right
                (colId >= tgtWidth * tgtHeight)            // filter exceeds nodeMap on the bottom
                ) colId = OUT_OF_RANGE;
                
            filterColIds->vals[id] = colId;            
            id++;
        }
    }
}

```

Once we calculated this vector of *targetColumnIds* we pass it into our *initNetworkBackwardConnectionsConvNode* function which then creates connections from the source node to the target nodes specified in the vector.

```c

void initNetworkBackwardConnectionsConvNode(Node *node, int srcLevel, Weight *srcLayerWeightPtr, Layer *targetLayer, Vector *filterColIds, Weight *nullWeight){
    
    int filterSize = filterColIds->count;
    int tgtDepth   = targetLayer->layerDef->nodeMap.depth;
    
    for (int posInsideFilter=0; posInsideFilter<filterSize; posInsideFilter++){
        
        int targetColId = (int)filterColIds->vals[posInsideFilter];
        
        for (int tgtLevel=0; tgtLevel<tgtDepth; tgtLevel++){
            
            Connection *conn = &node->connections[ (tgtLevel*filterSize)+posInsideFilter];
            
            if (targetColId!=OUT_OF_RANGE){
                
                int weightPosition = (srcLevel*(tgtDepth*filterSize)) + (tgtLevel*filterSize) + posInsideFilter;
                
                Weight *tgtWeight = srcLayerWeightPtr + weightPosition;
                
                Node *tgtNode = getNetworkNode(targetLayer, targetColId, tgtLevel);
                
                conn->nodePtr   = tgtNode;
                conn->weightPtr = tgtWeight;
            }    
            else {
            // if filter pixel is out of range of the target nodes then point to NODES if THIS layer
            // this kind of pointer needs to be captured later i.e. should NOT be CALCULATED/ACTIVATED
            conn->nodePtr   = NULL;
            conn->weightPtr = nullWeight;
            }
        }
    }
}
```

Depending on how many nodes we defined in our convolutional layer, some of the nodes in the target (=previous) layer are purposely skipped. 
This is called the *stride*. A *stride* of 2 means that every other node in the target layer is skipped. 
The idea is to *downsize* the original image layer by layer, and have each following layer represent some higher level feature of the previous layer.

```c
int calcStride(int tgtWidth, int filter, int srcWidth){
  return ceil(((double)tgtWidth - filter)/(srcWidth-1));
}
```


### Initialize Weights

Now that we've fully setup all of the connections between the nodes and from the nodes to their weights, we still need to initialize the weights.
Remember, all weights are located in a big weight block at the end of the *network* structure. 

All we need to do is initialize each weight with a random value from 0-1. 
For better performance I found that it helps to reduce the value to a random number smaller than 0.5 or 0.4 and also make every other number negative.

```c
void initNetworkWeights(Network *nn){
    
    // Init weights in the weight block
    for (int i=0; i<nn->weightCount; i++){
        Weight *w = &nn->weightsPtr[i];
        *w = 0.4 * (Weight)rand() / RAND_MAX;   // multiplying by a number <0 results in better performance
        if (i%2) *w = -*w;                      // make half of the weights negative (for better performance)
    }
    
    // Init weights in the nodes' bias
    for (int l=0; l<nn->layerCount;l++){
        Layer *layer = getNetworkLayer(nn, l);
        for (int c=0; c<layer->columnCount; c++){
            Column *column = getLayerColumn(layer, c);
            for (int n=0; n<column->nodeCount; n++){
                // init bias weight
                Node *node = getColumnNode(column, n);
                node->bias = (Weight)rand()/(RAND_MAX);
                if (n%2) node->bias = -node->bias;  // make half of the bias weights negative
            }
        }
    }
}
```


## Train the Network

Once the network has been fully setup and initialized, we can train it in exactly the same manner as described in my previous post: 
- read an image, 
- feed it into the network, 
- feed its value forward through all the layers, applying an activation function each time
- backpropagate the error (difference of desired and actual outcome) through all layers and update its weights

To obtain some indication of the network's progress we can count the number of correct classifications versus the total number of images and calculate an ongoing accuracy rate.

```c
void trainNetwork(Network *nn){
    
    // open MNIST files
    FILE *imageFile, *labelFile;
    imageFile = openMNISTImageFile(MNIST_TRAINING_SET_IMAGE_FILE_NAME);
    labelFile = openMNISTLabelFile(MNIST_TRAINING_SET_LABEL_FILE_NAME);
    
    int errCount = 0;
    
    // Loop through all images in the file
    for (int imgCount=0; imgCount<MNIST_MAX_TRAINING_IMAGES; imgCount++){
        
        // Reading next image and its corresponding label
        MNIST_Image img = getImage(imageFile);
        MNIST_Label lbl = getLabel(labelFile);
        
        // Convert the MNIST image to a standardized vector format and feed into the network
        Vector *inpVector = getVectorFromImage(&img);
        feedInput(nn, inpVector);

        // Feed forward all layers (from input to hidden to output) calculating all nodes' output
        feedForwardNetwork(nn);

        // Back propagate the error and adjust weights in all layers accordingly
        backPropagateNetwork(nn, lbl);

        // Classify image by choosing output cell with highest output
        int classification = getNetworkClassification(nn);
        if (classification!=lbl) errCount++;
    }
    // Close files
    fclose(imageFile);
    fclose(labelFile);
}
```

Once the network has been trained on the the 50,000 images in the training set, we stop training and run the network on the 10,000 images in the testing set.
The process during testing is exactly the same as during training, the only difference is that we switch off the *learning*, i.e. we don't back propagate the error and don't update any weights.

```c
void testNetwork(Network *nn){
    
    // open MNIST files
    FILE *imageFile, *labelFile;
    imageFile = openMNISTImageFile(MNIST_TESTING_SET_IMAGE_FILE_NAME);
    labelFile = openMNISTLabelFile(MNIST_TESTING_SET_LABEL_FILE_NAME);
    
    int errCount = 0;
    
    // Loop through all images in the file
    for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++){
        
        // Reading next image and its corresponding label
        MNIST_Image img = getImage(imageFile);
        MNIST_Label lbl = getLabel(labelFile);
        
        // Convert the MNIST image to a standardized vector format and feed into the network
        Vector *inpVector = getVectorFromImage(&img);
        feedInput(nn, inpVector);
        
        // Feed forward all layers (from input to hidden to output) calculating all nodes' output
        feedForwardNetwork(nn);
        
        // Classify image by choosing output cell with highest output
        int classification = getNetworkClassification(nn);
        if (classification!=lbl) errCount++;   
    }
    // Close files
    fclose(imageFile);
    fclose(labelFile);
}
```

In case you wondered how is the *classification* of an image in the output layer done: 
I'm not using *softmax* or any other function. 
Instead, I'm simply comparing all outputs of the 10 nodes in the output layer and take the highest. :)

```c
int getNetworkClassification(Network *nn){
    
    Layer *l = getNetworkLayer(nn, nn->layerCount-1);   // get output layer
    
    Weight maxOut = 0;
    int maxInd = 0;
    
    for (int i=0; i<l->columnCount; i++){   
        Node *on = getNetworkNode(l,i,0); // the output layer always has a depth of 1 
        if (on->output > maxOut) {
            maxOut = on->output;
            maxInd = i;
        }
    }
    return maxInd;
}

```

That's it. Now that we've done all the coding, let's look at performance.


## Network Performance

To assess the performance of the network let's look at its accuracy as well as its speed.
Here's a summary of the different network designs that I tried and their respective best results.


```
### Test Results

| Number of Layers | Input Layer | Layer 1 | Layer 2  | Layer 3 | Output Layer | Learning Rate | Number of Trains | Accuracy | Speed |
| ---------------- |-------------|---------|----------|---------|--------------|---------------|------------------|----------|-------|
| 2                | 28 x 28 pix |    /    |    /     |    /    | 10 nodes     |               |                  |          |       |
| 2                | 28 x 28 pix |    /    |    /     |    /    | 10 nodes     |               |                  |          |       |
| 2                | 28 x 28 pix |    /    |    /     |    /    | 10 nodes     |               |                  |          |       |
| 2                | 28 x 28 pix |    /    |    /     |    /    | 10 nodes     |               |                  |          |       |
| 2                | 28 x 28 pix |    /    |    /     |    /    | 10 nodes     |               |                  |          |       |
| 2                | 28 x 28 pix |    /    |    /     |    /    | 10 nodes     |               |                  |          |       |

```

This leads us to the most difficult part of machine learning which is: How to find the hyper-parameters that achieve the best results?


## Hyper-parameter Optimization

Surprisingly, optimizing hyper-parameters such as the learning rate or the number of nodes is still mostly a manual process. Based on my reading I understand that *grid search* or *Bayesian Optimization* are more or less equally good or bad than a *Random Search*.

Therefore, I plan to dedicate my next post to this topic and started working on my own optimization tool to get the best results from my network.
Stay tuned! :)


---

## Code & Documentation

As always you can find all the code for this exercise on my [Github project page](https://github.com/mmlind/mnist-dnn/), including full [code documentation](https://rawgit.com/mmlind/mnist-dnn/master/doc/html/index.html).

Happy Hacking!

![_config.yml]({{ site.baseurl }}/images/mnist_logo.png)

