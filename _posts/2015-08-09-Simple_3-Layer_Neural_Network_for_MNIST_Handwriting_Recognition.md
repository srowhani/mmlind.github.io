---
layout: post
title: Simple 3-Layer Neural Network for MNIST Handwriting Recognition
---

I've extended my simple 1-Layer neural network to include a hidden layer and use the back propagation algorithm for updating connection weights.
The size of the network (number of neurons per layer) is dynamic, yet it's accuracy in classifying the handwritten digits in the MNIST database is still disappointing. Read why...

![_config.yml]({{ site.baseurl }}/images/mnist-3lnn-logo.jpg)

In a previous blog post I introduced a simple [1-Layer neural network for MNIST handwriting recognition](../Simple_1-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/).
It was based on a single layer of perceptrons whose connection weights are adjusted during a supervised learning process.
The result was an 85% accuracy in classifying the digits in the MNIST testing dataset.

The network did not use a *real* activation function (only simple linear "normalization") and no weighted error back propagation. 
Now, I want to add these things to see how this will improve the network's effectiveness. 
At the same time I also want to make the code more versatile and re-usable, offering a standardized interface, and to allow for dyanmic network sizing. 

![_config.yml]({{ site.baseurl }}/images/3lnn.svg)


## From 1 Layer to 3 Layers

The first major change in the design is adding a hidden layer between input and output. 
So how come a 1 layer network becomes a 3 layer network?
It's simply a different naming convention.
The 1-layer network only had one single layer of perceptrons, the output layer. 
In my previous design, the input to the network was NOT part of the network, i.e. whenever this input was needed (when calculating a node's output and when updating a node's weights) I refered to a `MNIST image`, a variable *outside* of the network.

When I redesigned the network I found it advantagous to include the input feed *inside* the network.
It allows to treat the network as an independent object (data structure) without external references.

Therefore, by adding a hidden layer, plus treating the input as a network layer as well, the 1-layer network becomes a 3-layer network.


## Dynamically Sized Network

One of the challenges when redesigning the network was making its size dynamic. 
In C, dynamically sized objects are not as easy to implement as in higher-level languages. 


### The Data Model

I decided to make use of a C feature originally refered to as the *[struct hack](http://c-faq.com/struct/structhack.html)* until it officially became part of C as *[flexible array member](https://en.wikipedia.org/wiki/Flexible_array_member)*.
The idea is simple: place an empty array _at the end_ of a struct definition and manually allocate as much memory for the struct as it actually needs.

I make extensive use of this feature by stacking several layers of dynamically sized data structures. 

Let's start at the smallest, most inner level of a network, the node. (*Node* refers to the same concept as a *perceptron* or *cell*, as I previously had called it.)

```c
struct Node{
    double bias;
    double output;
    int wcount;
    double weights[];
};
```

The number of weights of a node normally depends on the number of nodes in the previous layer (assuming a fully connected design). 
Since we want this number to be dynamic we use an empty array or *flexible array member*.
But, in order to be able to access each individual weight later, we need to remember how many weights a node actually has.
This is what `wcount` (weight count) is for.

Next we put several of these nodes tegether to form a layer

```c
struct Layer{
    int ncount;
    Node nodes[];
};
```

and use `ncount` (node count) to remember how many nodes are actually stored inside this layer.

Lastly, we stack several (for now only 3: input, hidden and output) of such layers into a `network`:

```c
struct Network{
    int inpNodeSize;
    int inpLayerSize;
    int hidNodeSize;
    int hidLayerSize;
    int outNodeSize;
    int outLayerSize;
    Layer layers[];
};
```

Since the number of layers (for now) is fixed, i.e. always 3 (input, hidden, output), I did _not_ add a variable `lcount` (layer count) to remember the number of layers. This could be done later if the number of (hidden) layers inside the network is required to be dynamic.


### Manual Memory Allocation 

Once the data model for the network and its components is defined, we need to manually allocate memory for each part.
If we assume that all layers are *fully connected*, i.e. each node connects to all nodes in the following layer, then the overall size of the network only depends on 3 numbers:

```
1. Size of the input vector (= number of pixels of a MNIST image)
2. Number of nodes in the hidden layer 
3. Number of nodes in the output layer
```

Hence, we'll create an interface for creating the network which looks like this:

```c
Network *createNetwork(int size_of_input_vector,    
                       int number_of_nodes_in_hidden_layer, 
                       int number_of_nodes_in_output_layer);
```

Now we calculate the size of each node type (*input*, *hidden*, *output*) as well as the required memory for each of the 3 layers. Adding up the layers' sizes then gives us the size of the overall network.

```c
Network *createNetwork(int inpCount, int hidCount, int outCount){
    
    // Calculate size of INPUT Layer
    int inpNodeSize     = sizeof(Node);         // Input layer has 0 weights
    int inpLayerSize    = sizeof(Layer) + (inpCount * inpNodeSize);
    
    // Calculate size of HIDDEN Layer
    int hidWeightsCount = inpCount;
    int hidNodeSize     = sizeof(Node) + (hidWeightsCount * sizeof(double));
    int hidLayerSize    = sizeof(Layer) + (hidCount * hidNodeSize);
    
    // Calculate size of OUTPUT Layer
    int outWeightsCount = hidCount;
    int outNodeSize     = sizeof(Node) + (outWeightsCount * sizeof(double));
    int outLayerSize    = sizeof(Layer) + (outCount * outNodeSize);
    
    // Allocate memory block for the network
    Network *nn = (Network*)malloc(sizeof(Network) + inpLayerSize + hidLayerSize + outLayerSize);
    
    return nn;
}

```

Since we will need to refer to the aboves values throughout other parts of our code we store them as part of the network:

```c
    nn->inpNodeSize     = inpNodeSize;
    nn->inpLayerSize    = inpLayerSize;
    nn->hidNodeSize     = hidNodeSize;
    nn->hidLayerSize    = hidLayerSize;
    nn->outNodeSize     = outNodeSize;
    nn->outLayerSize    = outLayerSize;

```

Once the network is created, it's still only an empty memory block. 
We now need to fill this memory block with the 3-tier layer structure.

The way we do this is by creating each layer as a temporary, independent object, i.e. in a different memory block.
Then we copy this memory block into the larger memory block allocated for the network and delete (`free`) the memory block of the temporary layer.

Obviously, the most critical point here is to copy the layer to the correct memory address as required by the data model.
Since we calculated each node's and each layer's size, this can be easily done using a *single byte pointer*.

I put this into a separate function and call it `initializing the network`:


```c
void initNetwork(Network *nn, int inpCount, int hidCount, int outCount){
    
    // Copy the input layer into the network's memory block and delete it
    Layer *il = createInputLayer(inpCount);
    memcpy(nn->layers,il,nn->inpLayerSize);
    free(il);
    
    // Move pointer to end of input layer = beginning of hidden layer
    uint8_t *sbptr = (uint8_t*) nn->layers;     // single byte pointer
    sbptr += nn->inpLayerSize;
    
    // Copy the hidden layer into the network's memory block and delete it
    Layer *hl = createLayer(hidCount, inpCount);
    memcpy(sbptr,hl,nn->hidLayerSize);
    free(hl);
    
    // Move pointer to end of hidden layer = beginning of output layer
    sbptr += nn->hidLayerSize;
    
    // Copy the output layer into the network's memory block and delete it
    Layer *ol = createLayer(outCount, hidCount);
    memcpy(sbptr,ol,nn->outLayerSize);
    free(ol);
        
}
```

Let's look at this in more detail:
A *single byte pointer* is simply a pointer pointing to memory blocks of byte size 1. 
I.e. we can easily move the pointer throughout the address space either by incrementing it via `pointer++` or by adding the number of bytes that we want the pointer to move forward by `pointer += bytes_to_move_forward`.

First, we define the pointer and make it point to the memory address where the first layer, the *input* layer, should be located:

```c
uint8_t *sbptr = (uint8_t*) nn->layers;     // single byte pointer
```

Then, we move the pointer forward to the memory address of the 2nd layer, the *hidden* layer, simply by increasing the pointer by the size of the input layer:


```c
sbptr += nn->inpLayerSize;
```

Above should help to understand the above `initNetwork()` function.
Inside of it we are creating the layers by calling a `createInputLayer()` and `createLayer()` function.
These functions will create and return (a pointer to) a `Layer` object which is already pre-filled with default values.

Let's start with the `createInputLayer()` function:

```c
Layer *createInputLayer(int inpCount){
    
    int inpNodeSize     = sizeof(Node);         // Input layer has 0 weights
    int inpLayerSize    = sizeof(Layer) + (inpCount * inpNodeSize);
    
    Layer *il = malloc(inpLayerSize);
    il->ncount = inpCount;
    
    // Create a detault input layer node
    Node iln;
    iln.bias = 0;
    iln.output = 0;
    iln.wcount = 0;
    
    // Use a single byte pointer to fill in the network's content
    uint8_t *sbptr = (uint8_t*) il->nodes;
    
    // Copy the default input layer node x times
    for (int i=0;i<il->ncount;i++){
        memcpy(sbptr,&iln,inpNodeSize);
        sbptr += inpNodeSize;
    }
    
    return il;
}
```

The *input* layer is different from the other 2 layers (*hidden* and *output*) in two aspects:

1. As I outlined above, the input layer is strictly speaking not a real network layer. 
I.e. its nodes are not perceptrons because this layer does not have any connections, and hence any weights, to a previous layer.
For our coding this means that the size of a node's weight array (`weights[]`) is 0 and therefore the size of an input node is the same as the *actual* size of a node struct `sizeof(Node)`.

2. The second major difference of the input layer is that its main objective is stoing the input that is fed into the network.
But for coding consistency and simplicity I want to use the same data model with the same naming conventions for the input layer as for the other two layers.
Therefore, I use the node's `output` variable to store the network's *input*. 
(Keep this point in mind for later!)

Next, let's look at the function how to create a *hidden* and an *output* layer.
Since both layers share the same structure we can use the same function.

```c
Layer *createLayer(int nodeCount, int weightCount){
    
    int nodeSize = sizeof(Node) + (weightCount * sizeof(double));
    Layer *l = (Layer*)malloc(sizeof(Layer) + (nodeCount*nodeSize));
    
    l->ncount = nodeCount;
    
    // create a detault node
    Node *dn = (Node*)malloc(sizeof(Node) + ((weightCount)*sizeof(double)));
    dn->bias = 1;
    dn->output = 0;
    dn->wcount = weightCount;
    for (int o=0;o<weightCount;o++) dn->weights[o] = (0.5*(rand()/(double)(RAND_MAX)));
    
    uint8_t *sbptr = (uint8_t*) l->nodes;     // single byte pointer
    
    // copy the default node into the layer
    for (int i=0;i<nodeCount;i++) memcpy(sbptr+(i*nodeSize),dn,nodeSize);
    
    free(dn);
    
    return l;
}
```

The above code first creates an empty memory block for the layer and then fills it with n copies of a *default node*.
This default node is defined with weights of random size.


## Training the Network

Once the network has been created and pre-filled with random weights we can start training it.
The training algorithm uses the following steps:

```

1. Feed image data into the network
2. Calculate node outputs of *hidden* and *output* layers (=FEED FORWARD)
3. Back-propagate the error and adjust the weights (=FEED BACKWARD)
4. Classify the image (*guess* what digit is presented in the image)

```

### 1. Feeding Data into the Network

As outlined in my [previous MNIST code example](../Simple_1-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/) the data we want to feed into the network exists in the form of a `MNIST_Image` structure which holds a 28*28 pixel image.

In order to maintain a standardized interface for this neural network code, I don't want to feed the `MNIST_Image` object directly into the network.
Instead, I first convert it to a neutral `Vector` structure.

The `Vector` makes again use of C's *flexible array member* functionality so that it can be of dynamic size.

```c
struct Vector{
    int size;
    double vals[];
};
```

We convert the `MNIST_Image` structure into this `Vector` structure using the following function:


```c
Vector *getVectorFromImage(MNIST_Image *img){
    
    Vector *v = (Vector*)malloc(sizeof(Vector) + (28 * 28 * sizeof(double)));
    
    v->size = 28 * 28;
    
    for (int i=0;i<v->size;i++)
        v->vals[i] = img->pixel[i] ? 1 : 0;
    
    return v;
}
```

Now we can feed this input vector into the network:

```c
void feedInput(Network *nn, Vector *v) {
    
    Layer *il;
    il = nn->layers;
    
    Node *iln;
    iln = il->nodes;
    
    // Copy the vector content to the "output" field of the input layer nodes
    for (int i=0; i<v->size;i++){
        iln->output = v->vals[i];
        iln++;           
    }
    
}
```

As I explained above, I chose to utilize the `output` field of an input node to hold the image's pixels.


### 2. Feed Forward

Once the network is filled with an image's pixels, we can start *feeding* this data *forward*, from the input layer to the hidden layer and from the hidden layer to the output layer. 

```c
void feedForwardNetwork(Network *nn){
    calcLayer(nn, HIDDEN);
    calcLayer(nn, OUTPUT);
}
```

*Feed forward* means calculating the output values of each node by multiplying its weights with the output values of the previous layer's nodes that it is connected to.
This mechanism is the same as in my simple [1-Layer NN MNIST code](../Simple_1-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/) so feel free to check there if below is not clear.

```c
void calcLayer(Network *nn, LayerType ltype){
    Layer *l;
    l = getLayer(nn, ltype);
    
    for (int i=0;i<l->ncount;i++){
        calcNodeOutput(nn, ltype, i);
        activateNode(nn,ltype,i);
    }
}
```

Calculating the *hidden* layer and the *output* layer works the same way, hence we can use the same function.
However, in order to be able to know which *type of layer* we're working on we're introducing a 

```c
typedef enum LayerType {INPUT, HIDDEN, OUTPUT} LayerType;
```

and pass it as an argument to the `calcLayer()` function which itself applies the following steps to each of its nodes:

```
1. Calculating the node's output
2. Executing an 'activation function' on the node's output
```

So, let's first calculate a node's output value:


```c
void calcNodeOutput(Network *nn, LayerType ltype, int id){
    
    Layer *calcLayer = getLayer(nn, ltype);
    Node *calcNode = getNode(calcLayer, id);
    
    Layer *prevLayer;
    int prevLayerNodeSize = 0;
    
    if (ltype==HIDDEN) {
        prevLayer = getLayer(nn, INPUT);
        prevLayerNodeSize = nn->inpNodeSize;
    }
    else {
        prevLayer = getLayer(nn, HIDDEN);
        prevLayerNodeSize = nn->hidNodeSize;
    }
    
    calcNode->output = 0;
    uint8_t *sbptr = (uint8_t*) prevLayer->nodes;
    
    for (int i=0; i<prevLayer->ncount;i++){
        Node *prevLayerNode = (Node*)sbptr;
        calcNode->output += prevLayerNode->output * calcNode->weights[i];
        sbptr += prevLayerNodeSize;
    }

}

```

To do this we need to loop through the node's connections, i.e. loop through its weights and the output values in the previous layer that it is connected to.
However, since we defined the network's components (layer, nodes, weights) using *flexible array members* the compiler does not know where one component ends and where the next one starts. 
Hence, it is __not__ possible to locate, for example, a node by simply referring to the n'th member of the node array: ~~layer.node[n]~~!

Since both a layer's and a node's size are unknown to the compiler I use a *single byte pointer* to navigate through the allocated memory space. 
The above code does so by calling a `getLayer()` and `getNode()` function to access a particular layer or a particular node inside the network.

```c
Node *getNode(Layer *l, int nodeId) {
    
    int nodeSize = sizeof(Node) + (l->nodes[0].wcount * sizeof(double));
    uint8_t *sbptr = (uint8_t*) l->nodes;
    
    sbptr += nodeId * nodeSize;
    
    return (Node*) sbptr;
}

```

Let's take a more detailed look at how this works. To access a particular node we simply set the pointer to the first node of this layer and then move the pointer forward by *the number of nodes* (given as the node's id) times *the size of a node*. 
The latter is calculated by adding the default size of a node `sizeof(Node)` to the product of the number of weights (`wcount`) times the size of a weight `sizeof(double)`.

Accessing a particular `Layer` works similar. We place the pointer to the first layer (given by `nn->layers`) and then move it forward as required.

To access the *input* layer, we don't need to move forward at all since it *is* the first layer in the network.
To access the *hidden* layer, we need to move the pointer forward by *the size of the input layer*.
And, to access the *output* layer, we need to move the pointer forward by *the size of the input layer* plus *the size of the hidden layer*.

```c
Layer *getLayer(Network *nn, LayerType ltype){
    
    Layer *l;
    
    switch (ltype) {
        case INPUT:{
            l = nn->layers;
            break;
        }
        case HIDDEN:{
            uint8_t *sbptr = (uint8_t*) nn->layers;
            sbptr += nn->inpLayerSize;
            l = (Layer*)sbptr;
            break;
        }
            
        default:{ // OUTPUT
            uint8_t *sbptr = (uint8_t*) nn->layers;
            sbptr += nn->inpLayerSize + nn->hidLayerSize;
            l = (Layer*)sbptr;
            break;
        }
    }
    
    return l;
}
```

If you're following and understanding this post so far, the above code should be self-explanatory. 
Yet, one line worth highlightening is:

```c
l = (Layer*)sbptr;
```

I'm using a *single byte pointer* to move through the address space but I actually need to return a *Layer* pointer as reference. 
Therefore, I *cast* the *single byte pointer* into a `Layer` pointer and make it point to the same address.


### Activation Function

After the node's output value has been calculated we need to pass this value through an *activation function*.
The purpose of the *activation function* is to constrain the output value (which could be very large) to a standard range, for example, to between 0 and +1, or to between -1 and +1. 

The exact range depends on *what* activation function you use. The one most often refered to in the context of designing neural networks is the SIGMOID function. 
I'm not going into the mathematical details of this function as this is outside of the scope of this post. 
If you're interested, there are plenty of good explanations about SIGMOID and other *activation functions* on the web. 

Since I want the network to be flexible as to what *activation function* it uses I decided to make this a parameter of the network.
The current code suppports two in particular, SIGMOID and TANH, but this list could be expanded as needed.

```c
typedef enum ActFctType {SIGMOID, TANH} ActFctType;
```

What type of *activation function* is desired can be defined after the network has been created. 
It is also possible to choose a different *activation function* for hidden and output layer.

```c
    nn->hidLayerActType = SIGMOID;
    nn->outLayerActType = SIGMOID;
```

The reason why I made this a network parameter instead of a function argument passed into the `activation function` is that the back-propagation algorithm (see below) requires using the derivative of this function. 
So we need to remember how we *activated* a node because when we later adjust its weights the derivate of the same function must be used.

The different types of *activation function* the network supports are implement in below `activateNode()` function which applies what has been defined in `nn->hidLayerActType` and `nn->outLayerActType` variables to the hidden and ouput layer respectively:


```c
void activateNode(Network *nn, LayerType ltype, int id){
    
    Layer *l = getLayer(nn, ltype);
    Node *n = getNode(l, id);
    
    ActFctType actFct;
    
    if (ltype==HIDDEN) actFct = nn->hidLayerActType;
    else actFct = nn->outLayerActType;
    
    if (actFct==TANH)   n->output = tanh(n->output);
    else n->output = 1 / (1 + (exp((double)-n->output)) );
    
}
```


### 3. Feed Backward (Error Back-Propagation)

After we finished our *feed forward* we start to *back propagate* the *error*. 
The *error* here refers to the difference of the *desired* or *target* output (which in MNIST is provided as the image's *label*) and the *actual* output of the output layer nodes.

Let's do this step by step, or layer by layer:

```c
void backPropagateNetwork(Network *nn, int targetClassification){
    
    backPropagateOutputLayer(nn, targetClassification);
    
    backPropagateHiddenLayer(nn, targetClassification); 
}
```

We first back propagate the output layer:


```c
void backPropagateOutputLayer(Network *nn, int targetClassification){
    
    Layer *ol = getLayer(nn, OUTPUT);
    
    for (int o=0;o<ol->ncount;o++){
        
        Node *on = getNode(ol,o);
        
        int targetOutput = (o==targetClassification)?1:0;
        
        double errorDelta = targetOutput - on->output;
        double errorSignal = errorDelta * getActFctDerivative(nn, OUTPUT, on->output);
        
        updateNodeWeights(nn, OUTPUT, o, errorSignal);
        
    } 
}
```


Our `targetClassification` is the image's *label*, i.e. a single digit 0-9. 
The `targetOutput` of an output node, however, is of binary valye, i.e. either 0 or 1.

For example: if we train on an image presenting a "3", the corresponding *label* will be the integer "3". 
Then the `targetOutput` of node 3 (which is actually the 4th node since we started at the 0th) will be a "1" while the `targetOutput` of all other 9 nodes will be "0". This is done via below command:


```c
int targetOutput = (o==targetClassification)?1:0;
```

The back propagation of the *hidden* layer works the same say, yet the actual algorithm is a little more complex since it requires to first calculate the *sum of all weighted outputs* connected to a node.
I'm not going into the details of the back propagation algorithm in this post. 
If you're interested, there are plenty of detailed and more qualified sources about this subject on the web.

In both of the above back propagation functions (for the *output* layer and for the *hidden* layer) we need to get the derivative of the activation function and update the node's weights.

The calculation of the derivates of the supported SIGMOID and TANH functions is implemented in the getActFctDerivative() function:

```c
double getActFctDerivative(Network *nn, LayerType ltype, double outVal){
    
    double dVal = 0;
    ActFctType actFct;
    
    if (ltype==HIDDEN) actFct = nn->hidLayerActType;
                  else actFct = nn->outLayerActType;
    
    if (actFct==TANH) dVal = 1-pow(tanh(outVal),2);
                 else dVal = outVal * (1-outVal);
    
    return dVal;
}
```

Lastly, each node's weights are updated using the back propagated error:

```c
void updateNodeWeights(Network *nn, LayerType ltype, int id, double error){
    
    Layer *updateLayer = getLayer(nn, ltype);
    Node *updateNode = getNode(updateLayer, id);
    
    Layer *prevLayer;
    int prevLayerNodeSize = 0;
    if (ltype==HIDDEN) {
        prevLayer = getLayer(nn, INPUT);
        prevLayerNodeSize = nn->inpNodeSize;
    } else {
        prevLayer = getLayer(nn, HIDDEN);
        prevLayerNodeSize = nn->hidNodeSize;
    }
    
    uint8_t *sbptr = (uint8_t*) prevLayer->nodes;
    
    for (int i=0; i<updateNode->wcount; i++){
        Node *prevLayerNode = (Node*)sbptr;
        updateNode->weights[i] += (nn->learningRate * prevLayerNode->output * error);
        sbptr += prevLayerNodeSize;
    }
    
}
```

On a side note, I decided to make the *learning rate* part of the network (instead of a *global* constant).
It can be set as follows:

```c
nn->learningRate    = 0.5;
```


### 4. Classify Output

Now that the error has been back propagated and the weights have been updated, we can query our network to attempt to classify the image. This is simply done by retrieving the index of the output node with the highest value.


```c
int getNetworkClassification(Network *nn){
    
    Layer *l = getLayer(nn, OUTPUT);
    
    double maxOut = 0;
    int maxInd = 0;
    
    for (int i=0; i<l->ncount; i++){
        
        Node *on = getNode(l,i);
        
        if (on->output > maxOut){
            maxOut = on->output;
            maxInd = i;
        }
    }
    
    return maxInd;
}
```

That's it. We're done. So I thought. Only to discover that this is when the *other*, less obvious part of building a neural network starts.


## Fine-tuning Network Performance

The neural network's accuracy is defined as the ratio of correct classifications (_in the testing set_) to the total number of images processed. 

Using the code above, my 3-layer network achieves an out-of-the-box accuracy of 89% which is *only* slightly better than the simple 1-layer network I built before. How come? 

This post so far may give the impression that building and coding a neural network is a pretty straight forward and deterministic excercise. 
Unfortunately, it's not. 

There are a number of parameters that may strongly impact the network's performance. 
Sometimes even a slight change dramatically impacts the network's accuracy which during the coding process for this post often led me to believe that my algorithm and/or code were wrong while they were not. It was just wrongly set parameters.

In particular, I found the following parameters to most significantly impact the network's performance:

```
1. Number of nodes in the hidden layer
2. Type of activation function
3. Values of initial weights
4. Learning rate

```

Let's go through them.

### Number of Nodes in the Hidden Layer

This one is obvious. The higher the number of hidden nodes the more the network will adapt to the training data and "remember" it, thereby preventing *generalization*.
But *generalization*, i.e. the ability to apply features that have been *learned* during training to completely new and unknown data sets, is exactly what we do want from our network.

On the other side, the smaller the number of nodes in the hidden layer, the less complexity the network will be able to grasp. For example, in our MNIST database, there may be many different ways of hand-writing a "4". The more complex our data set, the more hidden nodes are needed.

Via numerous trials I found that using 20 nodes in the hidden layer achieves the best result. 


### Activation Function

This one is also obvious, albeit a little less so. 
I found that changing the *activation function* requires to make additional parameters change as well to avoid a significant performance hit.
In particular, the *activation function* is linked to what initial weights were chosen (see below) and to what learning rate was defined (see below).

In my tests I found that SIGMOID performed slightly better than TANH, although I suppose this is rather due to my overall design. Based on my reading I had expected that TANH outperforms SIGMOID.


### Initial Weights

This one is a lot less obvious. The weights in neural networks are usually initialized using random values 0 to +1 or -1 to +1. 
In fact, in my previous 1-layer MNIST network I had found that initializing all weights to 0.5 leads to almost the same result as initializing them to a random value between 0 and +1.

Now, with the added hidden layer and the use of an activation function, the network behaves very differently. 

I started by using random values between 0 and +1 and wondered about the desastrous performance. 
I then changed to include negative values, i.e. using random values -1 to +1 which improved performance a lot. 
If you understand how SIGMOID and TANH work, though, it's less of a surprise why this subtle change has a rather large impact.

Yet, while I was trying to further fine-tune the network, I found that another subtle change significantly improved the network's accuracy: 
instead of using random values between -1 and +1 I used values between 0 and +1 and made every second value negative.

```c
    for (int o=0;o<weightCount;o++) {
        dn->weights[o] = (0.5*(rand()/(double)(RAND_MAX)));
        if (o%2) dn->weights[o] = -dn->weights[o];  // make half of the numbers negative
    }
```

Wow! This was unexpected. It led me to the conclusion that for best performance the network needs to be initialized not only randomly but *evenly* random or *symetrically* random. 


### Learning Rate

While this parameter is obvious, it's significant impact on the network's performance did surprise me several times.

The *learning rate* defines the factor by which an *error* is applied as a *weight update*. 
The higher the *learning rate* the faster the network adapts, i.e. the faster it should reach its point of optimal performance. 
However, if the rate is too high the network may *jump* over the optimum and may settle in a point of lower accuracy.

On the other side, a smaller *learning rate* allows the network to make very fine changes but overall it may take so long that it reaches the end of the dataset before reaching its optimal point of performance.

In my tests I found that the *optimal* learning rate depends on the chosen type of activation function. 
While for SIGMOID my network did well using a learning rate of 0.5, I had to change it to 0.05 to achieve an almost equal performance with TANH.


## Conclusion

Coding this 3-layer neural network to recognize the MNIST digits has been an interesting excercise. 
It provided valuable insights into the *unpredictability* of neural networks. 
In particular, it helped to alert me to how small parameter changes can cause very different results.

Overall, the network's performance, i.e. its accuracy in recognizing the MNIST digits, is still disappointing. 
Further fine-tuning is required, e.g. using a dynamic learning rate.
I experimented with different parameters and algorithm changes which helped pushing accuracy to above 90% (but did not implement them in the published code). 

Yet, instead of spending much more time on fine tuning this rather simple feed forward network to further improve its performance on MNIST I'd rather want to move on to using a convolutional network next. ;)


---

## Code & Documentation

You can find all the code for this exercise on my [Github project page](https://github.com/mmlind/mnist-3lnn/), including full [code documentation](https://rawgit.com/mmlind/mnist-3lnn/master/doc/html/index.html).

When I run it on my 2010 MacBook Pro, using 784 input nodes, 20 hidden nodes and 10 output nodes, it takes about 19 seconds to process all 70,000 images (with the image rendering turned-off) and achieves an accuracy on the testing set of 89%.

Happy Hacking!

![_config.yml]({{ site.baseurl }}/images/mnist_logo.png)

