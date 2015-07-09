---
layout: post
title: What is a Neural Network?
---

When people talk about artificial intelligence and machine learning, they most often refer to neural networks (NN). 
Let's explore some ML basics, without excessive math, purely from a programmer's perspective.

![_config.yml]({{ site.baseurl }}/images/neuralnetwork.jpg)

A neural network is a concept modeled after the brain where individual units (neurons) form an organism (a network) to process information. 
In order to understand what this means and how this is special, let's start by looking at how a computer *normally* works.  

## Conventional Computing

Conventional computers use an algorithmic approach to solve a problem. 
They are provided with a set of instructions (the program) which they follow step by step to reach the objective.

![_config.yml]({{ site.baseurl }}/images/traditionalcomputing.svg)

Therefore, they can only solve what we already know. 
Somebody must first design all the required steps to solve the problem and thus knows exactly what the result will be.

Conventional computers/programs are **deterministic** (always lead to the same result) and **predictable** (the result is known in advance). 

## Neural Networks

A neural network uses a different approach to solve a problem. 
Instead of *calculating* the result, it's *guessing* it! 
Instead of providing the computer with an exact method how to calculate the output, we're providing it with data (lots of it) and let it start *guessing*. 

Every time the neural network guesses the output, we're letting it know whether its prediction (guess) was correct or not. 
If it was incorrect, we're telling it know by how much the prediction was off the desired result (target), so the network can adjust accordingly and try a slightly different guess next time.
This process is repeated a largue number of times. 
With every *guess* the networks's predicted result will move closer to the actual desired one. 

Please note that this is a **very simplistic** explanation and only describes one type of neural network and machine learning. 
But in principle this is how it works. 
You noticed that using this method we still need to know the correct answer (that's why this is called **supervised learning**), at least for a (large) number of sample cases.
The network is trained, under our supervision, on known data. 
Eventually it figures out an underlying *rule* for how to get from input to output and will be able to predict the correct results even for unknown data and without receiving any feedback.

Of course, even after training any prediction is still, strictly speaking, *guessing* (or more scientific, *approximating*) the solution.
Therefore one cannot be 100% sure that its prediction is correct. 
There is always a certain error rate (those cases where the neural network *guessed* wrong) and ML experts use this measure to compare the performance of each other's networks.
 
So, now let's look how this works in detail.

## The Perceptron

`error`

I mentioned above that a neural network consists of individual units (in the brain these units are called *neurons*). 
In the computer and machine learning world we refer to this unit as a *perceptron*, a computational model of a neuron.

![_config.yml]({{ site.baseurl }}/images/perceptron.svg)

The *perceptron* consists of a cell which is linked/connected to at least two inputs an one output.
It receives inputs, does some processing and spits out an output. 

*Processing*? Let's look at what's going on inside the perceptron. It's very simple. 

The connections on which the inputs are fed into the cell are assigned different weights. 
These weights are values between 0 to 1 and can be thought of as an expression of *priority* or *importance* of the respective input. 
In the beginning, ie. for our first *guess*, we use random numbers for these weights.

The perceptron's output is simply the sum of the weighted inputs.

```
output = (input0 * weight0) + (input1 * weight1)

```

This perceptron's `output` is the neural network's *guess* for the desired result.
In the scenario of *supervised learning*, as described above, this output is then compared to the desired output or `target` and the difference between both (error) is calculated.

```
error = target - output

```

Now the network not only knows that its *guess* was wrong but also by how much (`error`).
We now want the network to try/guess again, but of course not using the same values.
The `input` is a given, so the only values we can change are the `weights` of these inputs.

```
weight0 += input0 * error * learning_rate
weight1 += input1 * error * learning_rate

```

The `learning_rate` defines the interval for changing the weights, ie. high rates mean faster but likely less accurate change, low rates mean more accurate but slower change.

Above outlines the basics of the most simple neural network one can think of. 
There is no hidden layer, no sigmoid function, no back propagation, or any other mechanism neural networks usually have.
Yet, it still works.
To put this to a test, let's start coding. A few lines of code typically make things a lot clearer than pages of detailed explanations (at least for the programmer's mind).
In my next post we'll apply the above to the problem of automatic recognition of handwritten digits. 
