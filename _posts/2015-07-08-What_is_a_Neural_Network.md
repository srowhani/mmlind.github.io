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
Instead of *calculating* the result, it tries to first *guess* and then *remember* it. 
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

Of course, since even after training any prediction is still *guessing* (or more scientific, *approximating*) the solution and therefore we cannot be 100% sure that it is correct. 
There is always a certain error rate (those cases where the neural network *guessed* wrong) and ML experts use this measure to compare the performance of each other's networks.
 
So, now let's look how this works in detail.

## The Perceptron

I mentioned above that a neural network consists of individual units (in the brain these are called neurons).
In the computer world we refer to this units as a *perceptron* and for our coding purposes we will simply call this basic unit of a neural network a *cell*.

![_config.yml]({{ site.baseurl }}/images/perceptron.svg)



## The World is binary!











