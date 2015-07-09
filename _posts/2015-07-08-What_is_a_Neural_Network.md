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
Instead of providing the computer with an exact method how to calculate the output, we're providing it with data (lots of it) and let it start "guess". 

Every time the neural network guessed the output, we're letting it know whether its prediction (guess) was correct or incorrect. 
If it was incorrect, we're telling it by how much the prediction was off the desired result (target), so the network can adjust accordingly and try a slightly different guess next time.
This process is repeated a largue number of times. 
With every *guess* the networks's predicted result will move closer to the actual desired one. 

Please note that this is a **very simplistic** explanation of only one type of neural network and machine learning. 
But in principle this is how it works. 
You noticed that using this method we still need to know the correct answer, at least for a (large) number of sample cases.
The network is trained on these cases and eventually figures out an underlying *rule* for how to get from input to output.
Thereby it will be able to later (after training) to predict the correct results even for unknown inputs and without receiving feedback about correctness.

So, now let's look how this works in detail.

## The Perceptron





## The World is binary!











