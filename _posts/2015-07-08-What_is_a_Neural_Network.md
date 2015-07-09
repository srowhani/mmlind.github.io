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

Consequently, conventional computers/programs are **deterministic** (always lead to the same result) and **predictable** (the result is known in advance). 

## Neural Networks

A neural network uses a different approach to solve a problem. Instead of *calculating* the result, it tries to *remember* it. 
So, instead of providing the computer with an exact method how to calculate the output, we're providing it with data (lots of it) and let it "guess" the result. 

And every time the NN guessed the output, we're letting it know whether that prediction was correct or incorrect. 
If it was incorrect, we're telling it by how much it was incorrect, so it can adjust accordingly and try a slightly different guess.
With every *guess* it will move closer to the desired output. 

Via numerous guesses the NN will figure out the underlying rule how to get from input to output. 


The World is binary!

