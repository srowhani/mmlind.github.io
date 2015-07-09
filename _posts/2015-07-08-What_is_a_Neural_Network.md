---
layout: post
title: What is a Neural Network?
---

When people talk about artificial intelligence and machine learning, they most often refer to neural networks (NN). 
Let's explore some basics, without excessive math, purely from a programmer's perspective.

![_config.yml]({{ site.baseurl }}/images/perceptron.svg)

A neural network is a concept modeled after the brain where individual units (neurons) form an organism (a network) to process information and to do "calculations". In order to understand how this is special, let's look at how a computer normally works.  

Conventional computers use an algorithmic approach to solve a problem. They are provided with a set of instructions (the program) which they follow step by step to reach the objective.

Therefore, they can only solve what we already know. Because somebody must first design all the required steps to solve the problem, thereby knowing exactly what the result will be.

Hence, conventional computers or programs are deterministic (always lead to the same result) and predictable (we know the result in advance). 


A neural network uses a different approach to solve a problem. Instead of "calculating" the output/solution, it tries to "remember" it. 
So, instead of providing it with a method how to calculate the output, we're providing it with data (lots of it) and let it "guess" the solution. 
And every time the NN guessed or predicted the output, we're letting it know whether that prediction was correct or incorrect. 
If it was incorrect, we're telling it how by much it was incorrect and it will adjust accordingly and try a slightly different guess.
With every "guess" it will move closer to the desired output. 
Therefore, in order to be effective, it needs lots of data that it can "train" itself on. 

