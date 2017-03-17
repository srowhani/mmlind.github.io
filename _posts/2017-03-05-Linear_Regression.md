---
layout: post
title: Understanding Linear Regression
mathjax: true

---

Regression represents one of the cornerstones of machine learning. 
Comprehending its logic and math provides a solid foundation for learning more advanced machine learning techniques such as neural networks.

![_config.yml]({{ site.baseurl }}/images/linreg_blackboard.jpg)

Linear regression is a statistical method for modeling the relationship between variables. 
It's used for analyzing dependencies and predicting values.

The underlying idea of linear regression is simple: given a dataset, we want to find the dependency of one of the variables in the dataset on one, some or all of the other variables.

If we notate the dependent or target variable as $y$, and the independent or explanatory variables as $x$, what we're looking for is the function

$y = f(x)$

that, given an arbitrary value for $x$, outputs the desired target variable $y$.
The output represents our _prediction_ for y. It's an estimation of the unknown actual value of $y$.

This function is also referred to as our __model__. We're modeling the relationship between $x$ and $y$ or, more specifically, the dependency of $y$ on $x$.

The dependency is learned from a __dataset__.
The dataset is simply a large collection of examples of _x_ recorded in the real world.
These examples are used to _train_ the computer to estimate the target function and hence called __training examples__.
 
Each training example may consist of just one or many explanatory variables $x$ plus their respective _"to-be-predicted"_ target variable $y$. 
It's typically a many-to-one relationship, i.e. many independent variables are mapped to a single dependent variable.

The objective of linear regression is to have the computer learn dependencies inside the data simply from looking at many examples. 
Each example must include the _correct_ or desired output for the respective inputs, also referred to as __the ground truth__. 

This type of learning is therefore called __supervised learning__. 
By providing correct examples of $y$ we're _supervising_ the learning algorithm.


## The Methodology

Linear regression, or regression analysis in general, consists of 3 steps: define a hypothesis function, define a cost function and run an optimization algorithm.

Before explaining these components in detail, let me briefly address a hurdle that many people face when learning regression or machine learning in general: the math.

If you don't have a strong math background a series of equations presented in a tutorial or paper may present an immediate turn-off. 
I know. Why is that so?

__Math__ is a language, __the language of nature__. Unfortunately, the average reader is not fluent in or familiar with this language.
Mathematical equations therefore easily get intimidating because they may include unknown signs or greek letters.

Let's go step by step through the math, and you'll see it's not that hard at all. 
Especially if you're a software developer and used to systematic thinking. 


### Mathematical Notation

A dataset consists of a number of training samples. Typically, the more the better. 
Let's define or notate the size of the dataset as $m$:

``m = number of training examples in the dataset``

Every training sample consists of a number of independent or explanatory values, called __features__.
Let's notate the number of such features per training example as $n$:

``n = number of features per training example``

Then let's pack all of these features of a single training example into a vector. 
In this context a vector can be regarded as simply a list or array of values.
Let's notate this vector as $x$: 

$x = [x_1, x_2, ... x_n]$

The subscript index denotes the j-th feature of a training example, where $j \in $ {1..n}.

But of course there are many training examples, not just a single one. 
Therefore, we will use another index, a superscript $i$, to denote a specific training example $i$, where $i \in $ {1..m}.
A specific training example, with all its features, is then denoted as vector $x^{(i)}$:

$x^{(i)} = $ n-dimensional vector of all features of the i-th training sample 

Combining both, the value of a specific feature of a specific training example is denoted as $x_{j}^{(i)}$:

$x_{j}^{(i)}$ = value of the $j^{th}$ feature in the $i^{th}$ training sample

So far so good. That wasn't so hard. 

Now, what's missing? To compute, or estimate, the target variable linear regression _weighs_ each individual feature differently and then adds up all the weighted features.

These weights are called the __parameters__ and are notated with the greek letter "Theta" $\theta$.

Every feature gets weighted differently. 
The number of parameters that we're trying to find is therefore the same as the number of features.

But we don't compute separate parameters for each training example.
What we want is a _single_ model, one function that covers _all_ of the training examples.
So let's pack the parameters into a vector as well and denote this vector as $\theta$:

$\theta = [\theta_1, \theta_2, ... , \theta_n]$

Great. That's all we need. Now, let's get started with the 3 steps methodology mentioned above.


![_config.yml]({{ site.baseurl }}/images/linreg_hypothesis.jpg)


### Hypothesis Function

Regression attempts to model an unknown function $h$ that maps the given inputs $x$ to an output value $y$.
Let's call this our __hypothesis function__ and denote it as

$h = f(x)$

Again, there could be a single input value x or many input values. (Remember, $x$ is a vector and this vector may have a size of 1 which would make $x$ a scalar.) 

Depending on the number of explanatory variables $x$, we distinguish 3 types of linear regression:

- If there is only a __single__ input variable or feature $x$, we call the model a __uni-variate__ linear regression:
$$h_{\theta}(x) = \theta_{0} + \theta_{1} x $$

- If there are __multiple__ input variables or features $x$, we call the model a __multi-variate__ linear regression:
$$h_{\theta}(x) = \theta_{0} + \theta_{1} x_{1} + \ ... \theta_{i} x_{i} \ ... + \ \theta_{n} x_{n}$$

- If the independent variables or features include __exponentials__, we call the model a __polynomial__ linear regression.
$$h_{\theta}(x) = \theta_{0} + \theta_{1} x_{1} + \theta_{2} x_{1}^2 + \theta_{3} x_{1}x_{2} + \theta_{4} x_{2}^2 \ ... $$

Math and methodology are the same for all three. 

You may have noticed that I added an additional parameter $\theta_0$.
The reason is that the function that we're trying to find may have a certain _base_ value that is completely independent from any of the features. 
This base is also called the __bias__.

Since there is no feature mapped or related to the bias, we set the corresponding 

$x_0 = 1$

Having added the bias parameter to our feature vector $x$ _and_ to our parameter vector $\theta$, both vectors become of size __n+1__.


![_config.yml]({{ site.baseurl }}/images/linreg_costfunction.jpg)


### Cost Function

After we defined the hypothesis function, we can feed data into this function and it will return a prediction or an estimate of what the output value y should be for a particular set of inputs. But how good is this value computed via our hypothesis?

We need to measure the __accuracy__ of our hypothesis. 
We do so by computing the difference of the hypothesis with the actual "true" target value from the dataset.
(Remember from above that in supervised learning we need to provide the target value for each training example.)

Since there are many training examples and not just a single one, we can measure the difference for each.
That's what a cost function does.

The difference of hypothesis and target is called the __error__ or __cost__ or __loss__.

Conceptually, there are many different options how to measure or express this error.
The most common method is to compute th __mean squared error__ (MSE) and is defined as follows: 

- compute the difference of hypothesis and correct output for a particular training example $i$, 
- square the difference, 
- sum the squared differences across all $m$ training examples, and then 
- divide the total by $m$ to get the mean (average) cost. 

Mathematically, we'll denote this kind of cost function $J$ as

$J(\theta_0, \theta_1, ..., \theta_n) = \frac{1}{2m} \sum\limits_{i=1}^{m} ( h_{\theta} (x^{(i)}) - y^{(i)} )^2$

and read it as "_J with respect to Theta 1, ... Theta n__" stating in brackets the variable(s) that we're minimizing for (i.e. Theta).

Next, we want to __minimize__ this cost function, i.e. find those parameters or those values of $\theta$ that result in the minimum J. 

Minimizing the cost is an __optimization problem__: we want to find the optimal representation of $h(x)$ which leads us to the third and last step in the methodology for linear regression. 

Before we do that, a short side note. You probably noticed that in above formula for J we're actually dividing by 2m instead of simply by m. 
This is done for mathematical convenience later (when computing the derivate). 
Since it doesn't impact the minimization it represents a valid simplification.

At this point it may also be worth noting that in addition to mean squared error there are other possible cost functions, e.g. one could compute the mean _absolute_ error or the average cross entropy. 


![_config.yml]({{ site.baseurl }}/images/linreg_optimization.jpg)


### Optimization Algorithm

To find the optimal hypothesis we need to minimize the error or cost. 
The optimal values for our parameters Theta will result in the lowest cost or lowest value for J.

There are different ways to do this. 
The most popular optimization algorithm is called gradient descent.

### Gradient Descent

The idea of gradient descent is simple:
- Compute the slope at an arbitrary point of the cost curve.
- If the slope is _bigger_ than 0, slightly _decrease_ the input values, i.e. pick a new point a little further to the _left_ on the cost curve.
- If the slope is _smaller_ than 0, we slightly _increase_ our input values, i.e. pick a new point a little further to the _right_ on the cost curve.
- Repeat these steps until the slope is so small that there is hardly any further reduction or improvement.

Let's see how to do this mathematically.

The slope of a curve is computed via the derivative of the underlying function. 
That's easy. But our function does not have one but _multiple_ features and the same number of parameters.
Hence we need to compute the __partial derivate__ for each of those parameters.

Gradient descent incrementally updates the _parameters_ in our cost function using two components: the gradient and a step size. 

The partial derivative ("delta") of the cost function J

$J(\theta_0, \theta_1, ..., \theta_n) = \frac{1}{2m} \sum\limits_{i=1}^{m} ( h_{\theta} (x^{(i)}) - y^{(i)} )^2$

with respect to parameter $\theta_j$ is called the __gradient__:

$\frac{\delta}{\delta \theta_j} J(\theta_0, \theta_1, ..., \theta_n) = \frac{1}{m} \sum\limits_{i=1}^{m} ( h_{\theta} (x^{(i)}) - y^{(i)} ) \ x_j^{(i)} $

This gradient is used to update or change the parameters. It will be subtracted at an arbitrary "step size" called the __learning rate__ alpha ($\alpha$).
So, we simultaneously update all parameters $\theta_j$ for j = {0..n}:

$\theta_j := \theta_j - \alpha \frac{\delta}{\delta \theta_j} J(\theta) = \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} ( h_{\theta} (x^{(i)}) - y^{(i)} ) \ x_j^{(i)} $

Gradient descent automatically takes smaller steps when it approaches the minimum because the slope gets flatter.

There are 3 types of gradient descent:
- __batch__ gradient descent = each step uses all training samples (=> computationally expensive)
- __stochastic__ gradient descent = each step uses only a single sample
- __mini batch__ gradient descent = each step uses an arbitrary mini-batch size of training samples


#### Batch Gradient Descent

Batch Gradient Descent considers __all__ training examples for each optimization step. It directly converges to the global minimum.


#### Stochastic Gradient Descent

Stochastic gradient descent considers only a __single__ training example at a time, therefore does not directly converge but zig-zags in on the global minimum. It does not actually reach the global minimum, but "oscillates" around it.

Looping through the whole dataset only once may not be sufficient to converge to a global optimum. In practice SGD is often run 1-10 times on a given dataset. 

SGD is a type of __online learning__ algorithm because it does not require a whole dataset at once but can process training examples one-by-one, i.e. learn _online_. 


#### Mini Batch Gradient Descent

Mini Batch Gradient Descent is a kind of hybrid version trying to combine the advantages of both batch and stochastic gradient descent. Instead of summing (or looping through) the whole dataset for each optimization step, the algorithm uses a __randomly selected subset__ of examples. 

Its main advantage is that it's computationally much less expensive. It also converges faster than stochastic gradient descent because it avoids (much of) the zig-zag. 

Given today's parallel computing capabilities mini-batch is especially performant if the algorithm is vectorized.


![_config.yml]({{ site.baseurl }}/images/linreg_bigdata.jpg)

Vectorization is one of the subjects that I will address in a later blog post, together with the following remaining points of interest for linear regression:
- Reading and Visualizing the Data
- Optimization Techniques
- Feature Engineering
- Feature Scaling & Mean Normalization
- Normal Equation
- Vectorized Linear Regression

