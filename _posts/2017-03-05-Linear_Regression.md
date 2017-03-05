---
layout: post
title: Understanding Linear Regression
mathjax: true

---

Linear regression is a statistical method for modeling the relationship between different variables. 
It's often used for analyzing dependencies and predicting values based on a given data set.

Regression represents one of the cornerstones of machine learning. 
Hence, understanding its underlying logic and math provides a solid foundation and a good preparation for learning more advanced techniques such as neural networks.

![_config.yml]({{ site.baseurl }}/images/linreg_blackboard.jpg)

The underlying idea of linear regression is simple: Given a data set, we want to find the dependency of one of the variables in the data set on one, some or all of the other variables.

We notate the dependent or target variable as $y$, and the independent or explanatory variables as $x$. 
So what we're looking for is the function

$y = f(x)$

that when given a value for $x$ outputs or predicts the desired target variable $y$. 
We call this a __model__. We're modeling the relationship between $x$ and $y$ or, more specifically, the dependency of $y$ on $x$.

The dependency of $y$ on $x$ is learned from the dataset which includes many _training examples_, where each training examples includes a set of explanatory variables $x$ and the respective "to-be-predicted" target variable $y$. 

Because the computer is learning the dependency based on the correct examples provided in the dataset, this type of learning is called __supervised learning__ because by providing correct examples of $y$ we're _supervising_ the learning algorithm.


### Types of Linear Regression

Depending on the number of independent or explanatory variables $x$, we distinguish between the following types of linear regression:

- If there is only a single variable $x$, we call the model a __uni-variate__ linear regression
- If there are multiple $x$ variables, we call the model a __multi-variate__ linear regression
- If the $x$ variables include exponentials, we call the model a __polynomial__ linear regression

The underlying math and method are the same for all three. 

![_config.yml]({{ site.baseurl }}/images/linreg_bigdata.jpg)


## It's in the Data, Dude

Machine learning is the science of making computers learn from data instead of explicitly programming them.
At the center of any machine learning excercise is thus the __dataset__ that is to be analyzed. 

There are abundant sources of publicly available datasets nowadays. For example from the 

- [Machine Learning Repository at the University of California, Irvine](http://archive.ics.uci.edu/ml/datasets.html) or 
- [Kaggle](https://www.kaggle.com/datasets) the online platform for data scientists.

Many more sources for public datasets can be found on [KDNuggets](http://www.kdnuggets.com/datasets/index.html).

For this blog post I'm going to use the ["House Sales in King County, USA"](https://www.kaggle.com/harlfoxem/housesalesprediction) dataset from Kaggle.


### Reading a Dataset

Most datasets are provided as a CSV file which normally includes a header line describing the variables captured.


### Don't be afraid of math


### Mathmatical Notation


## The Methodolody


### Hypothesis Function


### Cost Function


### Optimization Algorithm


### Gradient Descent


##Optimization Techniques


### Feature Scaling & Mean Normalization


### Normal Equation


### Vectorized Linear Regression


...
