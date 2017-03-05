---
layout: post
title: Linear Regression
mathjax: true

---

Linear regression is a statistical method for modeling the relationship between different variables. 
It's often used for analyzing dependencies and predicting values based on a given data set.

Regression represents one of the cornerstones of machine learning. 
Hence, understanding its underlying logic and math provides a solid foundation and a good preparation for exploring more advanced techniques such as neural networks.

![_config.yml]({{ site.baseurl }}/images/linreg_blackboard.jpg)

The underlying idea of linear regression is simple: Given a data set, we want to find the dependency of one of the variables in the data set on one, some or all of the other variables.

The dependency of $y$ on $x$ is learned from a dataset which includes many examples of both the explanatory variables and the respective "to-be-predicted" target variable $y$. 
Because the computer is learning the dependency based on the correct examples provided in the dataset, this type of learning is called __supervised learning__ because by providing correct examples of $y$ we're _supervising_ the learning algorithm.

We notate the dependent or target variable as $y$, and the independent or explanatory variables as $x$. 
So what we're looking for is the function

$y = f(x)$

that when given a value for $x$ outputs or predicts the desired target variable $y$. 
We call this a __model__. We're modeling the relationship between $x$ and $y$ or, more specifically, the dependency of $y$ on $x$.


###Types of Linear Regression

Depending on the number of independent or explanatory variables $x$, we distinguish between the following types of linear regression:

- If there is a single variable $x$: __uni-variate__ linear regression
- If there are multiple $x$ variables: __multi-variate__ linear regression
- If the $x$ variables include exponentials: __polynomial__ linear regression

Underlying math and method are the same for all three. 


##It's in the Data, Dude

Machine learning is the science of making computers learn from data instead of explicitly program them.
At the center of any machine learning excercise is thus the dataset that is to be analyzed. 

Nowadays, there are abundant sources of publicly available datasets, for example the

Machine Learning Repository at the University of California, Irvine
http://archive.ics.uci.edu/ml/datasets.html

or Kaggle, the online platform for data scientists.

Kaggle Datasets
https://www.kaggle.com/datasets

For this blog post I'm going to use the ["House Sales in King County, USA"](https://www.kaggle.com/harlfoxem/housesalesprediction "House Sales in King County, USA") dataset from Kaggle.


###Reading a Dataset

Most datasets are provided as a CSV file which normally includes a header line describing the variables captured.


Test Latex Math, inline: $ \sum\limits_{i=1}^n x^{i}$

$$ \sum\limits_{i=1}^n x^{i}$$

In a previous blog post ... 
