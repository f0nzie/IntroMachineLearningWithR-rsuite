# An Introduction to Machine Learning with R

This introductory workshop on machine learning with R is aimed at
participants who are not experts in machine learning (introductory
material will be presented as part of the course), but have some
familiarity with scripting in general and R in particular. The
workshop will offer a hands-on overview of typical machine learning
applications in R, including unsupervised (clustering, such as
hierarchical and k-means clustering, and dimensionality reduction,
such as principal component analysis) and supervised methods (classification
and regression, such as k-nearest neighbour and linear regression). We will also address questions such as model selection using
cross-validation. The material has an important hands-on component and
readers should have a computer running R 3.4.1 or later.



## Objectives and pre-requisites

- The course aims at providing an accessible introduction to various
  machine learning methods and applications in R. The core of the
  courses focuses on unsupervised and supervised methods. 

- The course contains numerous exercises to provide numerous
  opportunities to apply the newly acquired material.
  
- Participants are expected to be familiar with the R syntax and basic
  plotting functionality.

- At the end of the course, the participants are anticipated to be
  able to apply what they have learnt, as well as feel confident
  enough to explore and apply new methods.

## Why R?

R is one of the major languages for data science. It provides
excellent visualisation features, which is essential to explore the
data before submitting it to any automated learning, as well as
assessing the results of the learning algorithm. Many R packages
for [machine learning](https://cran.r-project.org/) are available off
the shelf and many modern methods in statistical learning are
implemented in R as part of their development.

There are however other viable alternatives that benefit from similar
advantages. If we consider Python for example,
the [scikit-learn](http://scikit-learn.org/stable/index.html) library
provides all the tools that we will discuss in this course.

## Overview of machine learning (ML)

In **supervised learning** (SML), the learning algorithm is presented
with labelled example inputs, where the labels indicate the desired
output. SML itself is composed of **classification**, where the output
is categorical, and **regression**, where the output is numerical.

In **unsupervised learning** (UML), no labels are provided, and the
learning algorithm focuses solely on detecting structure in unlabelled
input data.

Note that there are also **semi-supervised learning** approaches that
use labelled data to inform unsupervised learning on the unlabelled
data to identify and annotate new classes in the dataset (also called
novelty detection). 

**Reinforcement learning**, the learning algorithm performs a task
using feedback from operating in a real or synthetic environment.

## Material and methods

### Example data

- *Observations*, *examples* or simply *data points* along the rows
- *Features* or *variables* along the columns

Using the *iris* data as an example, for UML, we would have 4 features
for each unlabelled example.


 Sepal.Length   Sepal.Width   Petal.Length   Petal.Width
-------------  ------------  -------------  ------------
          5.1           3.5            1.4           0.2
          4.9           3.0            1.4           0.2
          4.7           3.2            1.3           0.2
          4.6           3.1            1.5           0.2
          5.0           3.6            1.4           0.2
          5.4           3.9            1.7           0.4

The same dataset used in the context of SML contains an additional
column of labels, documenting the outcome or class of each example.


Species    Sepal.Length   Sepal.Width   Petal.Length   Petal.Width
--------  -------------  ------------  -------------  ------------
setosa              5.1           3.5            1.4           0.2
setosa              4.9           3.0            1.4           0.2
setosa              4.7           3.2            1.3           0.2
setosa              4.6           3.1            1.5           0.2
setosa              5.0           3.6            1.4           0.2
setosa              5.4           3.9            1.7           0.4

The different datasets that are used throughout the course are
collected and briefly described in the short *Data* chapter.

### Packages


```r
library(BiocStyle)
```


We will be using, directly or indirectly, the following packages
through the chapters:

- *[caret](https://CRAN.R-project.org/package=caret)*
- *[ggplot2](https://CRAN.R-project.org/package=ggplot2)*
- *[mlbench](https://CRAN.R-project.org/package=mlbench)*
- *[class](https://CRAN.R-project.org/package=class)*
- *[caTools](https://CRAN.R-project.org/package=caTools)*
- *[randomForest](https://CRAN.R-project.org/package=randomForest)*
- *[impute](https://CRAN.R-project.org/package=impute)*
- *[ranger](https://CRAN.R-project.org/package=ranger)*
- *[kernlab](https://CRAN.R-project.org/package=kernlab)*
- *[class](https://CRAN.R-project.org/package=class)*
- *[glmnet](https://CRAN.R-project.org/package=glmnet)*
- *[naivebayes](https://CRAN.R-project.org/package=naivebayes)*
- *[rpart](https://CRAN.R-project.org/package=rpart)*
- *[rpart.plot](https://CRAN.R-project.org/package=rpart.plot)*

See the full session information for more details.

A more comprehensive list of machine learning libraries in R can be found at the [CRAN Task View for Machine Learning and Statistical Learning](https://cran.r-project.org/web/views/MachineLearning.html).
