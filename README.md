# MLOps Specialization


1. [Introduction to Machine Learning in Production](#1)
   1. [Overview of the ML lifecycle and deployment](#2)
      1. [ML project lifecycle](#3)
      2. [Data drift and concept drift](#4)
      3. [Deployment patterns](#5)
      4. [Degrees of automation](#6)
      5. [Monitoring](#7)
   2. [Select and train a model](#8)
      1. [] 
3. [Machine Learning Data Lifecycle in Production](#2)
4. [Machine Learning Modeling Pipelines in Production](#3)
5. [Deploying Machine Learning Models in Production](#4) 


<a name="1"></a>
# Introduction to Machine Learning in Production

<a name="2"></a>
## Overview of the ML lifecycle and deployment

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/deployment%20example.PNG)

+ ML Model code is almost 5 to 10 % of the whole project code
+ This is called POC (proof of concept (developing in Jupyter Notebook)) to production gap

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/ML%20infrastructure.PNG)

<a name="3"></a>
### ML project lifecycle

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/ML%20Project%20lifecycle.PNG)

<a name="4"></a>
### Data drift and concept drift

Data drift, also known as feature drift or instance drift, occurs when the statistical properties of the input data used to train a machine learning model change over time. Concept drift, on the other hand, refers to the situation where the relationships between input features and the target variable (the concept or concept of interest) change over time. 

Sometimes the terminology of how to describe these data changes is not used completely consistently, but sometimes the term data drift is used to describe if the input distribution x changes, such as if a new politician or celebrity suddenly becomes well known and he's mentioned much more than before. The term concept drift refers to if the desired mapping from x to y changes such as if, before COVID-19. Perhaps for a given user, a lot of surprising online purchases, should have flagged that account for fraud. After the start of COVID-19, maybe those same purchases, would not have really been any cause for alarm, in terms of flagging that the credit card may have been stolen. Another example of Concept drift, let's say that x is the size of a house, and y is the price of a house, because you're trying to estimate housing prices. If because of inflation or changes in the market, houses may become more expensive over time. The same size house, will end up with a higher price. That would be Concept drift. Maybe the size of houses haven't changed, but the price of a given house changes. Whereas data drift would be if, say, people start building larger houses, or start building smaller houses and thus the input distribution of the sizes of houses actually changes over time


**Data change can be a gradual change or a sudden shock**

When data changes, sometimes it is a gradual change, such as the English language which does change, but changes very slowly with new vocabulary introduced at a relatively slow rate. Sometimes data changes very suddenly where there's a sudden shock to a system. For example, when COVID-19 the pandemic hit, a lot of **credit card fraud** started to not work because the **purchase patterns of individuals suddenly changed**. Many people that did relatively little online shopping suddenly started to use much more online shopping. So the way that people were using credit cards changed very suddenly, and this actually tripped up a lot of anti-fraud systems. This very sudden shift to the data distribution meant that many machine learning teams were scrambling a little bit at the start of COVID to collect new data and retrain systems in order to make them adapt to this very new data distribution. 

**Software Engineering issues**

In addition to managing these changes to the data, a second set of issues, that you will have to manage to deploy a system successfully, are software engineering issues:

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/SF%20ENg.%20issues.PNG)

<a name="5"></a>
### Deployment patterns

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/common%20deployment%20cases.PNG)

+ Shadow deployment
+ Canary deployment
+ Blue-green deployment

**Shadow mode deployment:**

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/shadow%20mode%20deployment.PNG)

**Canary deployment**:

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/canary%20deployment.PNG)

**Blue-green deployment**:

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/blue%20green%20deployment.PNG)


<a name="6"></a>
### Degrees of automation

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/degrees%20of%20automation.PNG) 

<a name="7"></a>
### Monitoring

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/monitoring%20dashboard.PNG)

Examples of metrics to track:

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/examples%20of%20metrics%20to%20track.PNG)

**Deployment is also an iterative process like ML modeling**

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/deployment%20is%20also%20iterative%20process.PNG)

**Pipeline monitoring**

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/metrics%20to%20monitor.PNG)

<a name="8"></a>
## Select and train a model



References

https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops
