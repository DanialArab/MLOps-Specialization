# MLOps Specialization


1. [Introduction to Machine Learning in Production](#1)
2. [Machine Learning Data Lifecycle in Production](#2)
3. [Machine Learning Modeling Pipelines in Production](#3)
4. [Deploying Machine Learning Models in Production](#4) 


<a name="1"></a>
## Introduction to Machine Learning in Production

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/deployment%20example.PNG)

+ ML Model code is almost 5 to 10 % of the whole project code
+ This is called POC (proof of concept (developing in Jupyter Notebook)) to production gap

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/ML%20infrastructure.PNG)

ML project lifecycle

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/ML%20Project%20lifecycle.PNG)

**Data drift and concept drift**

Data drift, also known as feature drift or instance drift, occurs when the statistical properties of the input data used to train a machine learning model change over time. Concept drift, on the other hand, refers to the situation where the relationships between input features and the target variable (the concept or concept of interest) change over time. 

Sometimes the terminology of how to describe these data changes is not used completely consistently, but sometimes the term data drift is used to describe if the input distribution x changes, such as if a new politician or celebrity suddenly becomes well known and he's mentioned much more than before. The term concept drift refers to if the desired mapping from x to y changes such as if, before COVID-19. Perhaps for a given user, a lot of surprising online purchases, should have flagged that account for fraud. After the start of COVID-19, maybe those same purchases, would not have really been any cause for alarm, in terms of flagging that the credit card may have been stolen. Another example of Concept drift, let's say that x is the size of a house, and y is the price of a house, because you're trying to estimate housing prices. If because of inflation or changes in the market, houses may become more expensive over time. The same size house, will end up with a higher price. That would be Concept drift. Maybe the size of houses haven't changed, but the price of a given house changes. Whereas data drift would be if, say, people start building larger houses, or start building smaller houses and thus the input distribution of the sizes of houses actually changes over time


**Data change can be a gradual change or a sudden shock**

When data changes, sometimes it is a gradual change, such as the English language which does change, but changes very slowly with new vocabulary introduced at a relatively slow rate. Sometimes data changes very suddenly where there's a sudden shock to a system. For example, when COVID-19 the pandemic hit, a lot of **credit card fraud** started to not work because the **purchase patterns of individuals suddenly changed**. Many people that did relatively little online shopping suddenly started to use much more online shopping. So the way that people were using credit cards changed very suddenly, and this actually tripped up a lot of anti-fraud systems. This very sudden shift to the data distribution meant that many machine learning teams were scrambling a little bit at the start of COVID to collect new data and retrain systems in order to make them adapt to this very new data distribution. 

**Software Engineering issues**

In addition to managing these changes to the data, a second set of issues, that you will have to manage to deploy a system successfully, are software engineering issues:

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/SF%20ENg.%20issues.PNG)

**Deployment patterns**

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/common%20deployment%20cases.PNG)

+ Shadow deployment
+ Canary deployment
+ Blue green deployment

Shadow mode deployment:

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/shadow%20mode%20deployment.PNG)


**Degrees of automation**



References

https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops
