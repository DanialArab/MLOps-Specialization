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

Data drift and concept drift

Data drift, also known as feature drift or instance drift, occurs when the statistical properties of the input data used to train a machine learning model change over time. Concept drift, on the other hand, refers to the situation where the relationships between input features and the target variable (the concept or concept of interest) change over time.

Data change can be a gradual change or a sudden shock 

When data changes, sometimes it is a gradual change, such as the English language which does change, but changes very slowly with new vocabulary introduced at a relatively slow rate. Sometimes data changes very suddenly where there's a sudden shock to a system. For example, when COVID-19 the pandemic hit, a lot of **credit card fraud ** started to not work because the purchase patterns of individuals suddenly changed. Many people that did relatively little online shopping suddenly started to use much more online shopping. So the way that people were using credit cards changed very suddenly, and his actually tripped up a lot of anti fraud systems. This very sudden shift to the data distribution meant that many machine learning teams were scrambling a little bit at the start of COVID to collect new data and retrain systems in order to make them adapt to this very new data distribution. Sometimes the terminology of how to describe these data changes is not used completely consistently, but sometimes the term data drift is used to describe if the input distribution x changes, such as if a new politician or celebrity suddenly becomes well known and he's mentioned much more than before. The term concept drift refers to if the desired mapping. From x to y changes such as if, before COVID-19. Perhaps for a given user, a lot of surprising online purchases, should have flagged that account for fraud. After the start of COVID-19, maybe those same purchases, would not have really been any cause for alarm, in terms of flagging. That the credit card may have been stolen. 

References

https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops
