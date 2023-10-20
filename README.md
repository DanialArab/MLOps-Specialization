# MLOps Specialization

This repository documents my understanding of putting ML models into production. Also included in this repo are my notes and solutions to the assignments of the specialization <a href="https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops">Machine Learning Engineering for Production (MLOps) Specialization</a> taught by Robert Crowe, Laurence Moroney, Andrew Ng. This specialization includes 4 courses:
+ Introduction to Machine Learning in Production
+ Machine Learning Data Lifecycle in Production
+ Machine Learning Modeling Pipelines in Production
+ Deploying Machine Learning Models in Production 

1. [Course 1: Introduction to Machine Learning in Production](#1)
   1. [Overview of the ML lifecycle and deployment](#2)
      1. [ML project lifecycle](#3)
      2. [Data drift and concept drift](#4)
      3. [Deployment patterns](#5)
      4. [Degrees of automation](#6)
      5. [Monitoring](#7)
   2. [Select and train a model](#8)
      1. [Data vs. model-centric AI development](#9)
      2. [Challenges in model development](#10)
      3. [Why low average error isn't good enough](#11)
      4. [Establish a baseline](#12)
      5. [Ways to establish a baseline](#13)
   3. [Error analysis and performance auditing](#14)
      1. [Prioritizing what to work on next](#15)
      2. [Skewed datasets](#16)
      3. [Performance auditing](#17)
   4. [Data-centric AI development](#18)
2. [Machine Learning Data Lifecycle in Production](#2)
3. [Machine Learning Modeling Pipelines in Production](#3)
4. [Deploying Machine Learning Models in Production](#4) 


<a name="1"></a>
# Course 1: Introduction to Machine Learning in Production


<a name="2"></a>
## Overview of the ML lifecycle and deployment

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/deployment%20example.PNG)

+ ML Model code is almost 5 to 10 % of the whole project code
+ All the work and software required beyond the 5 to 10 % ML code is called POC (proof of concept (developing in Jupyter Notebook)) to production gap

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

<a name="9"></a>
### Data vs. model-centric AI development 

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/Modeling.PNG)

AI System = code + data

<a name="10"></a>
### Challenges in model development 

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/challenges%20in%20model%20development.PNG)

<a name="11"></a>
### Why low average error isn't good enough

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/performance%20on%20disproportionately%20important%20examples.PNG)

For **informational and transactional queries**, a web search engine wants to return the most relevant results, but users are willing to forgive maybe ranking the best result, number two or number three. There's a different type of web search query such as Stanford, or Reddit, or YouTube. These are called **navigational queries**, where the user has a very clear intent, very clear desire to go to Stanford.edu, or Reddit.com, or YouTube.com. When a user has a very clear navigational intent, they will tend to be very unforgiving if a web search engine does anything other than return Stanford.edu as the Number one ranked results and the search engine that doesn't give the right results will quickly lose the trust of its users. Navigational queries in this context are a disproportionately important set of examples and if you have a learning algorithm that improves your average test set accuracy for web search but messes up just a small handful of navigational queries, that may not be acceptable for deployment. The challenge, of course, is that average test set accuracy tends to weight all examples equally, whereas, in web search, some queries are disproportionately important. Now one thing you could do is try to give these examples a higher weight. That could work for some applications, but in my experience, just changing the weights of different examples doesn't always solve the entire problem. 

Closely related to this is the question of performance on key slices of the data set. For example, let's say you've built a machine learning algorithm for loan approval to decide who is likely to repay a loan and thus to recommend approving certain loans for approval. For such a system, you will probably want to make sure that your system does not unfairly discriminate against loan applicants according to their ethnicity, gender, maybe their location, their language, or other protected attributes. Many countries also have laws or regulations that mandates that financial systems and loan approval processes not discriminate on the basis of a certain set of attributes, sometimes called protected attributes. Even if a learning algorithm for loan approval achieves high average test set accuracy, it would not be acceptable for production deployment if it exhibits an unacceptable level of bias or discrimination. 

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/performance%20on%20key%20slices%20of%20the%20data.PNG)

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/rare%20classes.PNG)

<a name="12"></a>
### Establish a baseline

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/establishing%20a%20baseline.PNG)

It turns out the best practices for establishing a baseline are quite different, depending on whether you're working on unstructured or structured data:

Because humans are so good at unstructured data tasks, measuring human-level performance or HLP, is often a good way to establish a baseline if you are working on unstructured data.

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/unstructured%20and%20structured%20data.PNG)

<a name="13"></a>
### Ways to establish a baseline

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/ways%20to%20establish%20a%20baseline.PNG)

The **Bayes error**, also known as the Bayes risk or the **irreducible error**, is a fundamental concept in statistics and machine learning. It represents the lowest possible error rate that any classifier or predictive model can achieve for a given problem, assuming knowledge of the true underlying probability distribution of the data. In other words, it's the minimum achievable error rate for a specific classification or prediction task.

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/getting%20started.PNG)

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/deployment%20constraints.PNG)

Finally, when trying out a learning algorithm for the first time, before running it on all your data, I would urge you to run a few quick sanity checks for your code and your algorithm. For example, I will usually try to overfit a very small training dataset before spending hours or sometimes even overnight or days training the algorithm on a large dataset. **Maybe even try to make sure you can fit one training example, especially, if the output is a complex output.**

The advantage of this is you may be able to train your algorithm on one or a small handful of examples in **just minutes or maybe even seconds and this lets you find bugs much more quickly.**

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/sanity%20check%20the%20code.PNG)

<a name="14"></a>
## Error analysis and performance auditing

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/iterative%20process%20of%20error%20analysis.PNG)

<a name="15"></a>
### Prioritizing what to work on next 

Rather than deciding to work on car noise because the gap to HLP is bigger, one other useful factor to look at is what's the percentage of data with that tag? 

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/prioritizing%20what%20to%20work%20on%20next.PNG)

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/prioritizing%20what%20to%20work.PNG)

<a name="16"></a>
### Skewed datasets

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/multi%20class%20classification.PNG)

<a name="17"></a>
### Performance auditing

Even when your learning algorithm is doing well on accuracy or F1 score or some appropriate metric. It's often worth one last performance audit before you push it to production. And this can sometimes save you from significant post deployment problems

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/performance%20auditng.PNG)

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/auditing%20framework.PNG)


<a name="18"></a>
## Data-centric AI development

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/data%20centric%20vs%20model%20centric%20AI.PNG)

One of the most important ways to improve the quality of a data set is data augmentation.

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/data%20augmentation.PNG) 

It turns out that for unstructured data problems, pulling up one piece of this rubber sheet is unlikely to cause a different piece of the rubber sheet to dip down really far below. Instead, pulling up one point causes nearby points to be pulled up quite a lot and far away points may be pulled up a little bit, or if you're lucky, maybe more than a little bit. 

Data augmentation

Data augmentation can be a very efficient way to get more data, especially for unstructured data problems such as images, audio, maybe text. But when carrying out data augmentation, there're a lot of choices you have to make. What are the parameters? How do you design the data augmentation setup? Let's dive into this to look at some best practices.

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/data%20augmentation%20guidelines.PNG)

Can adding data hurt?

Usually, for unstructured data performance, the answer is no, with some caveats, but let's dive more deeply into this: 

HERE 



