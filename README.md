# MLOps Specialization

This repository documents my understanding of putting ML models into production. Also included in this repo are my notes and solutions to the assignments of the specialization <a href="https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops">Machine Learning Engineering for Production (MLOps) Specialization</a> taught by Robert Crowe, Laurence Moroney, Andrew Ng. This specialization includes 4 courses:
+ Introduction to Machine Learning in Production
+ Machine Learning Data Lifecycle in Production
+ Machine Learning Modeling Pipelines in Production
+ Deploying Machine Learning Models in Production 

1. [Course 1: Introduction to Machine Learning in Production](#1)
   1. [Overview of the ML lifecycle](#2)
      1. [ML project lifecycle](#3)
   2. [Deployment](#4)
      1. [Key challenges in ML deployment](#5)
         1. [Data drift and concept drift](#6)
         2. [Software Engineering issues](#7)
      2. [Common deployment cases](#8)
      3. [Deployment patterns](#9)
         1. [Shadow mode deployment](#10)
         2. [Canary deployment](#11)
         3. [Blue-green deployment](#12) 
         4. [Degrees of automation](#13)
      4. [Monitoring](#14)
      5. [Pipeline monitoring](#15)
   3. [Select and train a model](#8)
      1. [Data vs. model-centric AI development](#9)
      2. [Challenges in model development](#10)
      3. [Why low average error isn't good enough](#11)
      4. [Establish a baseline](#12)
      5. [Ways to establish a baseline](#13)
   5. [Error analysis and performance auditing](#14)
      1. [Prioritizing what to work on next](#15)
      2. [Skewed datasets](#16)
      3. [Performance auditing](#17)
   6. [Data-centric AI development](#18)
2. [Machine Learning Data Lifecycle in Production](#2)
3. [Machine Learning Modeling Pipelines in Production](#3)
4. [Deploying Machine Learning Models in Production](#4) 


<a name="1"></a>
# Course 1: Introduction to Machine Learning in Production

<a name="2"></a>
## Overview of the ML lifecycle and deployment

As an example of putting an ML model into production, let's consider a cellphone factory. Our ML problem is to perform a visual inspection of the cellphones produced by the factory and make a decision on whether or not the device is defective. We do have an ML model for that which is put in the **prediction server**. We also have an edge device (like a mobile phone) where the inspection software lives. It captures a photo of the cellphone in the edge device then this image is sent to the prediction server through an API call, there the ML model makes a prediction on that image and then sends back the prediction to the edge device through another API call. And then the inspection software can make the appropriate control decision on whether to let it still move on in the manufacturing line. Or whether to shove it to a side, because it was defective and not acceptable. We can put the prediction server in the cloud or also at the edge device (for the factories it is recommended to put the prediction server on the edge device (**edge deployment**) to make sure that if the internet is down it does not make the factory down).

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/deployment%20example.PNG)

+ ML Model code is almost 5 to 10 % of the whole project code
+ All the work and software required beyond the 5 to 10 % ML code is called POC (**proof of concept (developing in Jupyter Notebook)**) to production gap

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/ML%20infrastructure.PNG)

<a name="3"></a>
### ML project lifecycle

Some considerations on different phases of the ML lifecycle:
+ Scoping questions:
   + Decide to work on what
   + Decide on key metrics
      + accuracy,
      + latency,
      + throughput
   + Estimate resources and timeline
+ Data definition questions:
   + Is the data labeled **consistently**? Otherwise, the learning algorithm would be confused! 
   + As an example let's say for the speech recognition problem, how much silence do we want to have before/after each clip?
   + Again for the speech recognition problem, how to perform volume normalization? 
+ Modeling (Code + data) 
   + code (algorithm/model)
   + hyperparameters
   + data 

In research work or academics, we tend to hold the data fixed and vary the code and may vary the hyperparameters in order to try to get good performance. In contrast, in the product teams, if your main goal is to just build and deploy a working valuable machine learning system, it is even more effective to hold the code fixed and instead focus on optimizing the data and maybe the hyperparameters,
+ Deployment, monitoring, and maintaining the system

As an example, a speech recognition system that is trained mainly on adult voices would not have a good performance on the data for the younger individuals. The reason is that the voices of very young individuals just sound different. In this case, we need to go back and find a way to collect more data to be fed into the training dataset and retrain the model in order to fix it. So one of the key challenges when it comes to deployment is concept drift or data drift, which is what happens when the data distribution changes.

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/ML%20Project%20lifecycle.PNG)

+ As part of error analysis before taking a system to deployments, I'll often also carry out a final check, maybe a final audit, to make sure that the system's performance is good enough and that it's sufficiently reliable for the application.
+ If the data distribution in the upcoming traffic changes, you may need to update the model. After the initial deployment, maintenance will often mean going back to perform more error analysis and maybe retrain the model, or it might mean taking the data you get back. Now that the system is deployed and is running on live data, and feeding that back into your dataset to then potentially update your data, retrain the model, and so on until you can put an updated model into deployment.

<a name="4"></a>
## Deployment 

<a name="5"></a>
### Key challenges in ML deployment

Key challenges in ML deployment are:
+ Statistical or ML issues like concept drift or data drift 
+ Software issues

<a name="6"></a>
#### Data drift and concept drift

Data drift, also known as feature drift or instance drift, occurs when the statistical properties of the input data used to train a machine learning model change over time. Concept drift, on the other hand, refers to the situation where the relationships between input features and the target variable (the concept or concept of interest) change over time. 

The term data drift is used to describe if the input distribution x changes. The term concept drift refers to if the desired mapping from x to y changes such as if, before COVID-19. Perhaps for a given user, a lot of surprising online purchases should have flagged that account for fraud. After the start of COVID-19, maybe those same purchases, would not have really been any cause for alarm, in terms of flagging that the credit card may have been stolen. Another example of Concept drift, let's say that x is the size of a house, and y is the price of a house because you're trying to estimate housing prices. If because of inflation or changes in the market, houses may become more expensive over time. The same size house will end up with a higher price. That would be Concept drift. Maybe the size of houses haven't changed, but the price of a given house changes. Whereas data drift would be if, say, people start building larger houses or start building smaller houses and thus the input distribution of the sizes of houses actually changes over time

**Data change can be a gradual change or a sudden shock**

When data changes, sometimes it is a gradual change, such as the English language which does change, but changes very slowly with new vocabulary introduced at a relatively slow rate. Sometimes data changes very suddenly where there's a sudden shock to a system. For example, when COVID-19 the pandemic hit, a lot of **credit card fraud** started to not work because the **purchase patterns of individuals suddenly changed**. Many people that did relatively little online shopping suddenly started to use much more online shopping. So the way that people were using credit cards changed very suddenly, and this actually tripped up a lot of anti-fraud systems. This very sudden shift to the data distribution meant that many machine learning teams were scrambling a little bit at the start of COVID to collect new data and retrain systems in order to make them adapt to this very new data distribution. 

<a name="7"></a>
#### Software Engineering issues

In addition to managing these changes to the data, a second set of issues, that you will have to manage to deploy a system successfully, are software engineering issues:

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/SF%20ENg.%20issues.PNG)

<a name="8"></a>
### Common deployment cases

Common deployment cases:
+ If you're offering a service that you have not offered before, a common design pattern is to start up a small amount of traffic and then gradually ramp it up.
+ If there's something that's already being done by a person, but we would now like to use a learning algorithm to either automate or assist with that task.
+ If you've already been doing this task with a previous implementation of a machine learning system, but you now want to replace it with hopefully an even better one. 

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/common%20deployment%20cases.PNG)

Key ideas:  In these cases, two recurring themes you see are that you often want a **gradual ramp-up with monitoring**. In other words, rather than sending tons of traffic to a maybe not fully proven learning algorithm, you may send it only a small amount of traffic and monitor it and then ramp up the percentage or amount of traffic. And the second idea is **rollback**. This means that if for some reason the algorithm isn't working, it's nice if you can revert back to the previous system if indeed there was an earlier system.

<a name="9"></a>
### Deployment patterns

+ Shadow deployment
+ Canary deployment
+ Blue-green deployment

<a name="10"></a>
#### Shadow mode deployment

When we have people initially doing a task, one common deployment pattern is to use shadow mode deployment. That means that you will start by having a machine-learning algorithm shadow the human inspector and run in parallel with the human inspector. During this initial phase, the learning algorithm output is not used for any decision in the factory. So whatever the learning algorithm says, we're going to go to the human judgment for now.

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/shadow%20mode%20deployment.PNG)

The purpose of a shadow mode deployment is that it allows you to gather data of how the learning algorithm is performing and how that compares to the human judgment. And by sampling the outputs you can then verify if the learning algorithm's predictions are accurate and therefore use that to decide whether or not to allow the learning algorithm to make some real decisions in the future. So when you already have some system that is making good decisions and that system can be human inspectors or even an older implementation of a learning algorithm, using a shadow mode deployment can be a very effective way to let you verify the performance of a learning algorithm before letting them make any real decisions. 

<a name="11"></a>
#### Canary deployment

When you are ready to let a learning algorithm start making real decisions, a common deployment pattern is to use a canary deployment. 

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/canary%20deployment.PNG)

By running the ML algorithm on only a small percentage of the traffic, hopefully, if the algorithm makes any mistakes it **will affect only a small fraction of the traffic**. And this gives you more of an opportunity to monitor the system and ramp up the percentage of traffic it gets only gradually and only when you have greater confidence in this performance.

<a name="12"></a>
#### Blue-green deployment

In a blue-green deployment, what you do is have the router send images to the old or the blue version and have that make decisions. And then when you want to switch over to the new version, what you would do is have the router stop sending images to the old one and suddenly switch over to the new version. So the way the blue-green deployment is implemented is you would have an old prediction service that may be running on some sort of service. You will then spin up a new prediction service, the green version, and you would have the router suddenly switch the traffic over from the old one to the new one. The advantage of a blue-green deployment is that there's an **easy way to enable rollback**. If something goes wrong, you can just very quickly have the router go back reconfigure their router to send traffic back to the old or the blue version, assuming that you kept your blue version of the prediction service running. In a typical implementation of a blue-green deployment, people think of switching over the traffic 100 % all at the same time. But of course, you can also use a more gradual version where you slowly send traffic over. 

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/blue%20green%20deployment.PNG)

<a name="13"></a>
### Degrees of automation

One of the most useful frameworks for thinking about how to deploy a system is to think about deployment not as a 0, 1 like either deploy or not deploy, but instead to design a system thinking about what is the appropriate degree of automation:

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/degrees%20of%20automation.PNG) 

<a name="14"></a>
### Monitoring

The most common way to monitor a machine learning system is to use a **dashboard to track how it is doing over time**. Depending on your application, your dashboards may monitor different metrics. For example, you may have one dashboard to monitor the server load, or a different dashboard to monitor the fraction of non-null outputs. Sometimes a speech recognition system output is null when the users didn't say anything. If this changes dramatically over time, it may be an indication that something is wrong, or one common one I've seen for a lot of structured data tasks is monitoring the fraction of missing input values. If that changes, it may mean that something has changed about your data.

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/monitoring%20dashboard.PNG)

Examples of metrics to track:

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/examples%20of%20metrics%20to%20track.PNG)

**Deployment is also an iterative process like ML modeling**

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/deployment%20is%20also%20iterative%20process.PNG)

It usually takes a few trials to converge to the right set of metrics to monitor. Sometimes you have deployed the machine learning system, and it's not uncommon for you to deploy a machine learning system with an initial set of metrics only to run the system for a few weeks and then to realize that something could go wrong with it that you hadn't thought of before and into pick a new metric to monitor. Or for you to have some metric that you monitor for a few weeks and then decide they're just metrics, hardly ever change, and to get rid of that metric in favor of focusing attention on something else.

After you've chosen a set of metrics to monitor, common practice would be to **set thresholds for alarms**. You may decide if the server load ever goes above 0.91, which may trigger an alarm or a notification to let you know or let the team know to see if there's a problem and maybe spin up some more servers:

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/monitoring%20dashboards.PNG) 

If something goes wrong with your learning algorithm, if it is a software issue such as server load is too high, then that may require changing the software implementation, or if it is a performance problem associated with the accuracy of the learning algorithm, then you may need to update your model. Or if it is an issue associated with the accuracy of the learning algorithm, then you may need to go back to fix that that's why many machine learning models will need a little bit of maintenance or retraining over time. Just like almost all software needs some level of maintenance as well.

When a model needs to be updated, you can either retrain it **manually**, where in Engineer will retrain the model perform error analysis on the new model and make sure it looks okay before pushing that to deployment. Or you could also put in place a system where there is **automatic retraining**. Today, **manual retraining is far more common than automatic training for many applications**, developers are reluctant to learning algorithm be fully automatic in terms of deciding to retrain and pushing new model to production, but there are some applications, especially in consumer software Internet, where automatically training does happen.

![](https://github.com/DanialArab/images/blob/main/MLOps-Specialization/model%20maintenance.PNG)

 **key takeaways**:
 
The key takeaways are that it is only by monitoring the system you can spot if there may be a problem that may cause you to go back to perform a deeper error analysis, or that may cause you to go back to get more data with which you can update your model so as to maintain or improve your system's performance.  

For more complex systems, where you don't have just one model and instead you have a more complex machine learning pipeline, how do you monitor the performance of that?

<a name="15"></a>
### Pipeline monitoring

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



