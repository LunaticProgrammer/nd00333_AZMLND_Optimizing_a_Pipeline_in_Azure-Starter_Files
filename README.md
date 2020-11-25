# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**The classification goal is to predict if the client will subscribe a term deposit (variable y)."**
<br>
**After trying Hyperparameter optimization using on Logistic Regression using HyperDrive gave an accuracy of 90.1%
  While when the dataset was fed to a AutoML it predicted a model which used**

## Scikit-learn Pipeline
The pipeline Architecture consisted of following steps:
1. Fetching dataset from the remote file server
2. Cleaning the dataset and converting non digit datapoints to digit based using encoders
3. Splitting the data into train and test 
4. Selecting model in our case it was Logistic Regression
5. Selecting Parameter Sampler which has two parameters --C and --Max_iter which are chosen randomly from the given field values (RandomParameterSampler)
6. Selecting Early stoping policy in out case it was BanditPolicy
7. Finally plugging all the configs, parameterSampler, early stopping policy into hyperDrive to get the model trained on various hyperParameters
8. Extracting the best run and hyperParameters and saving it into a pickle to use it later

**What are the benefits of the parameter sampler you chose?<br>
The Sampler which we chose was RandomSampler and it had a some or the other benefits over other samplers.
The most important factor of chosing a randomSampler was to give the pipeline to give a wild shot which can somehow give the best hyper-parameters
which other samplers might have missed or took more time to find it**

**We chose Bandit Policy for Early Stopping to keep the regularization.
it is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run which helped our training time to cut half by estimating the later behavior**

## AutoML
**The AutoML run as a best model gave a VotingEnsemble model while to get a real model the second best model or most contributing model was
LightGBMClassifier whose parameters were 
boosting_type:gbdt,class_weight=None,learning_rate=0.1,min_child_weight=0.001,min_split_gain=0.0,subsample=1.0 and subsample_for_bin=200000**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture?If there was a difference, why do you think there was one? <br>
The performance of both models were more or less similar the hyperDrive optimized the hyper-parameters in a LogisticRegression model to get an accuracy as primary <br>
metic of 0.906 while the AutoML gave a votingEnsemble which gave model an accuracy of 0.917 while the most contributing model was LightGBMClassifier<br>
The architectural difference in models were while LogisticRegression was more focused on fitting the data on a sigmoid scale LightGBMClassifier is a gradient boosting framework that uses tree based learning algorithms**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model? <br>
The Accuracy will remain more or less same until a better analysis of bank data is done and drawn into more data with additional datapoints which may help getting a better
prediction**


