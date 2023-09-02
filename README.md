# Customer-Conversion-Prediction


# Project Description :

In this project. We're going to predict if the customer will subscribe to our term insurance or not in telephonic marketing campaign using Machine Learning Models.

From the Given DataSet is:

* It is Supervised Machine Learning.

* We have target variable y.

* It is classification problem.

* y is a continuous variable.

## Project Approach :

In this project, I used googleColab  as IDE for Python Programming Language.

# Project In-Detailed :

* STEP 1: Understanding the whole dataset.
  
First and foremost we need to understand the concept of the data. In this Customer Conversion Prediction this is an logistics regression based problem in that It has an target variable so, it is classification problem . In class it is Binary classification problem.

* STEP 2:  Importing all the necessary python libraries.

Import all the necessary libraries.

import pandas as pd

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler 

from sklearn.metrics import roc_curve

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression 

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score,confusion_matrix

from imblearn.combine import SMOTETomek

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

* STEP 3:  Loading the dataset.

After importing all the necessary packages. Next step is to load the Data by using .csv Format.

* STEP 4:  Data cleaning AND Data Pre-processing.
  
In this step, we understand more about the data and prepare it for further analysis. Cleaning data essentially means removing discrepancies from data such as missing fields, improper values, setting the right format of the data, structuring data from raw files And By Analysing the statistical methods handle the outliers and treated the outliers by using box plot.

HANDLING OUTLIERS:

* AGE AND TREATING AGE BY USING BOXPLOT:

![IMG_20230902_135102](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/c028eca8-f2b6-4d9b-9b39-36b94fca4d19)
![IMG_20230902_135206 (1)](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/b9d6bcb9-42fd-42e8-847d-d2854182ec44)

* DAY NO OUTLIER IS FOUND:

![1693643345976](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/ce53d88c-d6cc-4e97-84e5-d7be35ad1eb0)
    
* DURATION AND TREATING DURATION BY USING BOXPLOT:
  
![1693643655327](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/5389e682-36ab-4726-beea-5a3f8deaa652)
![1693643655321](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/c0948b73-4b2d-4464-a0fb-d207c30564fc)

* NUM_CALLS AND TREATING NUM_CALLS BY USING BOXPLOT:
  
![1693644012380](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/f742b275-7a43-4f48-95c2-4eb43200d343)
![1693644012376](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/6d13b659-91e1-45f9-94e7-4bef25beaac1)

* STEP 5:  Exploratory Data Analysis [EDA].

This EDA is the backbone of data science. To analyz  and investigate the data sets and summarize their main characteristics. By using data visualization methods. 

TARGET VARIABLES: 

![1693644888988](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/990ee5bd-9af4-43c2-9dcd-260dde6ed718)

BY,This we can clearly see huge imbalanced data occur.

Univariate analysis feature of categorical variables.

In this Categorical Variable: Job, Marital_Status, Education_Qualification, Call_type, Month and Prev_Outcome.

![1693644888972](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/b1dd91eb-b84d-4429-9b3a-27d55800a897)
![1693644888965](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/b7f1c053-fdb1-4be7-b60b-292ffd594f8a)
![1693644888968](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/bb7ca907-dfc4-4469-a674-62e877675831)
![1693644888981](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/866abc21-e7b3-4b4b-ba61-200b39f007cb)
![1693644888977](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/c343f7eb-c72b-4567-9dc4-6a89785b6144)
![1693644888985](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/52537b8b-5ccf-4283-a47e-af7846e2607b)

Univariate analysis feature of numerical variables.

In this numerical Variable: Age, Day, Num_calls. 

Call duration impact the model to subcribe the insurance. num_calls made also the reason to sbscribe the insurance.

![1693645956035](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/a5f94ddb-cfb5-4751-8514-ef584e7d9b8c)
![1693645956040](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/898d2ed2-b1b5-4a93-aadd-32b8aabac081)
![1693645956045](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/53d56d76-5c8e-406f-8cbe-9e37ebe4eff8)


Bivariate analysis feature of Categorical Columns..

In this  Categorical Variable: Job, Marital_Status, Education_Qualification, Call_type, Month, Prev_Outcome And Y.

![1693647152916](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/6bb536db-4f07-45bb-8691-a7d3875d17e7)
![1693647152910](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/4efc2e08-03f1-4dfc-9ca6-d782a9d58af1)
![1693647152907](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/49cc5d5c-5d7f-464d-94c6-5feeec6e232c)
![1693647152913](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/a94355aa-7661-4f1a-a98e-3cabebf4dfc1)
![1693647152904](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/50af112d-a6a9-40ca-b9e4-aae55d3cd2a3)
![1693647152919](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/23f81eee-c29d-4744-b7df-0f279e6229c0)
![1693647152901](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/71113635-3657-4169-b260-48d979f456b3)

* STEP 6:  Encoding the data.
The process of converting the categorical features into numeric features.

![1693647562833](https://github.com/rakshithaelango/-Customer-Conversion-Prediction/assets/116090323/c352c257-506d-4699-9242-a2889810093b)

 
* STEP 7:  Splitting and Scaling the data.
  
   *  We used SMOTE method because this whole data set is imbalanced.
   *  And Splitted the data into train and test splitting.
   *  For Scaling purpose I used Standard scaler.

* STEP 8: Model Building.
  In this step 3 Machine learning models used. 

•	Logistic Regression.
•	Random forest classifier.
•	XGboost classifier.

* LogisticRegression
  The accuracy of Logistic Regression is :  91.95352088065701 %
  The aurroc_auc_score of Logistic Regression is :  0.9747468353247064

* RandomForest
  The accuracy of Random Forest is :  93.40380919098375 %
  The aurroc_auc_score of  random forest is :  0.9852591716689892

*  XGboost
  The accuracy of X Gradient Boosting is :  93.4169142058361 %
  The aurroc_auc_score of Gradient Boosting is :  0.9858277950477008

# STEP 9: Feature Importance.
  Techniques that calculate a score for all the input features for a given model.

# Step 9: Conclusion.
Based on the model Evaluation.Got the good score.

 
 
 
 
 
 
 
