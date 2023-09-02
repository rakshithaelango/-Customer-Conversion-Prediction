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

* STEP 4:  Data cleaning.
  
In this step, we understand more about the data and prepare it for further analysis. Cleaning data essentially means removing discrepancies from data such as missing fields, improper values, setting the right format of the data, structuring data from raw files.

* STEP 5:  Exploratory Data Analysis [EDA].

This EDA is the backbone of data science. To analyz  and investigate the data sets and summarize their main characteristics. By using data visualization methods. 
And handling outliers in this method I have  used box plot.

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
  
* STEP 6:  Encoding the data.
The process of converting the categorical features into numeric features.

 
#  Step 7:  Splitting and Scaling the data.
We used SMOTE method because this whole data set is imbalanced.
And Splitted the data into train and test splitting.

For Scaling purpose I used Standard scaler.


# Step 8:  Model.
In this step,3 Machine learning models used. 

•	Logistic Regression.
•	Random forest classifier.
•	XGboost classifier.


# Step 9: Conclusion.
Based on the model Evaluation.Got the good score.

 
 
 
 
 
 
 
