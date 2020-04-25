# -*- coding: utf-8 -*-
"""
Spyder Editor
# 
This is a temporary script file.
"""
import pandas as pd
import numpy as np
import os

os.chdir(r'C:/Users/madimulam001/Downloads/Personal/Latest/Personal_Projects/Titanic')

path_train = 'data/train.csv'
data = pd.read_csv(path_train)

#Columns
'''['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
 'Ticket', 'Fare', 'Cabin', 'Embarked']'''

# Exploratory analysis on the dataset
print(data.isna().sum())

# Age = 177
# Cabin = 687
# Embarked = 2

'''
Assumptions - 
1. Age is imputed with average value but based on the cabin they are in
2. Cabin is dropped altogether
3. dropped two rows as its too low number
'''
def drop_column(data,column_name_list):
    return data.drop(columns = column_name_list,inplace = True)

def imputer(data, column_name_list):
    for column in column_name_list:
        data[column_name] = data[column_name].fillna(data[column_name].mean())
    return data
                
data.drop(columns = ['Cabin'] , inplace = True)
data['Embarked'] = data['Embarked'].dropna()

print(data.columns)
print((data.get('Age').mean()))
data['Age'] = data['Age'].fillna(data['Age'].mean())

print(data['Age'].mean())

data.drop(columns = ['Ticket'],inplace = True)

list(data.columns)

data.drop(columns = ['PassengerId'],inplace = True)
data.drop(columns = ['Name'],inplace = True)

y_var = data['Survived']
x_var = data.drop(columns = ['Survived'])

x_var['Sex'] = np.where(x_var['Sex'] == 'male',1,0)

x_var.head(2)

data.dropna(inplace = True)

pd.unique(x_var['Embarked'])

# Function to encode target variable
# Needs rework but on the right path
def target_encoder(data, target_column):
    '''
    Encodes the input data's target column into integers based on the unique
    values in the column.
    
    Note - Replaces the input column and makes the changes inplace
    
    Args
    ------
        data : dataframe
        target_column : column where the string values are to be converted
        
    Returns
    ------
        data : data with same column names with target column converted to numerical
    '''
    
    unique_values = pd.unique(data[target_column])
    map_to_int = {name: n for n, name in enumerate(unique_values)}
    data[target_column] = data[target_column].replace(map_to_int)
    return data

x_var = target_encoder(x_var, 'Embarked')


# Checking distribution of classes
y_var.value_counts() # 0 - 549 / 1 - 342

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from time import time
from operator import itemgetter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib import pyplot as plt
%matplotlib inline

# Basic cross-validation fit of DecisionTree
def model_eval_cv(clf, x_var, y_var,cv):
    '''
    Function to fit a model using cross validation and print the mean and 
    maximum cross validation scores
    
    Args
    ------
        clf : Classifier or model which needs to be evaluated
        cv : # of folds in cross validation
        
    Returns
    ------
        scores : scores in each of cv folds
        
    '''
    clf.fit(x_var,y_var)
    scores = cross_val_score(clf,x_var,y_var,cv = cv)
    print("\nMean cross validation score is {} \n\nHighest cross validation score is {}\n".format(scores.mean(),scores.max()))
    return scores

clf = DecisionTreeClassifier(random_state = 0)
scores = model_eval_cv(clf, x_var, y_var, cv=10)

# Grid searching
def run_gridsearch(clf, param_grid, x_var, y_var, cv):
    '''
    Perform grid search in pre-defined parameter grid and return the best 
    performing parameter combination
    
    Args
    ------
        clf : Classifier to be used for current task
        param_grid : initiated range for parameters
        
    Returns
    ------
        best_estimator : the best possible estimator from the grid defined
        
    '''
    grid_search = GridSearchCV(clf, param_grid, cv = cv)
    start = time()
    grid_search.fit(x_var, y_var)
    print('\nGrid Search took {} time\n'.format(time() - start))
    best_estimator = grid_search.best_estimator_
    return best_estimator

# set of parameters to test
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }

clf = DecisionTreeClassifier()

best_estimator = run_gridsearch(clf, param_grid, x_var, y_var, cv = 10)

scores = model_eval_cv(best_estimator, x_var, y_var, cv = 10)

def confusion_matrix_calculator(clf, x_var, y_var):
    '''
    Calcuates the confusion matrix and returns the matrix
    the horizontal values of the matrix denote the actuals
    the vertical values fo the matrix denote the predictions
    
    Args
    ------
    clf : best_estimator or the classifer used for making predictions
    
    Returns
    ------
    confusion_mat : confustion matrix using y_actuals and y_predictions

    '''
    y_pred = clf.predict(x_var)
    confusion_mat = confusion_matrix(y_var, y_pred)
    return confusion_mat

def auc_roc_Plotter(clf, x_var, y_var):
    '''
    Plots the AUC and ROC curve for the given classifier for different probabilites
    
    Args
    ------
    clf : best_estimator or the classifier used for making predictions
    
    Returns
    ------
    Plots  AUC-ROC Curve on the plots tab of spyder
    
    '''
    y_pred_prob = clf.predict_proba(x_var)
    fpr, tpr, _ = roc_curve(y_var, y_pred_prob[:,1])
    plt.clf()
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.show()  

conf_mat = confusion_matrix_calculator(best_estimator, x_var, y_var)
auc_roc_Plotter(best_estimator, x_var, y_var)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Initial Cross Validation Model
clf = RandomForestClassifier(random_state = 0)
model_eval_cv(clf, x_var, y_var, cv = 10)

# Optimized Parameters to find best estimator
param_grid = {'n_estimators' : [10,25,50,75,100],
              'min_samples_leaf' : [2,5,10,20],
              'max_samples' : [0.2,0.5,0.75],
              "criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20]
              }

best_estimator = run_gridsearch(clf, param_grid, x_var, y_var, cv = 10)

scores = model_eval_cv(best_estimator, x_var, y_var, cv = 10)

conf_mat = confusion_matrix_calculator(best_estimator, x_var, y_var)
print(conf_mat)
auc_roc_Plotter(best_estimator, x_var, y_var)   
    
    
