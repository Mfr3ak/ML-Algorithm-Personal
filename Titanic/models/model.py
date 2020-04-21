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

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(random_state = 0)
cross_val_score(clf,x_var,y_var,cv = 10)
