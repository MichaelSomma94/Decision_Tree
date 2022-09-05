# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 13:02:01 2022

@author: Michael
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import precision_score,confusion_matrix

data = pd.read_csv("Iris.csv")
data = data.dropna()

data["Species"] = data["Species"].map({'Iris-setosa': 0, "Iris-versicolor": 1, 'Iris-virginica': 2})



#print(data.nunique())

x = np.array(data.iloc[:, [1,2,3,4]])
y = np.array(data[['Species']])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=1)

DecTree = tree.DecisionTreeClassifier()
DecTree = DecTree.fit(xtrain, ytrain)

ypred = DecTree.predict(xtest)

# cross validation for the Tree decision model

scores = cross_val_score(DecTree, x, y, cv=7)

tree.plot_tree(DecTree)

# Here the comparison to a multinominal regression
LogisticRegr = LogisticRegression(multi_class='multinomial', solver='lbfgs',random_state=42, max_iter=100)

logisticRegr = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
logisticRegr = logisticRegr.fit(xtrain, ytrain.ravel())
classifier = logisticRegr.named_steps['logisticregression']
ypred = logisticRegr.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)
print(data.columns)
# printing the weights for the features of the three lables
print(classifier.coef_)
# cross validation for the Logistic Regression model
n_scores = cross_val_score(logisticRegr, x, y.ravel(), scoring='accuracy', cv=7)
print('Logistic regression model:', np.mean(n_scores), '+/-', np.std(n_scores))
print('Decission Tree model:', np.mean(scores), '+/-', np.std(scores))


