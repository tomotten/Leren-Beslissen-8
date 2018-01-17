import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn import svm
import re
import csv
from data import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier 
import warnings # Prevent warnings on windows OS


warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
install Xgboost
conda install -c conda-forge xgboost
https://www.lfd.uci.edu/~gohlke/pythonlibs/
"""

def output_pred(model, test_data, ids, y_test=None):
    y_pred = model.predict(test_data)
    predictions = [int(round(value)) for value in y_pred]  # logistic regression
    tmp = pd.DataFrame(predictions ,columns=['Survived'])
    tmp['PassengerId'] = ids
    indexes = ['PassengerId', 'Survived']
    tmp = tmp.reindex(columns=indexes)
    # evaluate predictions
    if y_test != None:
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return tmp

def check(path1, path2):
    f1 = load_file(path1)
    f2 = load_file(path2)
    accuracy = accuracy_score(f1['Survived'], f2['Survived'])
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


# Load in train data and split to x and y
df = load_data('train.csv')
x, y = split_X_R(df)
# x = remove_missing(x)
columns = list(x)

print(x.describe())

# Load in test data, and clean if needed
test_df = load_data('test.csv')
test_df = remove_missing(test_df)

# Fit model to training data
# Xgboost Classifier
model = xgb.XGBClassifier()
model.fit(x.drop('PassengerId',1), y)

scores = cross_val_score(model, x.as_matrix(), y.as_matrix(), cv=10)
scores = scores*100
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

true_y = load_file('gender_submission.csv')

res = output_pred(model, test_df.drop('PassengerId', 1), test_df['PassengerId'])
print("F1-score: %0.2f" % (f1_score(true_y['Survived'], res['Survived'])*100))

filename = 'output.csv'
res.to_csv(filename, index=False)

check(filename, 'gender_submission.csv')
