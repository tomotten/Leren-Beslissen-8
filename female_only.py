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

# Output the predictions in the correct format.
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

def get_wrongly_classified(predictions, real_y):
    predictions['Real_sur'] = real_y['Survived']
    predictions['Correct'] = predictions.apply(lambda x: 1 if x['Survived'] == x['Real_sur'] else 0, axis=1)
    # print(predictions.describe())
    return predictions.loc[lambda df: df['Correct'] == 0, :], predictions.loc[lambda df: df['Correct'] == 1, :]

# Compare two csv files, and print accuracy.
def check(path1, path2):
    f1 = load_file(path1)
    f2 = load_file(path2)
    accuracy = accuracy_score(f1['Survived'], f2['Survived'])
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


# Load in train data and split to x and y
df = load_file('test.csv')

# classify all female as survivors and male as dead.
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
df['Survived'] = df['Sex']
df['Survived'] = df['Survived'].map({0:1, 1:0})

tmp = pd.DataFrame(df ,columns=['PassengerId', 'Survived'])

# Save predictions as csv-file and compare to the actual (correct) classifications.
filename = 'output.csv'
tmp.to_csv(filename, index=False)
check(filename, 'gender_submission.csv')
