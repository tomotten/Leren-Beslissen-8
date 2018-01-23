import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFECV
import re
import csv
from data import *
from tuning import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import f1_score
import warnings # Prevent warnings on windows OS
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

# Compare two csv files, and print accuracy.
def check(path1, path2):
    f1 = load_file(path1)
    f2 = load_file(path2)
    accuracy = accuracy_score(f1['Survived'], f2['Survived'])
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Load in train data and split to x and y
df = load_data('train.csv')
x, y = split_X_R(df)
columns = list(x)

print(x.describe())

# Load in test data, and clean if needed
test_df = load_data('test.csv')
test_df = remove_missing(test_df)

# Fit model (Xgboost Classifier) to training data
model = xgb.XGBClassifier()
model.fit(x.drop('PassengerId',1), y)

# Calculate and print cross-validation accuracy and std.
scores = cross_val_score(model, x.as_matrix(), y.as_matrix(), cv=10)
scores = scores*100
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Load in the correct classification for the test data
true_y = load_file('gender_submission.csv')

# Get the predictions of the model and print the f1-score
res = output_pred(model, test_df.drop('PassengerId', 1), test_df['PassengerId'])
print("F1-score: %0.2f" % (f1_score(true_y['Survived'], res['Survived'])*100))

# Save predictions as csv-file and compare to the actual (correct) classifications.
filename = 'output.csv'
res.to_csv(filename, index=False)
check(filename, 'gender_submission.csv')

fit_importance(model,x.drop('PassengerId',1),y)
