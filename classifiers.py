import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn import svm, linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import re
import csv
from data import *
from test import *
import warnings # Prevent warnings on windows OS
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load in train data and split to x and y
df = load_data('train.csv')
x, y = split_X_R(df)
x = remove_missing(x)

#print(x.describe())

# Load in test data, and clean if needed
test_df = load_data('test.csv')
test_df = remove_missing(test_df)

# Fit model to training data

# Classifiers
#xgbmodel = xgb.XGBClassifier()
logistic = linear_model.LogisticRegression(C=1e5)
svc = svm.SVC()
knn = KNeighborsClassifier()
rforest = RandomForestClassifier(max_depth=4, random_state=0)

models = {"logistic": logistic, \
"svc": svc, "knn":knn,"rforest":rforest}
result = []
n = 0
for key,model in models.items():
    print(key)
    model.fit(x.drop('PassengerId',1), y)
    scores = cross_val_score(model, x.as_matrix(), y.as_matrix(), cv=10)
    scores = scores*100
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

res = output_pred(rforest, test_df.drop('PassengerId', 1), test_df['PassengerId'])
filename = 'output.csv'
res.to_csv(filename, index=False)

check(filename, 'gender_submission.csv')
