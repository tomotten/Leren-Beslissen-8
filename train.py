import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import csv
import math
from data import *
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import warnings # Prevent warnings on windows OS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier





warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
install Xgboost
conda install -c conda-forge xgboost
https://www.lfd.uci.edu/~gohlke/pythonlibs/
"""

# Output the predictions in the correct format.
def output_pred(model, test_data, ids, y_test=None):
    y_pred = model.predict(test_data)
    predictions = [int(round(value)) for value in y_pred]  # logistic regression
    tmp = pd.DataFrame(predictions ,columns=['Survived'])
    tmp['PassengerId'] = ids.values
    indexes = ['PassengerId', 'Survived']
    tmp = tmp.reindex(columns=indexes)
    # evaluate predictions
    if y_test != None:
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return tmp


def get_wrongly_classified(predictions, real_y):
    predictions['Real_sur'] = real_y.values
    predictions['Correct'] = predictions.apply(lambda x: 1 if x['Survived'] == x['Real_sur'] else 0, axis=1)
    return predictions.loc[lambda df: df['Correct'] == 0, :], predictions.loc[lambda df: df['Correct'] == 1, :]


# Compare two csv files, and print accuracy.
def check(path1, path2):
    f1 = load_file(path1)
    f2 = load_file(path2)
    accuracy = accuracy_score(f1['Survived'], f2['Survived'])
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


# Load in train data and split to x and y
df = load_data('train.csv')
X, y = split_X_R(df)
columns = list(X)
N = 100
accs, f1_scores, tmp = [], [], []
# Loop for Cross-validation
ind_params = {'n_estimators': 12,'max_depth': 5,'min_child_weight': 1}#{'learning_rate': 0.01,
for i in range(N):
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=i)

    # Fit model (Xgboost Classifier) to training data
    model = xgb.XGBClassifier(**ind_params)
    model.fit(train_x.drop('PassengerId',1), train_y)

    # Get the predictions of the model and print the f1-score
    res = output_pred(model, test_x.drop('PassengerId', 1), test_x['PassengerId'])
    # print("F1-score: %0.2f" % (f1_score(test_y, res['Survived'])*100))
    f1_scores.append((f1_score(test_y, res['Survived'])*100))
    accuracy = accuracy_score(test_y, res['Survived'])
    accs.append(accuracy*100)
    tmp.append((i, accuracy*100))
    # print("Accuracy: %.2f%% \n" % (accuracy * 100.0))


tmp.sort(key=lambda tup: tup[1])
ind = math.floor(N/2.0)
# print("med:", tmp[ind][0], tmp[ind][1])

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=tmp[ind][0])
model = xgb.XGBClassifier(**ind_params)
model.fit(train_x.drop('PassengerId',1), train_y)
res = output_pred(model, test_x.drop('PassengerId', 1), test_x['PassengerId'])
wr, good = get_wrongly_classified(res, test_y)
id_list = wr['PassengerId'].tolist()
wrong = df.loc[df['PassengerId'].isin(id_list)]

print(wrong.describe())
# print(good.describe())

# svc = svm.SVC()
wrong_y = wrong.loc[:,('Survived')]
svc = KNeighborsClassifier(n_neighbors=5)

svc.fit(train_x.drop('PassengerId', 1), train_y)
res2 = output_pred(svc, wrong.drop(['PassengerId','Survived'], 1), wrong['PassengerId'])
acc = accuracy_score(wrong_y, res2['Survived'])
print("Accuracy: %.2f%% \n" % (acc * 100.0))

wr2, good2 = get_wrongly_classified(res2, wrong_y)
id_list2 = wr2['PassengerId'].tolist()
wrong2 = df.loc[df['PassengerId'].isin(id_list2)]
print(wrong2.describe())
# print(good2.describe())

print("Overall Accuracy: %.2f%%" % (sum(accs) / float(len(accs))))
print("Overall F1-score: %0.2f" % (sum(f1_scores) / float(len(f1_scores))))

# Save predictions as csv-file and compare to the actual (correct) classifications.
test_data = load_data('test.csv')
model = xgb.XGBClassifier()
model.fit(X.drop('PassengerId',1), y)
res = output_pred(model, test_data.drop('PassengerId',1), test_data['PassengerId'])
filename = 'output.csv'
res.to_csv(filename, index=False)
