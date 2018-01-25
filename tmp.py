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


def main(trainFile, testFile, N, model1, model2, ind_params):
    # Load in train data and split to x and y
    df = load_data(trainFile)
    X, y = split_X_R(df)
    columns = list(X)
    accs, f1_scores, tmp = [], [], []
    # Loop for Cross-validation
    for i in range(N):
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=i)
        # Fit model (Xgboost Classifier) to training data
        model = xgb.XGBClassifier(**ind_params)
        model.fit(train_x.drop('PassengerId',1), train_y)
        # Get the predictions of the model and print the f1-score
        res = output_pred(model, test_x.drop('PassengerId', 1), test_x['PassengerId'])
        # print("F1-score: %0.2f" % (f1_score(test_y, res['Survived'])*100))
        # f1_scores.append((f1_score(test_y, res['Survived'])*100))
        accuracy = accuracy_score(test_y, res['Survived'])
        accs.append(accuracy*100)
        tmp.append((i, accuracy*100))
        # print("Accuracy: %.2f%% \n" % (accuracy * 100.0))

    tmp.sort(key=lambda tup: tup[1])
    ind = math.floor(N/2.0)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=tmp[ind][0])

    model1.fit(train_x.drop('PassengerId',1), train_y)
    res = output_pred(model1, test_x.drop('PassengerId', 1), test_x['PassengerId'])
    test_x['Survived'] = res['Survived'].values
    dead, alive = split_classes(test_x)


    id_list = dead['PassengerId'].tolist()
    tmp1 = df.loc[df['PassengerId'].isin(id_list)]
    real_y1 = tmp1.loc[:, 'Survived']

    id_list1 = alive['PassengerId'].tolist()
    tmp2 = df.loc[df['PassengerId'].isin(id_list1)]
    real_y2 = tmp2.loc[:, 'Survived']
    res1 = pd.DataFrame(np.ones(len(alive)), dtype=np.int, columns=['Survived'])
    res1['PassengerId'] = alive['PassengerId'].values
    res1 = res1.reindex(columns=['PassengerId', 'Survived'])
    # print(res1)

    model2.fit(train_x.drop('PassengerId', 1), train_y)
    res2 = output_pred(model2, tmp1.drop(['PassengerId', 'Survived'], 1), tmp1['PassengerId']) # output for dead

    # res2 = res2.loc['Survived']
    tmp1 = tmp1.append(tmp2, ignore_index=True)
    real_y1 = real_y1.append(real_y2, ignore_index=True)
    res2 = res2.append(res1, ignore_index=True)

    # print(len(tmp1), len(real_y1))
    acc = accuracy_score(res2['Survived'], real_y1)
    print("Accuracy: %.2f%% \n" % (acc * 100.0))
    # print("Overall Accuracy: %.2f%%" % (sum(accs) / float(len(accs))))
    # print("Overall F1-score: %0.2f" % (sum(f1_scores) / float(len(f1_scores))))

    # Save predictions as csv-file and compare to the actual (correct) classifications.
    test_data = load_data(testFile)
    model1.fit(X.drop('PassengerId',1), y)
    res = output_pred(model1, test_data.drop('PassengerId',1), test_data['PassengerId'])
    test_data['Survived'] = res['Survived'].values
    dead1, alive1 = split_classes(test_data)

    res1 = pd.DataFrame(np.ones(len(alive1)), dtype=np.int, columns=['Survived'])
    res1['PassengerId'] = alive1['PassengerId'].values
    res1 = res1.reindex(columns=['PassengerId', 'Survived'])

    id_list2 = dead1['PassengerId'].tolist()
    tmp3 = test_data.loc[test_data['PassengerId'].isin(id_list2)]
    real_y1 = tmp3.loc[:, 'Survived']

    id_list1 = alive1['PassengerId'].tolist()
    tmp2 = test_data.loc[test_data['PassengerId'].isin(id_list1)]
    real_y2 = tmp2.loc[:, 'Survived']

    model2.fit(X.drop('PassengerId',1), y)
    res2 = output_pred(model2, tmp3.drop(['PassengerId', 'Survived'], 1), tmp3['PassengerId'])
    tmp3 = tmp3.append(tmp2, ignore_index=True)
    real_y1 = real_y1.append(real_y2, ignore_index=True)
    res2 = res2.append(res1,ignore_index=True)

    filename = 'output.csv'
    res2.to_csv(filename, index=False)



ind_params = {'n_estimators': 12,'max_depth': 5,'min_child_weight': 1}
# model1 = KNeighborsClassifier(n_neighbors=5)
model1 = svm.SVC()
model2 = RandomForestClassifier(max_depth=4, random_state=0)
main('train.csv', 'test.csv', 100, model1, model2, ind_params)
