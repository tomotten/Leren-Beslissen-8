import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import csv
from data import *
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import warnings # Prevent warnings on windows OS


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
    tmp['PassengerId'] = ids
    indexes = ['PassengerId', 'Survived']
    tmp = tmp.reindex(columns=indexes)
    # evaluate predictions
    if y_test != None:
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return tmp

def get_wrongly_classified(predictions, real_y):
    predictions['Real_sur'] = real_y
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
df = load_data('train.csv')
X, y = split_X_R(df)
columns = list(X)
N = 150
accs, f1_scores = [], []
for i in range(N):
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=i)

    # Fit model (Xgboost Classifier) to training data
    model = xgb.XGBClassifier()
    model.fit(train_x.drop('PassengerId',1), train_y)

    # Get the predictions of the model and print the f1-score
    res = output_pred(model, test_x.drop('PassengerId', 1), test_x['PassengerId'])
    print("F1-score: %0.2f" % (f1_score(test_y, res['Survived'])*100))
    f1_scores.append((f1_score(test_y, res['Survived'])*100))
    accuracy = accuracy_score(test_y, res['Survived'])
    accs.append(accuracy*100)
    print("Accuracy: %.2f%% \n" % (accuracy * 100.0))

# feature importance
# print(model.feature_importances_)
# # plot
# xgb.plot_importance(model)
# plt.show()

print("Overall Accuracy: %.2f%%" % (sum(accs) / float(len(accs))))
print("Overall F1-score: %0.2f" % (sum(f1_scores) / float(len(f1_scores))))


# # Save predictions as csv-file and compare to the actual (correct) classifications.
# filename = 'output.csv'
# res.to_csv(filename, index=False)
# # check(filename, 'gender_submission.csv')
#
# test_x['Survived'] = res.loc[res['Survived'], 'Survived']
# wrong, good = get_wrongly_classified(test_x, test_y)
#
#
# print("Wrong: \n", wrong.describe(), good.describe())
# dead, sur = split_classes(x)
# print(dead.describe())
