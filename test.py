import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import re
import csv

"""
install Xgboost
conda install -c conda-forge xgboost
https://www.lfd.uci.edu/~gohlke/pythonlibs/
"""

def load_file(filename):
    return pd.read_csv(filename, delimiter=',')

def split_X_R(data):
    return data.drop('Survived', 1), data['Survived']


def remove_missing(df, thresh=100):
    columns = list(df)
    l = df.isnull().sum()
    for i, x in enumerate(l):
        # print(x, columns[i])
        if x > thresh:
            df.drop(columns[i], 1, inplace=True)
        elif x > 0 and x < thresh:  # fill in missing values
            df[columns[i]] = df[columns[i]].fillna(df[columns[i]].mean())
    return df


def split_classes(df):
    return df.loc[lambda df: df.Survived == 0, :], df.loc[lambda df: df.Survived == 1, :]


# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Define function to extract deck letter from cabin names
def get_deck(cabin):
    if cabin[0].isalpha():
        return cabin[0]
    return ''

def load_data(filename):
    # Load data
    df = load_file(filename)

    # Create a new feature Title, containing the titles of passenger names
    df['Title'] = df['Name'].apply(get_title)

    # Remove name and Ticket features
    df.drop('Name', 1, inplace=True)
    df.drop('Ticket', 1, inplace=True)

    # Group all non-common titles into one single grouping "Rare"
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',
                                                 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

     # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)

    # Replace text with numerical data
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

    # Map Embarked to numerical values
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map({'C': 0, 'S': 1, 'Q': 2})

    # Add new feature FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Add new feature has_cabin
    df['Has_Cabin'] = df['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

    # Add new feature Deck
    df['Deck'] = df['Cabin'].apply(lambda x: 0 if type(x) == float else x[0])
    df.drop('Cabin', 1, inplace=True)
    df.loc[df['Deck']=='A', 'Deck'] = 1
    df.loc[df['Deck']=='B', 'Deck'] = 2
    df.loc[df['Deck']=='C', 'Deck'] = 3
    df.loc[df['Deck']=='D', 'Deck'] = 4
    df.loc[df['Deck']=='E', 'Deck'] = 5
    df.loc[df['Deck']=='F', 'Deck'] = 6
    df.loc[df['Deck']=='G', 'Deck'] = 7
    df.loc[df['Deck']=='T', 'Deck'] = 8

    # Add AgeGroups, dividing age into 5 groups
    df['AgeGroup'] = df['Age']
    df.loc[df['AgeGroup']<=19, 'AgeGroup'] = 0
    df.loc[(df['AgeGroup']>19) & (df['AgeGroup']<=30), 'AgeGroup'] = 1
    df.loc[(df['AgeGroup']>30) & (df['AgeGroup']<=45), 'AgeGroup'] = 2
    df.loc[(df['AgeGroup']>45) & (df['AgeGroup']<=63), 'AgeGroup'] = 3
    df.loc[df['AgeGroup']>63, 'AgeGroup'] = 4
    df['AgeGroup'] = df['AgeGroup'].fillna(2)
    df.drop('Age', 1, inplace=True)

    # Round Fare feature to 2 decimals
    df['Fare'] = df['Fare'].apply(lambda x: round(x,2))

    return df

def output_pred(model, test_data, y_test=None):
    y_pred = model.predict(test_data)
    predictions = [round(value) for value in y_pred]  # logistic regression
    tmp = pd.DataFrame(predictions ,columns=['Survived'])
    tmp['PassengerId'] = test_data['PassengerId']
    indexes = ['PassengerId', 'Survived']
    tmp = tmp.reindex(columns=indexes)
    # evaluate predictions
    if y_test != None:
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return tmp


# def compare_csv(path1, path2):
#     f1 = file(path1, 'r')
#     f2 = file(path2, 'r')
#     c1 = csv.reader(f1)
#     c2 = csv.reader(f2)


# Load in train data and split to x and y
df = load_data('train.csv')
print(df.describe())
x, y = split_X_R(df)

# Load in test data, and clean if needed
test_df = load_data('test.csv')
test_df = remove_missing(test_df)

# Fit model to training data
# Xgboost Classifier
model = xgb.XGBClassifier()
model.fit(x, y)

# CLF Classifier
# clf = svm.SVC(gamma=0.001, C=100.)
# clf.fit(X_train, y_train)

res = output_pred(model, test_df)
filename = 'output.csv'
res.to_csv(filename, index=False)
