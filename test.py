import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import re

"""
install Xgboost
conda install -c conda-forge xgboost
"""

def load_file(filename):
    return pd.read_csv(filename, delimiter=',')

def split_X_R(data):
    return data.drop('Survived', 1), data['Survived']


def remove_missing(df, thresh=200):
    columns = list(df)
    l = df.isnull().sum()
    for i, x in enumerate(l):
        if x > thresh:
            df.drop(columns[i], 1, inplace=True)
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
    df.drop('Cabin', 1, inplace=True)

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

    # df = remove_missing(df)


df = load_data('train.csv')
# x0, x1 = split_classes(df)
print(df.describe())

x, y = split_X_R(df)

# Split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(x.values, y.values, test_size=test_size, random_state=seed)


# Fit model to training data
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(X_train, y_train)


# make predictions for test data
y_pred = clf.predict(X_train)
predictions = y_pred
# predictions = [round(value) for value in y_pred]  # logistic regression


# evaluate predictions
accuracy = accuracy_score(y_train, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
