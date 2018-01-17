import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn import svm
import re
import csv

def load_file(filename):
    return pd.read_csv(filename, delimiter=',')

def split_X_R(data):
    return data.drop('Survived', 1), data['Survived']

def split_classes(df):
    return df.loc[lambda df: df.Survived == 0, :], df.loc[lambda df: df.Survived == 1, :]


def remove_missing(df, thresh=100):
    columns = list(df)
    l = df.isnull().sum()
    for i, x in enumerate(l):
        # print(x, columns[i])
        if x > thresh:
            df.drop(columns[i], 1, inplace=True)
        elif x > 0 and x < thresh:  # fill in missing values
            df[columns[i]] = df[columns[i]].fillna(df[columns[i]].median())
    return df

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

    # Remove name, PassengerId and Ticket features
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
    df['Sex'] = df.loc[df['Age']<=18,'Sex'] = 'child'

    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1, 'child': 2})

    # Add AgeGroups, dividing age into 5 groups
    df['AgeGroup'] = df['Age']
    df.loc[df['AgeGroup']<=18, 'AgeGroup'] = 0
    df.loc[(df['AgeGroup']>18) & (df['AgeGroup']<=30), 'AgeGroup'] = 1
    df.loc[(df['AgeGroup']>30) & (df['AgeGroup']<=45), 'AgeGroup'] = 2
    df.loc[(df['AgeGroup']>45) & (df['AgeGroup']<=63), 'AgeGroup'] = 3
    df.loc[df['AgeGroup']>63, 'AgeGroup'] = 4
    df['AgeGroup'] = df['AgeGroup'].fillna(2)
    df.drop('Age', 1, inplace=True)


    # Map Embarked to numerical values
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked_C'] = df['Embarked']
    df['Embarked_S'] = df['Embarked']
    df['Embarked_Q'] = df['Embarked']
    df['Embarked_C'] = df['Embarked_C'].apply(lambda x: 1 if x == 'C' else 0)
    df['Embarked_S'] = df['Embarked_S'].apply(lambda x: 1 if x == 'S' else 0)
    df['Embarked_Q'] = df['Embarked_Q'].apply(lambda x: 1 if x == 'Q' else 0)
    df.drop('Embarked', 1, inplace=True)

    # Add new feature FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df['Fare'].fillna(df['Fare'].median(), inplace = True)

    # df['Fare_cat'] = df['Fare']
    # df['Fare_cat'] = df['Fare_cat'].apply(lambda x: np.floor(np.log10(x + 1)).astype('int'))
    # df.drop('Fare', 1, inplace=True)


    # Add new feature Deck
    df['Deck'] = df['Cabin']
    df['Deck'] = df['Deck'].str.replace('[H-Z]+.*', '8')
    df['Deck'] = df['Deck'].apply(lambda x: 0 if type(x) == float else x[0])
    df.drop('Cabin', 1, inplace=True)
    df.loc[df['Deck']=='A', 'Deck'] = 1
    df.loc[df['Deck']=='B', 'Deck'] = 2
    df.loc[df['Deck']=='C', 'Deck'] = 3
    df.loc[df['Deck']=='D', 'Deck'] = 4
    df.loc[df['Deck']=='E', 'Deck'] = 5
    df.loc[df['Deck']=='F', 'Deck'] = 6
    df.loc[df['Deck']=='G', 'Deck'] = 7
    df['Deck'] = df['Deck'].astype(int)

    return df
