import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_file(filename):
    return pd.read_csv(filename, delimiter=',')

def split_X_R(data):
    return data.drop('Survived', 1), data['Survived']


def remove_missing(df):
    columns = list(df)
    l = df.isnull().sum()
    for i, x in enumerate(l):
        if x > 200:
            df.drop(columns[i], 1, inplace=True)
    return df

def split_classes(df):
    return df.loc[lambda df: df.Survived == 0, :], df.loc[lambda df: df.Survived == 1, :]


# Load data
df = load_file('train.csv')

# Remove name and Ticket features
df.drop('Name', 1, inplace=True)
df.drop('Ticket', 1, inplace=True)

# Replace text with numerical data
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'S': 1, 'Q': 2})

# Add new feature FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Add new feature has_cabin
df['Has_Cabin'] = df['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
df.drop('Cabin', 1, inplace=True)

# Round Fare feature to 2 decimals
df['Fare'] = df['Fare'].apply(lambda x: round(x,2))


df = remove_missing(df)
# x0, x1 = split_classes(df)
x, y = split_X_R(df)

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(x.values, y.values, test_size=test_size, random_state=seed)
print(y_train.shape)

# fit model no training data
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
