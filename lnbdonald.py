import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re

def load_csv(filename):
    return pd.read_csv(filename, delimiter=',')

def split_X_R(data):
    return data.drop('Survived', 1), data['Survived']

# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

    # Load the datasets
train = load_csv('train.csv')
train.head(5)

full_data = [train]

# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

# Remove unused name, cabin and ticket features
train.drop('Name', 1, inplace=True)
train.drop('Ticket', 1, inplace=True)
train.drop('Cabin', 1, inplace=True)

full_data = [train]

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',
                                                 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

     # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


for dataset in full_data:
    # Replace all NULLS in the Embarked column with the most common embarkation point (S).
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# get average, std, and number of NaN values in titanic_df
average_age   = train["Age"].mean()
std_age       = train["Age"].std()
count_nan_age = train["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
random_age = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)

# fill NaN values in Age column with random values generated
train["Age"][np.isnan(train["Age"])] = random_age

# convert from float to int
train['Age'] = train['Age'].astype(int)

#train.head(5)
#train.info()
train.describe()

from sklearn import svm
from sklearn.metrics import accuracy_score

clf = svm.SVC(gamma=0.001, C=100.)

# split data into train and test sets
seed = 7
test_size = 0.33
x, y  = split_X_R(train)

X_train, X_test, y_train, y_test = train_test_split(x.values, y.values, test_size=test_size, random_state=seed)

clf.fit(X_train, y_train)
print(list(y_test == clf.predict(X_test)).count(True) / y_test.size)
accuracy_score(y_test, clf.predict(X_test))
