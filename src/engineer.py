import pandas as pd
import config 
from clean import map_columns
import re

def feature_engineer(dataframe, columns_to_engineer=[], features_to_ohe=[]):
    
    dataframe = create_new_features(dataframe)

    dataframe = remove_unspecified_features(dataframe, columns_to_engineer)

    dataframe = categorise_features(dataframe)

    dataframe = one_hot_encode(dataframe, features_to_ohe)

    return dataframe

def family_size(sibsp, parch):
    return sibsp + parch + 1

def is_alone(family_size):
    if family_size == 1:
        return 1
    else:
        return 0

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:
        return title_search.group(1)
    else:
        return ""

def categorise_title(name):

    title = get_title(name)

    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major','Rev',
                  'Sir', 'Jonkheer', 'Dona']
    miss_titles = ['Mlle', 'Ms']
    mrs_titles = ['Mme']

    if title in rare_titles:
        return "Rare"
    elif title in miss_titles:
        return "Miss"
    elif title in mrs_titles:
        return "Mrs"
    else:
        return title


def categorise_age(age):
    if age <=16:
        return 0
    elif (age > 16) & (age <= 32):
        return 1
    elif (age > 32) & (age <= 48):
        return 2
    elif (age > 48) & (age <= 64):
        return 3
    else:
        return 4
    

def categorise_fare(fare):
    if fare <= 7.91:
        return 0
    elif (fare > 7.91) & (fare <= 14.454):
        return 1
    elif (fare > 14.454) & (fare <= 31):
        return 2
    elif fare > 31:
        return 3


def remove_unspecified_features(dataframe, columns_to_enginner):
    all_possible_features = set(["family_size", "is_alone"])

    features_to_remove = all_possible_features - set(columns_to_enginner) 

    return dataframe.drop(features_to_remove, axis=1)

def create_new_features(dataframe):
    dataframe["family_size"] = family_size(dataframe["SibSp"],
                                           dataframe["Parch"])

    dataframe["is_alone"] = dataframe["family_size"].apply(is_alone).astype(int)
    return dataframe

def categorise_features(dataframe):
    # Age
    dataframe["Age"] = dataframe["Age"].apply(categorise_age)
    # Fare
    dataframe["Fare"] = dataframe["Fare"].apply(categorise_fare)
    # Title
    title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataframe["Name"] = dataframe["Name"].apply(categorise_title).map(title_map)
    #Cabin
    dataframe["Cabin"] = dataframe["Cabin"].map(lambda x: x[0])
    return dataframe

def one_hot_encode(dataframe, features):

    for feature in features:
        dummies = pd.get_dummies(dataframe[feature], feature)
        dataframe = pd.concat([dataframe, dummies], axis=1)
    return dataframe.drop(features, axis=1)

def reconcile_test_set(df_train, df_test):
    train_columns = df_train.columns
    test_columns = df_test.columns

    columns_to_remove = set(test_columns) - set(train_columns)
    columns_to_add = set(train_columns) - set(test_columns)

    for column in columns_to_add:
        df_test[column] = 0

    df_test = df_test.drop(columns_to_remove, axis=1)

    ordered_columns = list(train_columns)
    ordered_columns.remove("Survived")
    reordered_test_df = df_test[ordered_columns] 

    return reordered_test_df

def main():
    print("Nothing to see here: Feature Engineer")
    return

if __name__ == '__main__':
    main()