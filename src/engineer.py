import pandas as pd
import config 

def feature_engineer(dataframe, columns_to_enginner, features_to_ohe):
    dataframe["family_size"] = family_size(dataframe["SibSp"],
                                           dataframe["Parch"])
    dataframe["is_alone"] = dataframe["family_size"].apply(is_alone).astype(int)

    dataframe = remove_unspecified_features(dataframe, columns_to_enginner)

    dataframe = one_hot_encode(dataframe, features_to_ohe)

    return dataframe

def family_size(sibsp, parch):
    return sibsp + parch + 1

def is_alone(family_size):
    if family_size == 1:
        return 1
    else:
        return 0

def remove_unspecified_features(dataframe, columns_to_enginner):
    all_possible_features = set(["family_size", "is_alone"])

    features_to_remove = all_possible_features - set(columns_to_enginner) 

    return dataframe.drop(features_to_remove, axis=1)

def one_hot_encode(dataframe, features):

    for feature in features:
        dummies = pd.get_dummies(dataframe[feature], feature)
        dataframe = pd.concat([dataframe, dummies], axis=1)
    return dataframe.drop(features, axis=1)

def reconcile_test_set(df_train, df_test):
    train_columns = df_train.columns
    test_columns = df_train.columns

    columns_to_remove = set(test_columns) - set(train_columns)
    columns_to_add = set(train_columns) - set(test_columns)

    for column in columns_to_add:
        df_test[column] = 0

    df_test = df_test.drop(columns_to_remove, axis=1)
    reordered_test_df = df_test[train_columns] 

    return reordered_test_df

def main():
    return

if __name__ == '__main__':
    main()