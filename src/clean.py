import pandas as pd
import config 

def clean(dataframe, columns_to_drop=[], maps={}):
    #FILL
    df_filled = fill_columns(dataframe)

    #MAP
    df_mapped = map_columns(df_filled, maps)

    #DROP
    df_dropped = df_mapped.drop(columns_to_drop, axis=1)

    return df_dropped

def fill_columns(dataframe):
    #Age
    age_value = dataframe["Age"].median()
    dataframe["Age"] = dataframe["Age"].fillna(age_value)

    #Embarked
    embarked_value = 'U'
    dataframe["Embarked"] = dataframe["Embarked"].fillna(embarked_value)

    #Fare
    fare_value = dataframe["Fare"].median()
    dataframe["Fare"] = dataframe["Fare"].fillna(fare_value)

    return dataframe

def map_columns(dataframe, maps):
    # maps = {"column_name" : {'s' : 0, ...}} = {"column_name" : map}
    for col, map_of_column in maps.items():
        dataframe[col] = dataframe[col].map(map_of_column)
    return dataframe

def main():
    print("Nothing to see here: Clean")
    return

if __name__ == '__main__':
    main()