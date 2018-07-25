import pandas as pd
import config 

def clean(dataframe, columns_to_drop, maps):
    #FILL
    df_filled = fill_columns(dataframe)

    #DROP
    df_dropped = drop_columns(df_filled, columns_to_drop)

    #MAP
    df_mapped = map_columns(df_dropped, maps)

    return df_mapped

def fill_columns(dataframe):
    #Age
    age_value = dataframe["Age"].median()
    dataframe["Age"] = dataframe["Age"].fillna(age_value)

    #Embarked
    embarked_value = 'S'
    dataframe["Embarked"] = dataframe["Embarked"].fillna(embarked_value)

    #Fare
    fare_value = dataframe["Fare"].medain()
    dataframe["Fare"] = dataframe["Fare"].fillna(fare_value)

    return dataframe

def drop_columns(dataframe, columns_to_drop):
    return dataframe.drop(columns_to_drop, axis=1)

def map_columns(dataframe, maps):
    # maps = {"column_name" : {'s' : 0, ...}} = {"column_name" : map}
    for col, map_of_column in maps.items()
        dataframe[col] = dataframe[col].map(map_of_column).astype(int)
    return dataframe

def main():
    print("Nothing to see here: Clean")
    return

if __name__ == '__main__':
    main()