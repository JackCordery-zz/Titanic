import pandas as pd
import config 

def clean(dataframe, fill_values={}, columns_to_drop=[], maps={}, custom=[]):
    #FILL
    #TODO: Account for adding same fills to test instead of own stats

    df_filled = fill_columns(dataframe, fill_values)

    #MAP
    df_mapped = map_columns(df_filled, maps)

    #DROP
    df_dropped = df_mapped.drop(columns_to_drop, axis=1)

    return df_dropped

def fill_columns(dataframe, fill_values={}):
    for column_name, value in fill_values.items():
        dataframe[column_name] = dataframe[column_name].fillna(value)
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