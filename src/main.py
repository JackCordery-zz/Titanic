import pandas as pd
import config
from clean import clean

def load_data(config):
    training_set_path = config.TRAIN_INPUT_PATH
    test_set_path = config.TEST_INPUT_PATH

    df_train = pd.read_csv(training_set_path)
    df_test = pd.read_csv(test_set_path)

    return df_train, df_test

def main():
    df_train, df_test = load_data(config)

    df_train_clean = clean(df_train, maps={"Embarked":{'S': 0, 'C':1, 'Q':2}})

    print(df_train_clean.head())

    return

if __name__ == '__main__':
    main()