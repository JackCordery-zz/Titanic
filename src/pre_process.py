import config
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(config):
    training_set_path = config.TRAIN_INPUT_PATH
    test_set_path = config.TEST_INPUT_PATH

    df_train = pd.read_csv(training_set_path)
    df_test = pd.read_csv(test_set_path)

    return df_train, df_test

def pre_process(df_train, df_test, config, cols_drop=[]):
    split = 1 - config.TRAIN_VAL_SPLIT
    random_seed = config.RANDOM_SEED

    X = df_train.drop(["Survived", "PassengerId"] + cols_drop, axis=1)
    y = df_train["Survived"]

    id_test = df_test["PassengerId"]
    X_test = df_test.drop(["PassengerId"] + cols_drop, axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split,
                                                      random_state=random_seed)

    data = {"X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "id_test": id_test}

    return data

def main():
    print("Nothing to see here: Pre-process")
    return

if __name__ == '__main__':
    main()