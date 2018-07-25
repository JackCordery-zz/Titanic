import config
import pandas as pd
from sklearn.model_selection import train_test_split

def pre_process(df_train, df_test, config):
    split = 1 - config.TRAIN_VAL_SPLIT
    random_seed = config.RANDOM_SEED

    X = df_train.drop("Survived", axis=1)
    y = df_train["Survived"]

    X_test = df_test.drop("PassengerId", axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split,
                                                      random_state=random_seed)

    data = {"X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val}

    return data

def main():
    print("Nothing to see here: Pre-process")
    return

if __name__ == '__main__':
    main()