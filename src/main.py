import pandas as pd
import config
from clean import clean
from engineer import feature_engineer, reconcile_test_set
from pre_process import pre_process
from model import assemble_models, fit_models, model_tuning, feature_selection, fit_modelsCV

def load_data(config):
    training_set_path = config.TRAIN_INPUT_PATH
    test_set_path = config.TEST_INPUT_PATH

    df_train = pd.read_csv(training_set_path)
    df_test = pd.read_csv(test_set_path)

    return df_train, df_test

def main():
    df_train, df_test = load_data(config)

    columns_to_drop = ["Embarked", "Ticket", "Cabin"]
    maps = {"Embarked": {'S': 0, 'C':1, 'Q':2},
            "Sex":{"female": 0, "male": 1}}
    columns_to_engineer=["family_size", "is_alone"]
    features_to_ohe=["Age", "Fare"]

    df_train_clean = clean(df_train, columns_to_drop=columns_to_drop, maps=maps)
    df_test_clean = clean(df_test, columns_to_drop=columns_to_drop, maps=maps)

    df_train = feature_engineer(df_train_clean, columns_to_engineer=columns_to_engineer, features_to_ohe=features_to_ohe)
    df_test = feature_engineer(df_test_clean, columns_to_engineer=columns_to_engineer, features_to_ohe=features_to_ohe)
    df_test = reconcile_test_set(df_train, df_test)

    data = pre_process(df_train, df_test, config)

    models = assemble_models(config)
    means, stds = fit_modelsCV(data["X_train"], data["y_train"], models)


    print(means)
    print(stds)
    return

if __name__ == '__main__':
    main()