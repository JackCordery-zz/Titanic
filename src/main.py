import pandas as pd
import config
from clean import clean
from engineer import feature_engineer, reconcile_test_set
from pre_process import pre_process, load_data
from log import log_results
from model import assemble_models, fit_models, score_test_set
from model_validation import hyperparameter_tuning, feature_selection, fit_modelsCV, plot_roc, plot_prec_rec
import numpy as np

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
    stats_hyper = hyperparameter_tuning(models, {"LogisticRegression":{"C":np.logspace(0, 4, 10), "penalty": ["l1","l2"]}}, data["X_train"], data["y_train"] )
    stats_feature = feature_selection(models, data["X_train"], data["y_train"])

    df_feature, df_hyper, full_results = log_results(stats_feature, stats_hyper)
    print(df_feature.head())
    print(df_hyper.head())
    print(full_results["LogisticRegression"].head())

    """plt = plot_roc(models[0], data["X_train"], data["y_train"])
                plt.show()"""

    df_score = score_test_set(models[0], data["X_test"], data["id_test"], ["PassengerId", "Survived"])

    df_score.to_csv(config.OUTPUT_PATH + "submission.csv", sep=',', index=False)

    """plt2 = plot_prec_rec(models[0], data["X_train"], data["y_train"])
                plt2.show()"""

    return

if __name__ == '__main__':
    main()