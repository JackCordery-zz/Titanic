import pandas as pd
import config
from clean import clean
from engineer import feature_engineer, reconcile_test_set
from pre_process import pre_process, load_data
from log import log_results
from model import assemble_models, fit_models, score_test_set
from model_validation import hyperparameter_tuning, feature_selection,\
    fit_modelsCV, plot_roc, plot_prec_rec, compute_cv_scores
import numpy as np


def experiment_process(data, models, param_grids):
    means, stds = fit_modelsCV(data["X_train"], data["y_train"], models)
    stats_hyper = hyperparameter_tuning(models, param_grids, data["X_train"],
                                        data["y_train"])
    stat_feature = feature_selection(models, data["X_train"], data["y_train"])
    df_feature, df_hyper, full_results = log_results(stat_feature, stats_hyper)

    means_cv, stds_cv = compute_cv_scores(models,
                                          data["X_train"],
                                          data["y_train"])
    print("means: ", means_cv)
    print("stds: ", stds_cv)

    pd.set_option('display.max_columns', None)
    print(data["X_train"].head())
    print(df_feature.head())
    print(df_hyper.head())
    print(full_results["LogisticRegression"].head())
    plt = plot_roc(models["LogisticRegression"], data["X_train"],
                   data["y_train"])
    plt.show()
    plt2 = plot_roc(models["LogisticRegressionCV"],
                    data["X_train"], data["y_train"])
    plt2.show()
    plt3 = plot_prec_rec(models["LogisticRegression"], data["X_train"],
                         data["y_train"])
    plt3.show()
    plt4 = plot_prec_rec(models["LogisticRegressionCV"], data["X_train"],
                         data["y_train"])
    plt4.show()
    return


def submission_process(data, models, config):
    trained_models, training_acc = fit_models(models, data["X_train"],
                                              data["y_train"])

    df_scored = score_test_set(trained_models[config.MODEL], data["X_test"],
                               data["id_test"], ["PassengerId", "Survived"],
                               config.THRESHOLD)
    df_scored.to_csv(config.OUTPUT_PATH + "submission.csv",
                     sep=',', index=False)

    print("Submission saved to: {}".format(config.OUTPUT_PATH +
                                           "submission.csv"))
    print("Training Accuracy: {}".format(training_acc))
    return


def process(config, mode="experiment"):
    df_train, df_test = load_data(config)

    # Custom transformations can go here
    #
    #
    #
    for df in [df_train, df_test]:
        # Change cabin to exits, not exists
        df["Cabin"] = df["Cabin"].apply(lambda x: 0 if pd.isnull(x) else 1)

    # Information for standard transforamtion here, noting that the process
    # is fill, map, drop (The objects below are in order of process)

    fill_values = {"Age": df_train["Age"].median(), "Embarked": "S",
                   "Fare": df_train["Fare"].mean()}
    maps = {"Sex": {"female": 0, "male": 1}}
    columns_to_drop = ["Ticket", "Embarked"]

    columns_to_engineer = ["family_size", "is_alone"]
    features_to_ohe = ["Age", "Name", "Pclass"]

    # Information for modelling stages
    param_grids = {"LogisticRegression": {"C": np.logspace(0, 4, 10),
                   "penalty": ["l1", "l2"]}}

    df_train_clean = clean(df_train, fill_values=fill_values,
                           columns_to_drop=columns_to_drop, maps=maps)
    df_test_clean = clean(df_test, fill_values=fill_values,
                          columns_to_drop=columns_to_drop, maps=maps)

    df_train = feature_engineer(df_train_clean,
                                columns_to_engineer=columns_to_engineer,
                                features_to_ohe=features_to_ohe)
    df_test = feature_engineer(df_test_clean,
                               columns_to_engineer=columns_to_engineer,
                               features_to_ohe=features_to_ohe)
    df_test = reconcile_test_set(df_train, df_test)

    data = pre_process(df_train, df_test, config)
    models = assemble_models(config, penalty='l1')

    if mode == "experiment":
        experiment_process(data, models, param_grids)
    elif mode == "submission":
        submission_process(data, models, config)

    return


def main():
    process(config, mode="experiment")
    return


if __name__ == '__main__':
    main()
