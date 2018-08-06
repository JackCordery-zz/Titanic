import config
from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, roc_auc_score, make_scorer, \
    roc_curve, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit,\
    cross_val_score
from sklearn.feature_selection import RFECV
plt.style.use("ggplot")

sss = StratifiedShuffleSplit(n_splits=config.K_FOLD,
                             test_size=0.1,
                             random_state=config.RANDOM_SEED)


def compute_cv_scores(models, X, y, cv=sss, scoring="accuracy"):
    means = {}
    stds = {}
    for _, model in models.items():
        name = model.__class__.__name__
        cv_score = cross_val_score(model, X, y, cv=sss, scoring="accuracy")
        means[name] = np.mean(cv_score)
        stds[name] = np.std(cv_score)

    return means, stds


def fit_modelsCV(X, y, models):
    test_accuracies = {}

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        for _, model in models.items():
            name = model.__class__.__name__
            model.fit(X_train, y_train)
            y_test_predicitions = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_predicitions)
            if name in test_accuracies:
                test_accuracies[name] += [test_accuracy]
            else:
                test_accuracies[name] = [test_accuracy]

    means = {name: np.mean(np.array(list_of_acc)) for name, list_of_acc
             in test_accuracies.items()}
    stds = {name: np.std(np.array(list_of_acc)) for name, list_of_acc
            in test_accuracies.items()}

    return means, stds


def hyperparameter_tuning(models, param_grids, X, y, refit='accuracy_score'):
    # params_grid={model_name:{param_grid}, ...}
    best_parameters = {}
    scores = {}
    full_results = {}

    scorers = {"accuracy_score": make_scorer(accuracy_score),
               "precision_score": make_scorer(precision_score),
               "recall_score": make_scorer(recall_score),
               "auc_score": make_scorer(roc_auc_score)}

    columns_to_get_from_results = list(scorers.keys())
    prefixes_to_get = ["mean_test_", "mean_train_", "std_test_", "std_train_"]
    columns_to_get_from_results = [pre + col for pre in prefixes_to_get for col
                                   in columns_to_get_from_results]

    for model_name in param_grids.keys():
        model = models[model_name]
        name = model.__class__.__name__
        params = param_grids[name]
        param_names = list(params.keys())
        param_columns = ["param_" + p for p in param_names]
        columns_to_get = param_columns + columns_to_get_from_results
        clf = GridSearchCV(model, params, scoring=scorers, cv=sss, refit=refit,
                           return_train_score=True)
        clf.fit(X, y)
        result = pd.DataFrame(clf.cv_results_)[columns_to_get]
        best_parameter = clf.best_estimator_.get_params()
        best_parameter = {k: v for k, v in best_parameter.items() if
                          k in param_names}
        best_score = clf.best_score_
        best_parameters[name] = best_parameter
        scores[name] = best_score
        full_results[name] = result

    stats = {"best_parameters": best_parameters, "scores": scores,
             "full_results": full_results}

    return stats


def feature_selection(models, X, y):
    support = {}
    score_mean = {}
    score_std = {}
    transformed_X = {}
    features = {}
    total_number_features = X.shape[1]
    feature_names = list(X.columns)
    for _, model in models.items():
        for n in range(1, total_number_features):
            name = model.__class__.__name__
            fit = RFECV(model, step=n, cv=config.K_FOLD)
            x_transform = fit.fit_transform(X, y)
            name = name + "-" + str(fit.n_features_) + "-" + str(n)

            support[name] = fit.support_
            score_mean[name] = np.mean(fit.grid_scores_)
            score_std[name] = np.std(fit.grid_scores_)
            transformed_X[name] = x_transform
            features[name] = support_to_features(fit.support_, feature_names)

    stats = {"support": support,
             "features": features,
             "all_features": feature_names,
             "score_mean": score_mean, "score_std": score_mean,
             "transformed_X": transformed_X}

    return stats


def support_to_features(support, feature_names):
    features = list(compress(feature_names, support))
    return features


def plot_roc(model, X, y):

    y_predicted = model.predict_proba(X)

    fpr, tpr, _ = roc_curve(y, y_predicted[:, 1])
    area_under_curve = roc_auc_score(y, y_predicted[:, 1])

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC Curve (area = {})'.format(area_under_curve))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Reciever Operating Characteristic')
    plt.legend(loc="lower right")
    return plt


def plot_prec_rec(model, X, y):

    y_predicted = model.predict_proba(X)

    precision, recall, thresholds = precision_recall_curve(y, y_predicted[:,1])
    acc, thresholds_acc = acc_score_threshold(model, X, y)
    max_acc = max(acc)
    max_threshold = thresholds_acc[np.argmax(acc)]

    plt.figure()
    plt.plot(thresholds, precision[:-1], color='green', lw=2,
             label="Precision")
    plt.plot(thresholds, recall[:-1], color='navy', lw=2,
             label="Recall")
    plt.plot(thresholds_acc, acc, color='darkorange', linestyle='--',
             label="Accuracy (max: {}, threshold: {}"
             .format(max_acc, max_threshold))
    plt.xticks(np.arange(0, 1, 0.05), rotation=90)
    plt.yticks(np.arange(0, 1, 0.05))
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs Threshold')
    plt.legend(loc="best")
    return plt


def binary_class(prob, t):
    if prob >= t:
        return 1
    else:
        return 0


def acc_score_threshold(model, X, y):
    # TODO: This needs to be done with CV
    thresholds = np.arange(0.1, 0.95, 0.01)
    accuracies = []
    for t in thresholds:
        y_pred = model.predict_proba(X)[:, 1]
        y_pred_label = [binary_class(p, t) for p in y_pred]
        acc = [accuracy_score(y, y_pred_label)]
        accuracies += acc
    return accuracies, thresholds


def main():
    print("Nothing to find here: Model Validation")
    return


if __name__ == '__main__':
    main()
