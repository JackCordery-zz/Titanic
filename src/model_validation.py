import config
from itertools import compress
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.feature_selection import RFECV

sss = StratifiedShuffleSplit(n_splits=config.K_FOLD,
                             test_size=0.1,
                             random_state=config.RANDOM_SEED)

def fit_modelsCV(X, y, models):
    test_accuracies = {}
    trained_models = {}
    
    for train_index, test_index in sss.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        for model in models:
            name = model.__class__.__name__
            model.fit(X_train, y_train)
            y_test_predicitions = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_predicitions) 
            if name in test_accuracies:
                test_accuracies[name] += [test_accuracy]
            else:
                test_accuracies[name] = [test_accuracy]

    means = {name: np.mean(np.array(list_of_acc)) for name, list_of_acc in test_accuracies.items()}
    stds = {name: np.std(np.array(list_of_acc)) for name, list_of_acc in test_accuracies.items()}

    return means, stds

def hyperparameter_tuning(models, param_grids, X, y):
    # What do param_grids look like?
    
    best_parameters = {}
    scores = {}
    for model in models:
        name = model.__class__.__name__
        params = param_grids[name]
        param_names = list(params.keys())
        clf = GridSearchCV(model, params)
        clf.fit(X, y)
        results = clf.cv_results_
        best_parameter = clf.best_estimator_.get_params()
        best_parameter = {k:v for k,v in best_parameter.items() if k in param_names}
        best_score = clf.best_score_

        best_parameters[name] = best_parameter
        scores[name] = best_score

    stats = {"best_parameters": best_parameters, "scores": scores}

    return stata

def feature_selection(models, X, y):
    support = {}
    score_mean = {}
    score_std = {}
    transformed_X = {}
    features = {}
    total_number_features = X.shape[1]
    feature_names = list(X.columns)
    for model in models:
        for n in range(1, total_number_features):
            name = model.__class__.__name__
            fit = RFECV(model,step=n, cv=config.K_FOLD)
            x_transform = fit.fit_transform(X, y)
            name = name + "-" + str(fit.n_features_) 

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


def main():
    print("Nothing to find here: Model Validation")
    return

if __name__ == '__main__':
    main()