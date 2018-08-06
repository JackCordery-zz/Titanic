import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score
from model_validation import binary_class


def initialise_logistic_model(config, penalty='l2', C=1.0):
    random_seed = config.RANDOM_SEED
    model = LogisticRegression(random_state=random_seed, penalty=penalty, C=C)
    return model


def assemble_models(config, penalty='l2', C=1.0):
    logistic_regresion = initialise_logistic_model(config, penalty, C)
    logistic_cv = LogisticRegressionCV(random_state=config.RANDOM_SEED, cv=10)
    return {"LogisticRegression": logistic_regresion,
            "LogisticRegressionCV": logistic_cv}


def score_test_set(model, X_test, id_col, columns, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y = [binary_class(p, threshold) for p in y_proba]
    df = pd.DataFrame(np.vstack((id_col, y)).T)
    df.columns = columns
    return df


def fit_models(models, X_train, y_train):
    training_accuracies = {}
    trained_models = {}
    for _, model in models.items():
        name = model.__class__.__name__
        model.fit(X_train, y_train)
        y_train_predicitions = model.predict(X_train)

        training_accuracy = accuracy_score(y_train, y_train_predicitions)

        trained_models[name] = model
        training_accuracies[name] = training_accuracy

    return trained_models, training_accuracies


def main():
    print("Nothing to see here: Model")
    return


if __name__ == '__main__':
    main()
