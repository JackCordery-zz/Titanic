import config
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def initialise_logistic_model(config, penalty='l2', C=1.0):
    random_seed = config.RANDOM_SEED
    model = LogisticRegression(random_state=random_seed, penalty=penalty, C=C)
    return model


def assemble_models(config):
    logistic_regresion = initialise_logistic_model(config)
    return [logistic_regresion]



def fit_models(X_train, X_val, y_train, y_val, models):
    training_accuracies = {}
    validation_accuracies = {}
    trained_models = {}
    for model in models:
        name = model.__class__.__name__
        model.fit(X_train, y_train)
        y_train_predicitions = model.predict(X_train)
        y_val_predicitions = model.predict(X_val)

        training_accuracy = accuracy_score(y_train, y_train_predicitions)
        validation_accuracy = accuracy_score(y_val, y_val_predicitions)
        
        training_accuracies[name] = training_accuracy
        validation_accuracies[name] = validation_accuracy
        trained_models[name] = model

    return trained_models, training_accuracies, validation_accuracies


def main():
    print("Nothing to see here: Model")
    return

if __name__ == '__main__':
    main()