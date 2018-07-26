from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def initialise_logistic_model(config, penalty='l2', C=1.0):
    random_seed = config.RANDOM_SEED
    model = LogisticRegression(random_state=random_seed, penalty=penalty, C=C)
    return model


def assemble_models():
    logistic_regresion = initialise_logistic_model(config)
    return [logistic_regresion]

def fit_models(X_train, X_val, y_train, y_val, models):
    training_accuracies = {}
    validation_accuracies = {}
    trained_models = {}
    for model in models:
        name = model.__class__.__name__
        model.fit(X, y)

        y_train_predicitions = model.predict(X)
        y_val_predicitions = model.predict(X_val)

        training_accuracy = accuracy_score(y_train, y_train_predicitions)
        validation_accuracy = accuracy_score(y_val, y_val_predicitions)
        
        training_accuracies[name] = training_accuracy
        validation_accuracies[name] = validation_accuracy
        trained_models[name] = model

    return trained_models, training_accuracies, validation_accuracies

def model_tuning(models, param_grids, X, X_val, y, y_val):
        best_models = {}
        best_parameters = {}
        train_scores = {}
        validation_scores = {}
    for model in models:
        name = model.__class__.__name__
        clf = GridSearchCV(model, param_grids[name])
        best_model = clf.fit(X, y)
        best_parameter = best_model.best_estimator_.get_params()

        y_train_prediction = model.predict(X)
        y_validation_prediction = model.predict(X_val)

        train_score = accuracy_score(y_train, y_train_prediction)
        validation_score = accuracy_score(y_val, y_val_predicition)

        best_models[name] = best_model
        best_parameters[name] = best_parameter
        train_scores[name] = train_score
        validation_scores[name] = validation_score

    return best_models, best_parameters, train_scores, validation_scores


def feature_selection():
    return 


def main():
    print("Nothing to see here: Model")
    return

if __name__ == '__main__':
    main()