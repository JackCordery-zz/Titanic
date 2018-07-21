import pandas as pd
import config
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import validation_curve

def loadData(config):
    trainingSetPath = config.TRAIN_INPUT_PATH
    testSetPath = config.TEST_INPUT_PATH

    trainingDf = pd.read_csv(trainingSetPath)
    testDf = pd.read_csv(testSetPath)

    return trainingDf, testDf

def transformGender(gender):
    gender = gender.lower()
    if gender == "male":
        return 0
    elif gender == "female":
        return 1
    else:
        return None

def cleanData(dataFrame):
    try:
        filteredDf = dataFrame[["Survived", "Age", "Fare", "Sex"]].copy()
    except:
        filteredDf = dataFrame[["Age","Fare", "Sex"]].copy()

    filteredDf["Sex"] = filteredDf["Sex"].apply(transformGender)

    filteredDf.dropna(inplace=True)

    return filteredDf

def splitData(trainingDf, config):
    trainValidationSplit = config.TRAIN_VAL_SPLIT
    randomSeed = config.RANDOM_SEED

    X = trainingDf.drop("Survived", axis=1)
    y = trainingDf["Survived"]

    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                     test_size=(1-trainValidationSplit),
                                     random_state=randomSeed)

    return X_train, X_val, y_train, y_val

def initialiseModel(config):
    randomSeed = config.RANDOM_SEED
    model = LogisticRegression(random_state=randomSeed)
    return model

def fitModel(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def calculateAuc(model, X_val, y_val):
    y_pred = model.predict_proba(X_val)
    y_predPositiveClass = y_pred[:,1]
    areaUnderCurve = roc_auc_score(y_val, y_predPositiveClass)
    return areaUnderCurve

def modelValidation(model, X_train, X_val, y_train, y_val):
    modelScoreValidationSet = model.score(X_val, y_val)
    modelScoreTrainingSet = model.score(X_train, y_train)
    areaUnderCurve = calculateAuc(model, X_val, y_val)
    return {"modelScoreValidationSet" : modelScoreValidationSet,
            "modelScoreTrainingSet" : modelScoreTrainingSet,
            "areaUnderCurve" : areaUnderCurve}

def plotLearningCurve(model, X, y, cv=None,trainSizes=[0.1, 0.33, 0.55, 0.78, 1. ]):
    plt.figure()
    plt.title("learning_curve")

    plt.xlabel("Training examples")
    plt.ylabel("Score")

    trainSizes, trainScores, validScores = learning_curve(
        model, X, y, cv=cv, train_sizes=trainSizes)

    trainScoresMean = np.mean(trainScores, axis=1)
    validScoresMean = np.mean(validScores, axis=1)

    trainScoresStd = np.std(trainScores, axis=1)
    validScoresStd = np.std(validScores, axis=1)

    plt.grid()

    #Plot band of variance
    plt.fill_between(trainSizes, trainScoresMean - trainScoresStd,
                                 trainScoresMean + trainScoresStd,
                                 alpha=0.1, color='r',
                                 label="Training Score")
    plt.fill_between(trainSizes, validScoresMean - validScoresStd,
                                 validScoresMean + validScoresStd,
                                 alpha=0.1, color='g',
                                 label="Cross-Validation Score")
    #Plot mean as point
    plt.plot(trainSizes, trainScoresMean, 'o-', color='r')
    plt.plot(trainSizes, validScoresMean, 'o-', color='g')

    plt.legend(loc='best')
    return plt

def plotValidationCurve(model, X, y, paramName, paramRange, scoring, cv=None, ylim=[0.5,1]):

    trainScores, validScores = validation_curve(model, X, y,
                                                param_name=paramName,
                                                param_range=paramRange,
                                                scoring=scoring,
                                                cv=cv
                                                )

    trainScoresMean = np.mean(trainScores, axis=1)
    validScoresMean = np.mean(validScores, axis=1)

    trainScoresStd = np.std(trainScores, axis=1)
    validScoresStd = np.std(validScores, axis=1)

    plt.title("validation_curve")
    plt.xlabel(paramName)
    plt.ylabel("Score")
    plt.ylim(ylim)

    plt.plot(paramRange, trainScoresMean, label="Training score",
             color="darkorange")
    plt.fill_between(paramRange, trainScoresMean - trainScoresStd,
                     trainScoresMean + trainScoresStd, alpha=0.1,
                     color="darkorange")
    plt.plot(paramRange, validScoresMean, label="Cross-validation score",
                 color="navy")
    plt.fill_between(paramRange, validScoresMean - validScoresStd,
                     validScoresMean + validScoresStd, alpha=0.1,
                     color="navy")
    plt.legend(loc="best")
    return plt

def plotROC(model, X, y):

    yPredicted = model.predict_proba(X)

    print(roc_curve(y, yPredicted[:,1]))

    fpr, tpr, _ = roc_curve(y, yPredicted[:,1])
    areaUnderCurve = roc_auc_score(y, yPredicted[:,1])

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC Curve (area = {})'.format(areaUnderCurve))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Reciever Operating Characteristic')
    plt.legend(loc="lower right")
    return plt

def logRun(modelStats, config):
    runsFilePath = config.LOG_PATH

    configVariables = {"dateTime" : str(datetime.now()),
                        "trainTestSplit" : config.TRAIN_VAL_SPLIT,
                        "randomSeed" : config.RANDOM_SEED,
                        "model" : config.MODEL}

    logInputs = {**configVariables, **modelStats}
    with open(runsFilePath + "model_runs.csv", "a") as file:
        writer = csv.writer(file)
        writer.writerow(logInputs.values())        
    return

def logTrainingDataStats(dataframe):
    return

def pipeline(config):

    trainingDf, testDf = loadData(config)

    cleanTrainingDf = cleanData(trainingDf)
    cleanTestDf = cleanData(testDf)

    X_train, X_val, y_train, y_val = splitData(cleanTrainingDf, config)

    model = initialiseModel(config)
    fittedModel = fitModel(model, X_train, y_train)

    modelStats = modelValidation(fittedModel, X_train, X_val, y_train, y_val)

    print(modelStats)
    logRun(modelStats, config)

    data = {"X_train" : X_train,
            "X_val" : X_val,
            "y_train" : y_train,
            "y_val" : y_val}

    return fittedModel, modelStats, data 


def main():

    model, modelStats, data = pipeline(config)

    plt = plotROC(model, data["X_val"], data["y_val"])
    plt.show()

    return 1 

if __name__ == '__main__':
    main()