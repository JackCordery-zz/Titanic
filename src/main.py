import pandas as pd
import config
import csv
from datetime import datetime
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

def loadData(config):
    trainingSetPath = config.TRAIN_INPUT_PATH
    testSetPath = config.TEST_INPUT_PATH

    trainingDf = pd.read_csv(trainingSetPath)
    testDf = pd.read_csv(testSetPath)

    return trainingDf, testDf

def cleanData(dataFrame):
    try:
        filteredDf = dataFrame[["Survived", "Age", "Fare"]].copy()
    except:
        filteredDf = dataFrame[["Age","Fare"]].copy()

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

def modelValidation(model, X_val, y_val):
    modelScore = model.score(X_val, y_val)
    areaUnderCurve = calculateAuc(model, X_val, y_val)
    return {"modelScore" : modelScore, "areaUnderCurve" : areaUnderCurve}

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

def main():
    trainingDf, testDf = loadData(config)

    cleanTrainingDf = cleanData(trainingDf)
    cleanTestDf = cleanData(testDf)

    X_train, X_val, y_train, y_val = splitData(cleanTrainingDf, config)

    model = initialiseModel(config)
    fittedModel = fitModel(model, X_train, y_train)

    modelStats = modelValidation(fittedModel, X_val, y_val)

    print(modelStats)
    logRun(modelStats, config)

    return 1 

if __name__ == '__main__':
    main()