import pandas as pd
import config
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import validation_curve, GridSearchCV
from sklearn.feature_selection import RFE

def loadData(config):
    trainingSetPath = config.TRAIN_INPUT_PATH
    testSetPath = config.TEST_INPUT_PATH

    trainingDf = pd.read_csv(trainingSetPath)
    testDf = pd.read_csv(testSetPath)

    return trainingDf, testDf

def categoriseAge(age):

    if age <=16:
        return 0
    elif (age > 16) & (age <= 32):
        return 1
    elif (age > 32) & (age <= 48):
        return 2
    elif (age > 48) & (age <= 64):
        return 3
    else:
        return 4

def getTitle(name):
    titleSearch = re.search(' ([A-Za-z]+)\.', name)

    if titleSearch:
        return titleSearch.group(1)
    else:
        return ""

def categoriseTitles(name):
    rareTitles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major','Rev',
                  'Sir', 'Jonkheer', 'Dona']
    missTitles = ['Mlle', 'Ms']
    mrsTitles = ['Mme']

    if name in rareTitles:
        return "Rare"
    elif name in missTitles:
        return "Miss"
    elif name in mrsTitles:
        return "Mrs"
    else:
        return name

def isAlone(familySize):
    if familySize == 1:
        return 1
    else:
        return 0

def categoriseFare(fare):
    if fare <= 7.91:
        return 0
    elif (fare > 7.91) & (fare <= 14.454):
        return 1
    elif (fare > 14.454) & (fare <= 31):
        return 2
    elif fare > 31:
        return 3

def cleanData(dataFrame, mode="train"):

    columnsToUse = ["Age", "Fare", "Sex", "SibSp", "Parch", "Name", "Embarked",
                    "Pclass"]

    if mode == "test":
        columnsToUse = ["PassengerId"] + columnsToUse
    else:
        pass

    try:
        filteredDf = dataFrame[["Survived"] +  columnsToUse].copy()
    except:
        filteredDf = dataFrame[columnsToUse].copy()

    filteredDf["Age"] = filteredDf["Age"].fillna(filteredDf["Age"].median())
    filteredDf["Embarked"] = filteredDf["Embarked"].fillna('S')
    filteredDf["Fare"] = filteredDf["Fare"].fillna(filteredDf["Fare"].median())

    filteredDf["Fare"] = filteredDf["Fare"].apply(categoriseFare).astype(int)

    filteredDf["Embarked"] = filteredDf["Embarked"].map({'S': 0, 'C':1, 'Q':2}).astype(int)

    filteredDf["Name"] = filteredDf["Name"].apply(getTitle).apply(categoriseTitles)
    filteredDf["Name"] = filteredDf["Name"].map({"Mr": 1,
                                                   "Miss": 2,
                                                   "Mrs": 3,
                                                   "Master": 4,
                                                   "Rare": 5})


    filteredDf["Sex"] = filteredDf["Sex"].map({'female': 0, 'male': 1 }).astype(int)

    filteredDf["FamilySize"] = filteredDf['SibSp'] + filteredDf['Parch'] + 1
    filteredDf["isAlone"] = filteredDf["FamilySize"].apply(isAlone).astype(int)

    filteredDf["Age"] = filteredDf["Age"].apply(categoriseAge)



    dropColumns = ["Embarked"]
    filteredDf = filteredDf.drop(dropColumns, axis=1)

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

def scoreTestSet(model, df):
    passengerId = df["PassengerId"].astype(int).values
    X = df.drop(["PassengerId"], axis=1)
    predictions = model.predict(X)

    submission = np.vstack((passengerId, predictions)).T

    return submission

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

def pipeline(config, modelComment, mode="train"):

    trainingDf, testDf = loadData(config)

    cleanTrainingDf = cleanData(trainingDf, mode="train")
    cleanTestDf = cleanData(testDf, mode="test")

    X_train, X_val, y_train, y_val = splitData(cleanTrainingDf, config)

    model = initialiseModel(config)
    fittedModel = fitModel(model, X_train, y_train)

    modelStats = modelValidation(fittedModel, X_train, X_val, y_train, y_val)
    modelStats = {**modelStats, **{"comment": modelComment, "mode": mode}}

    if mode=="train":
        print("Running in TRAINING MODE")
        submission = scoreTestSet(fittedModel, cleanTestDf)
        submissionFilePath = config.OUTPUT_PATH + "submission.csv"
        np.savetxt(submissionFilePath, submission, fmt='%i', delimiter=',',
               header='PassengerID,Survived', comments='')
    elif mode=="test":
        print("Running in TESTING MODE")
        featurePipeline(config)
        logisticGridSearch(X_train, y_train, X_val, y_val)

    print(modelStats)
    logRun(modelStats, config)

    data = {"X_train" : X_train,
            "X_val" : X_val,
            "y_train" : y_train,
            "y_val" : y_val}

    return fittedModel, modelStats, data 

def featureSelection(model,rangeFeatures, X, X_val, y, y_val):
    stats = {}
    print("See the affect of n_features and combination of features:")
    print("-----------------------------------------------------------")
    for n in rangeFeatures:
        selector = RFE(model, n)
        fit = selector.fit(X, y)
        individualStats = {"n": n,
                           "n_features": fit.n_features_,
                           "support": fit.support_,
                           "ranking": fit.ranking_,
                           "score_train": fit.score(X, y),
                           "score_val":  fit.score(X_val, y_val)}
        print(individualStats)

    print('------------------------------------------------------------')
    return

def logisticGridSearch(X, y, X_val, y_val):
    logistic = LogisticRegression()

    penaltySpace = ['l1', 'l2']
    cSpace = np.logspace(0,4,10)

    hyperparameters = dict(C=cSpace, penalty=penaltySpace)

    clf = GridSearchCV(logistic, hyperparameters, cv=4, verbose=0)

    bestModel = clf.fit(X, y)
    print('See results for GridSearch:')
    print('-------------------------------------------------------------')
    print('Best Penalty:', bestModel.best_estimator_.get_params()['penalty'] )
    print('Best C:', bestModel.best_estimator_.get_params()['C'] )
    print('Model Train Score:', bestModel.score(X, y))
    print('Model Test Score:', bestModel.score(X_val, y_val))
    print('-------------------------------------------------------------')

    return bestModel

def featurePipeline(config):
    trainingDf, testDf = loadData(config)

    cleanTrainingDf = cleanData(trainingDf)
    cleanTestDf = cleanData(testDf)

    X_train, X_val, y_train, y_val = splitData(cleanTrainingDf, config)

    model = initialiseModel(config)

    featureSelection(model, range(1,X_train.shape[1] + 1), X_train, X_val,
                     y_train, y_val)
    return



def main():

    modelComment = input("Please insert a message to describe model:")

    model, modelStats, data = pipeline(config, modelComment, mode="test")

    plt = plotValidationCurve(model, data["X_val"], data["y_val"], "C",np.logspace(0,4,10) , "accuracy")
    #plt.show()
   

    return 1 

if __name__ == '__main__':
    main()