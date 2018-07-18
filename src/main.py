import pandas as pd
import config

def pipelineLoadData(config):
    trainingSetPath = config.TRAIN_INPUT_PATH
    testSetPath = config.TEST_INPUT_PATH

    trainingDf = pd.read_csv(trainingSetPath)
    testDf = pd.read_csv(testSetPath)

    return trainingDf, testDf


def main():
    trainingDf, testDf = pipelineLoadData(config)
    print(trainingDf.head())
    return 1 

if __name__ == '__main__':
    main()