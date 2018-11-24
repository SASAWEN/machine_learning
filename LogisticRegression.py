"""
Created on Tue Nov  6 20:33:51 2018

@author: SASAWEN
"""
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

def sigmoid(X):
    return 1.0/(1+np.exp(-X))

def ReLU(X):
    return np.maximum(0, X)

def rdGrandAscent(dataMat, labels, test_x = [], test_label = [], maxIter = 150):
    # print(dataMat)
    if (len(test_x) > 0):
        test_size_x, test_size_y = np.shape(test_x)
    else:
        test_size_x = 0
        test_size_y = 0

    m,n = np.shape(dataMat)
    weights = np.ones(n)
    rates = []
    for j in range(maxIter):
        print(j)
        dataIndex = list(range(m))
        for i in range(m):
            # generally small
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            # print(randIndex)
            h = sigmoid(np.sum(dataMat[randIndex]*weights))
            error = labels[randIndex] - h
            weights += alpha*error*dataMat[randIndex]
            # delete the direction that has been scanned
            del(dataIndex[randIndex])

        if (test_size_x > 0):
            error_count = 0
            for i in range(test_size_x):
                if int(classifyVector(test_x[i], weights)) != int(test_label[i]):
                    error_count += 1
            print('Iter:'+str(j)+ ' error_rate: ' + str(error_count / float(test_size_x)))
            rates.append(error_count / float(test_size_x))
    if (test_size_x > 0):
        from matplotlib import pyplot as plt
        plt.plot(rates)
        plt.xlabel('iter')
        plt.ylabel('error_rate')
        # plt.xticks(xlabels)
        plt.show()

    return weights

def classifyVector(X, weights):
    prob = sigmoid(sum(X*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def trainData(file):
    train_data = pd.read_csv(file)
    train_X = train_data.iloc[:, :20]
    train_Label = train_data.iloc[:, 20]
    train_Label = train_Label.replace('female', 1)
    train_Label = train_Label.replace('male', 0)
    return train_X.values, train_Label.values

def submitData(file):
    submit_data = pd.read_csv(file)
    return submit_data.values

def logistic(train_file, maxIter = 20):
    train_X, train_label = trainData(train_file)
    # x为数据集的feature熟悉，y为label.
    train_x, test_x, train_label, test_label = train_test_split(train_X, train_label, test_size=0.2)
    test_size_x, test_size_y = np.shape(test_x)
    weights = rdGrandAscent(train_x, train_label, test_x, test_label, maxIter)

def submit(train_file, submit_file, maxIter = 20):
    train_X, train_label = trainData(train_file)
    test_x = submitData(submit_file)
    test_size_x, test_size_y = np.shape(test_x)
    weights = rdGrandAscent(train_X, train_label, maxIter = maxIter)
    pred = []
    for i in range(test_size_x):
        pred.append(int(classifyVector(test_x[i], weights)))
    pred = pd.DataFrame(pred)
    pred = pred.replace(0.0, 'male')
    pred = pred.replace(1.0, 'female')
    pred.to_csv('submission.csv', header = False, index=False)

# logistic('Train.csv', 2000)
submit('Train.csv', 'Test.csv', maxIter = 2000)
