
"""
Created on Tue Nov  6 20:33:51 2018

@author: SASAWEN
"""
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split


def loadDataSet(file):
    train = pd.read_csv(file)
    train_data = train.drop(['minfun'], axis=1)
    train_data = train_data.drop(['mindom'], axis=1)
    data = train.iloc[:, 0:18]
    data = data.apply(lambda x:(x-x.min())/(x.max()-x.min()))
    data = data.values
    label = train.iloc[:, 18]
    label = label.replace('female', 1)
    label = label.replace('male', -1)
    label = label.values.T
    return data, label


def loadDataSetTest(file):
    data = []
    label = []
    train = pd.read_csv(file)
    data = train.iloc[1640:, 0:20].values
    label = train.iloc[1640:, 20]
    label = label.replace('female', 1)
    label = label.replace('male', -1)
    label = label.values.T
    return data, label


def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


class optStruct:
    def __init__(self, dataIn, label, C, tlr, kTup):
        self.X = dataIn
        self.label = label
        self.C = C
        self.tol = tlr  # torance of error measurement
        self.m = np.shape(dataIn)[0]
        self.alpha = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # error cache = [valid_or_not, error_measure_value]
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


#             print(self.K[:,i])

def calcEk(os_, k):
    f_Xk = np.mat(np.multiply(os_.alpha, os_.label).T) * os_.K[:, k] + os_.b
    float_label = os_.label[k]
    Ek = f_Xk - float_label  # error measure
    return Ek


def selectJ(i, os_, Ei):
    maxK = -1;
    maxDeltaE = 0;
    Ej = 0
    os_.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(os_.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(os_, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k;
                maxDeltaE = deltaE;
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, os_.m)
        Ej = calcEk(os_, j)
    return j, Ej


def updateEk(os_, k):
    Ek = calcEk(os_, k)
    os_.eCache[k] = [1, Ek]


def alphaChange(i, os_):
    Ei = calcEk(os_, i)
    print(os_.label[i])
    print(Ei)
    if ((os_.label[i] * Ei < -os_.tol) and (os_.alpha[i] < os_.C)) or \
            ((os_.label[i] * Ei > os_.tol) and (os_.alpha[i] > 0)):
        print("to change")
        j, Ej = selectJ(i, os_, Ei)
        # save old alpha_1 and alpha_2
        Iold = os_.alpha[i].copy()
        Jold = os_.alpha[j].copy()
        # new alpha_2 lmitation
        if (os_.label[i] != os_.label[j]):
            L = max(0, os_.alpha[j] - os_.alpha[i])
            H = min(os_.C, os_.C + os_.alpha[j] - os_.alpha[i])
        else:
            L = max(0, os_.alpha[j] + os_.alpha[i] - os_.C)
            H = min(os_.C, os_.alpha[j] + os_.alpha[i])
        if L == H:
            print("L=H")
            return 0
        eta = 2.0 * os_.K[i, j] - os_.K[i, i] - os_.K[j, j]
        if eta >= 0:
            print("eta>=0 eta=" + str(eta))
            return 0
        print('alpha_j:' + str(os_.alpha[j]) + ' label:' + str(os_.label[j]) + ' Ei-Ej:' + str(Ei - Ej) + ' eta:' + str(
            eta) + ' H,L:' + str(H) + ',' + str(L))
        os_.alpha[j] -= os_.label[j] * (Ei - Ej) / eta
        os_.alpha[j] = clipAlpha(os_.alpha[j], H, L)
        print('alpha_j:' + str(os_.alpha[j]))
        updateEk(os_, j)
        if (abs(os_.alpha[j] - Jold) < os_.tol):
            print("less than tolerance")
            return 0
        os_.alpha[i] += os_.label[j] * os_.label[i] * (Jold - os_.alpha[j])
        updateEk(os_, i)
        b1 = os_.b - Ei - os_.label[i] * (os_.alpha[i] - Iold) * \
             os_.K[i, i] - os_.label[j] * \
             (os_.alpha[j] - Jold) * os_.K[i, j]
        b2 = os_.b - Ej - os_.label[i] * (os_.alpha[i] - Iold) * \
             os_.K[i, j] - os_.label[j] * \
             (os_.alpha[j] - Jold) * os_.K[j, j]
        if (0 < os_.alpha[i]) and (os_.C > os_.alpha[i]):
            os_.b = b1
        elif (0 < os_.alpha[j]) and (os_.C > os_.alpha[j]):
            os_.b = b2
        else:
            os_.b = (b1 + b2) / 2.0
        return 1
    else:
        print("alpha not in ")
        return 0


def SMO(dataIn, label, C, tol, maxIter, kTup=('lin', 0), testData=[], testLabel=[]):
    if (len(testData) > 0):
        test_size_x, test_size_y = np.shape(testData)
    else:
        test_size_x = 0
        test_size_y = 0
    errorRates = []
    oS = optStruct(np.mat(dataIn), np.mat(label).transpose(), C, tol, kTup)
    iter = 0
    entireSet = True
    alphaChanged = 0
    if (maxIter == -1):
        while ((alphaChanged > 0) or (entireSet)):
            alphaChanged = 0
            if entireSet:
                for i in range(oS.m):
                    alphaChanged += alphaChange(i, oS)
                    print("entireSet iter:" + str(iter) + " i:" + str(i) + " changed" + str(alphaChanged))
                iter += 1
            else:
                nonBound = np.nonzero((oS.alpha.A > 0) * (oS.alpha.A < C))[0]
                for i in nonBound:
                    alphaChanged += alphaChange(i, oS)
                    print("nonBound iter:" + str(iter) + " i:" + str(i) + " changed" + str(alphaChanged))
                iter += 1
            if entireSet:
                entireSet = False
            elif (alphaChanged == 0):
                entireSet = True
            print("iter:" + str(iter))

            if (test_size_x > 0):
                dataMat = np.mat(dataIn)
                labelMat = np.mat(label).transpose()
                SVind = np.nonzero(oS.alpha.A > 0)[0]
                SVs = dataMat[SVind]  # support vectors
                labelSV = labelMat[SVind]
                print("There are %d Support vectors." %np.shape(SVs)[0])
                dataMat = np.mat(testData);
                m, n = np.shape(dataMat)
                errorCount1 = 0
                for i in range(m):
                    kernelEval = kernelTrans(SVs, dataMat[i, :], kTup)
                    predict = kernelEval.T * np.multiply(labelSV, oS.alpha[SVind]) + oS.b
                    print(predict)
                    print(label[i])
                    if np.sign(predict) != np.sign(testLabel[i]):
                        errorCount1 += 1
                test_error_rate = float(errorCount1 / m)
                errorRates.append(test_error_rate)
                print("Testing error rate is %d", test_error_rate)
    else:
        while (iter < maxIter) and ((alphaChanged > 0) or (entireSet)):
            alphaChanged = 0
            if entireSet:
                for i in range(oS.m):
                    alphaChanged += alphaChange(i, oS)
                    print("entireSet iter:" + str(iter) + " i:" + str(i) + " changed" + str(alphaChanged))
                iter += 1
            else:
                nonBound = np.nonzero((oS.alpha.A > 0) * (oS.alpha.A < C))[0]
                for i in nonBound:
                    alphaChanged += alphaChange(i, oS)
                    print("nonBound iter:" + str(iter) + " i:" + str(i) + " changed" + str(alphaChanged))
                iter += 1
            if entireSet:
                entireSet = False
            elif (alphaChanged == 0):
                entireSet = True
            print("iter:" + str(iter))

            if (test_size_x > 0):
                dataMat = np.mat(dataIn)
                labelMat = np.mat(label).transpose()
                SVind = np.nonzero(oS.alpha.A > 0)[0]
                SVs = dataMat[SVind]  # support vectors
                labelSV = labelMat[SVind]
                print("There are %d Support vectors." %SVs[0])
                dataMat = np.mat(testData);
                m, n = np.shape(dataMat)
                errorCount1 = 0
                for i in range(m):
                    kernelEval = kernelTrans(SVs, dataMat[i, :], kTup)
                    predict = kernelEval.T * np.multiply(labelSV, oS.alpha[SVind]) + oS.b
                    print(predict)
                    print(label[i])
                    if np.sign(predict) != np.sign(testLabel[i]):
                        errorCount1 += 1
                test_error_rate = float(errorCount1 / m)
                errorRates.append(test_error_rate)
                print("Testing error rate is %f" %test_error_rate)
        print(alphaChanged)
    if (test_size_x > 0):
        from matplotlib import pyplot as plt
        plt.plot(errorRates)
        plt.xlabel('iter')
        plt.ylabel('error_rate')
        # plt.xticks(xlabels)
        plt.show()

    return oS.b, oS.alpha


def calcWs(alpha, dataArr, label):
    X = np.mat(dataArr);
    label = np.mat(label).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alpha[i] * label[i], X[i, :].T)
    return w


def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        print("Kernel is not recognized!")
    return K

def submit( k1=10, C=200, tol=0.0001, iter=-1):
    data, label = loadDataSet('sex/VoiceGenderTrain.csv')
    print("Data finish!")
    b, alpha = SMO(data, label, C, tol, iter, ('rbf', k1))
    print("SMO READY")
    dataMat = np.mat(data);
    labelMat = np.mat(label).transpose()
    SVind = np.nonzero(alpha.A > 0)[0]
    SVs = dataMat[SVind]  # support vectors
    labelSV = labelMat[SVind]
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(SVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alpha[SVind]) + b
        print(predict)
        print(label[i])
        if np.sign(predict) != np.sign(label[i]):
            errorCount += 1
    train_error_rate = float(errorCount / m)
    print("Training error rate is %d", train_error_rate)

    test_x = pd.read_csv('sex/VoiceGenderTest.csv')
    Test_x = test_x.apply(lambda x:(x-x.min())/(x.max()-x.min()))
    Test_x = Test_x.values
    dataMat = np.mat(Test_x);
    m, n = np.shape(dataMat)
    predicts = []
    for i in range(m):
        kernelEval = kernelTrans(SVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alpha[SVind]) + b
        predicts.append(np.sign(predict[0,0]))

    print(predicts)
    pred = pd.DataFrame(predicts)
    pred = pred.replace(-1.0, 'male')
    pred = pred.replace(1.0, 'female')
    print(pred)
    pred.to_csv('sex/submission.csv', index = False, header = False)



def Ebf(k1=10, C=200, tol=0.0001, iter=-1):
    data, label = loadDataSet('sex/VoiceGenderTrain.csv')
    print("Data finish!")
    # x为数据集的feature熟悉，y为label.
    data, Testdata, label, Testlabel = train_test_split(data, label, test_size=0.2)


Ebf(0.1, 10, 0.0001, -1)

