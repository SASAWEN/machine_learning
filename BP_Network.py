import math
import random
import numpy as  np
import pandas as pd
from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score

random.seed(0)

def rand(a, b):#随机数函数
    return (b - a) * random.random() + a

def make_matrix(m, n, fill=0.0):#矩阵生成函数
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat

def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))

def sigmoid_der(x):
    return x*(1-x)

class BPNetwork:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.hidden2_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.hidden2_cells = []
        self.output_cells = []
        self.input_weights = []
        self.hidden_weights = []
        self.output_weights = []
        self.input_correction = []
        self.hidden_correction = []
        self.output_correction = []

    def setup(self, ni, nh, nh2, no):
        # 初始化输入、隐层、输出元数
        self.input_n = ni + 1
        self.hidden_n = nh
        self.hidden2_n = nh2
        self.output_n = no
        # 初始化神经元
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.hidden2_cells = [1.0] * self.hidden2_n
        self.output_cells = [1.0] * self.output_n
        # 初始化权重矩阵
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.hidden_weights = make_matrix(self.hidden_n, self.hidden2_n)
        self.output_weights = make_matrix(self.hidden2_n, self.output_n)
        # 初始化权重
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.3, 0.3)
        for h in range(self.hidden_n):
            for o in range(self.hidden2_n):
                self.hidden_weights[h][o] = rand(-0.3, 0.3)
        for h in range(self.hidden2_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-0.3, 0.3)
        # 初始化偏置
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.hidden_correction = make_matrix(self.hidden_n, self.hidden2_n)
        self.output_correction = make_matrix(self.hidden2_n, self.output_n)

    def predict(self, inputs):
        # 激活输入层
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # 激活隐层
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # 激活隐层
        for j in range(self.hidden2_n):
            total = 0.0
            for i in range(self.hidden_n):
                total += self.hidden_cells[i] * self.hidden_weights[i][j]
            self.hidden2_cells[j] = sigmoid(total)
        # 激活输出层
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden2_n):
                total += self.hidden2_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        # 反向传播
        self.predict(case)
        # 求输出误差
        output_deltas = [0.0] * self.output_n
        # print(label)
        # print(self.output_cells)
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_der(self.output_cells[o]) * error
        # 求隐层误差
        hidden_deltas = [0.0] * self.hidden_n
        hidden2_deltas = [0.0] * self.hidden2_n
        for h in range(self.hidden2_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden2_deltas[h] = sigmoid_der(self.hidden2_cells[h]) * error
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.hidden2_n):
                error += hidden2_deltas[o] * self.hidden_weights[h][o]
            hidden_deltas[h] = sigmoid_der(self.hidden_cells[h]) * error
        # 更新输出权重
        for h in range(self.hidden2_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden2_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # 更新隐层权重
        for h in range(self.hidden_n):
            for o in range(self.hidden2_n):
                change = hidden2_deltas[o] * self.hidden_cells[h]
                self.hidden_weights[h][o] += learn * change + correct * self.hidden_correction[h][o]
                self.hidden_correction[h][o] = change
        # 更新输入权重
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # 求全局误差
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        # 训练神经网络
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn+1/(j+100), correct)


    def fit(self, x_test):  # 离散预测函数用于输出数据
        y_pre_1d = []
        for case in x_test:
            y_pred = self.predict(case)
            for i in range(len(y_pred)):
                if (y_pred[i] > 0.5):
                    y_pred[i] = 1
                else:
                    y_pred[i] = 0
            y_pre_1d.append(y_pred)
        return y_pre_1d

    def train_rate_show(self, cases, labels, x_test, y_test, limit=10000, learn=0.05, correct=0.1):
        scores = []
        # 训练神经网络
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn+1/(j+100), correct)
            y_pred = nn.fit(x_test)
            sc = accuracy_score(y_pred, y_test)
            scores.append(sc)
            print("iter:"+str(j)+" score:"+str(sc))
        from matplotlib import pyplot as plt
        plt.plot(scores)
        plt.show()

def loadDataSet(file):
    train = pd.read_csv(file)
    data = train.iloc[:, 0:20]
    data = data.apply(lambda x:(x-x.min())/(x.max()-x.min()))
    data = data.values
    label = train.iloc[:, 20]
    label = label.replace('female', 1)
    label = label.replace('male', 0)
    label = label.values.T
    return data, label

data, label = loadDataSet("Train.csv")
label = label.reshape((2050,1))
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
nn = BPNetwork()
nn.setup(20,50,50,1)
nn.train_rate_show(x_train, y_train, x_test, y_test, 100, 0.01, 0.1)

