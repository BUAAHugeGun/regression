import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import csv
import os

if __name__ == "__main__":
    nSample = 20
    x1 = np.linspace(0, 10, nSample)  # 起点为 0，终点为 10，均分为 nSample个点
    e = np.random.normal(size=len(x1))  # 正态分布随机数
    yTrue = 2.36 + 1.58 * x1  # y = b0 + b1*x1
    yTest = yTrue + e  # 产生模型数据

    X = sm.add_constant(x1)
    model = sm.OLS(yTest, X)  # 最小二乘法
    res = model.fit()  # 拟合数据
    Bata = res.params  # 取系数
    print(res.summary())  # 结果
    print(res.f_pvalue)
    # print(X, y)

    Y = res.fittedvalues  # 预测值
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x1, yTest, 'o', label='data')  # 原始数据
    ax.plot(x1, Y, 'r--.', label='test')  # 拟合数据
    ax.legend(loc='best')  # 展示各点表示意思，即label
    plt.show()
