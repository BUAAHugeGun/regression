import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import csv
import os

f_quantile = {41: 4.07855, 40: 4.08475, 39: 4.09128, 38: 4.09817, 37: 4.10546, 36: 4.11317}
number = 43
header = {0: "年份", 1: "财政收入", 2: "第一产业", 3: "工业总产值", 4: "建筑业总产值", 5: "社会商品零售总额", 6: "人口", 7: "受灾面积"}
data = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
data_path = "../data.csv"
f = open(data_path)
f = csv.reader(f)
rows = []
for row in f:
    rows.append(row)
rows = rows[1:]
for i in range(0, 8):
    for row in rows:
        data[i].append(float(row[i]))
id = {}
for i in range(2, 8):
    id[data[i][0]] = i
x = [np.array(data[i]) for i in range(2, 8)]
y = np.array(data[1])


def sqr_sum(ar):
    return np.sum(ar * ar)


def PF(X, I):
    x = sm.add_constant(np.column_stack(X))
    model = sm.OLS(y, x)
    res = model.fit()
    yur = res.fittedvalues

    x = []
    p = len(X)
    for i in range(p):
        if i != I:
            x.append(X[i])
    if len(x) == 0:
        x = [[0 for i in range(number)]]
    x = sm.add_constant(np.column_stack(x))
    model = sm.OLS(y, x)
    res = model.fit()
    yr = res.fittedvalues

    y_avg = np.average(y)
    ESS = sqr_sum(yur - y_avg) - sqr_sum(yr - y_avg)
    RSS = sqr_sum(yur - y)
    return ESS / (RSS / (number - p - 1))


def cc():
    select = []
    remain = x
    while True:
        useful = False
        pFs = []
        for tx in remain:
            pFs.append((PF(select + [tx], len(select)), tx))
        pFs.sort()
        best = pFs[-1]
        if best[0] > f_quantile[number - len(select) - 2]:
            select.append(best[1])
            useful = True
            print('add')

        pFs.clear()
        for i in range(len(select)):
            tx = select[i]
            pFs.append((PF(select, i), tx))
        pFs.sort()
        best = pFs[0]
        if best[0] <= f_quantile[number - len(select) - 2]:
            select.remove(best[1])
            useful = True
            print("erase")
        if not useful:
            break
    return select


def dfs(k, select):
    if k == len(x):
        if len(select) == 0:
            return [], 0.
        X = sm.add_constant(np.column_stack(select))
        model = sm.OLS(y, X)
        res = model.fit()
        return select, res.fvalue
    p1 = dfs(k + 1, select)
    p2 = dfs(k + 1, select + [x[k]])
    return p1 if p1[1] > p2[1] else p2


if __name__ == "__main__":
    X = sm.add_constant(np.column_stack(x))
    model = sm.OLS(y, X)  # 最小二乘法
    res = model.fit()  # 拟合数据
    Bata = res.params  # 取系数
    print(res.summary())  # 结果
    print(res.f_pvalue)
    # print(X, y)

    Y = res.fittedvalues  # 预测值
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.linspace(1978, 2019, number), y, 'o', label='data')  # 原始数据
    ax.plot(np.linspace(1978, 2019, number), Y, 'r--.', label='test')  # 拟合数据
    ax.legend(loc='best')  # 展示各点表示意思，即label
    plt.show()

    x = cc()
    for xx in x:
        print(header[id[xx[0]]])
    X = sm.add_constant(np.column_stack(x))
    model = sm.OLS(y, X)  # 最小二乘法
    res = model.fit()  # 拟合数据
    print(res.summary())  # 结果

    Y = res.fittedvalues  # 预测值
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.linspace(1978, 2019, number), y, 'o', label='data')  # 原始数据
    ax.plot(np.linspace(1978, 2019, number), Y, 'r--.', label='test')  # 拟合数据
    ax.legend(loc='best')  # 展示各点表示意思，即label
    plt.show()

    x = dfs(0, [])[0]
    for xx in x:
        print(header[id[xx[0]]])
    X = sm.add_constant(np.column_stack(x))
    model = sm.OLS(y, X)  # 最小二乘法
    res = model.fit()  # 拟合数据
    print(res.summary())  # 结果

    Y = res.fittedvalues  # 预测值
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.linspace(1978, 2019, number), y, 'o', label='data')  # 原始数据
    ax.plot(np.linspace(1978, 2019, number), Y, 'r--.', label='test')  # 拟合数据
    ax.legend(loc='best')  # 展示各点表示意思，即label
    plt.show()
