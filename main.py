#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# 線形回帰を行う関数．直線の傾き b と切片 a を返す．
def linear_regression(x, y):
    assert len(x) == len(y)
    
    n = len(x)
    x_mean, y_mean = np.sum(x) / n, np.sum(y) / n
    b = (np.dot(x, y) - n * x_mean * y_mean) / (np.dot(x, x) - n * (x_mean ** 2))
    a = y_mean - b * x_mean

    return b, a

if __name__ == "__main__":
    # 『機械学習図鑑』P.41 参照
    x = np.array([10.0, 8.0, 13.0, 9.0, 11.0, 14.0,
                  6.0, 4.0, 12.0, 7.0, 5.0])

    y = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96,
                  7.24, 4.26, 10.84, 4.82, 5.68])

    # 線形回帰を実行
    b, a = linear_regression(x, y)

    # 回帰直線の方程式
    _y = b * x + a

    # 結果を表示
    plt.figure(figsize=(15, 15))
    plt.scatter(x, y)
    plt.plot(x, _y, color="red")
    plt.show()
