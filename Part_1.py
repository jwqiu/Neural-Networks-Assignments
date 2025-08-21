import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

def load_data(path:str):
    df=pd.read_csv(path)
    X_df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce")
    y_df = df.iloc[:, 2].apply(pd.to_numeric, errors="coerce")
    X = X_df.to_numpy()
    y = y_df.to_numpy()
    return X, y


def main():
    path="Fish_data.csv"
    X, y = load_data(path)
    clf = make_pipeline(
        StandardScaler(),
        Perceptron(max_iter=100, tol=None, random_state=42)  # 100 epochs
    )
    clf.fit(X, y)
    acc = clf.score(X, y)

    print("Model trained successfully.")
    print(f"Model accuracy: {acc:.2f}")

    plt.figure(figsize=(6,5))

    # 画散点：按类别显示不同颜色
    plt.scatter(X[y==0, 0], X[y==0, 1], c="blue", label="Canadian_0")
    plt.scatter(X[y==1, 0], X[y==1, 1], c="red", label="Alaskan_1")

    # 取出 StandardScaler 和 Perceptron
    scaler = clf.named_steps["standardscaler"]
    perce  = clf.named_steps["perceptron"]

    # 感知机学到的 w 和 b（在标准化后的空间）
    w = perce.coef_[0]
    b = perce.intercept_[0]

    # === 关键一步：把 w,b 转换回原始空间 ===
    A = w / scaler.scale_  
    # 这里除以 scale_ 就是把“标准化后的斜率”换算回原始尺度

    c = b - np.sum(w * scaler.mean_ / scaler.scale_)
    # 这里减去的是“均值的偏移”，让直线回到原始坐标系

    # 在原始坐标上画线
    x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)

    y_vals = -(A[0] * x_vals + c) / A[1]

    plt.plot(x_vals, y_vals, "k--", label="Decision boundary")
    
    m = -A[0] / A[1]
    q = -c / A[1]
    print(f"Decision boundary in original space: y = {m:.4f} * x + {q:.4f}")

    plt.xlabel("RingDiam_fresh_water")
    plt.ylabel("RingDiam_salt_water")
    plt.legend()
    # plt.title("Perceptron Classification Boundary")
    plt.show()


if __name__ == "__main__":
    main()
