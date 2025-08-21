import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def load_data(path:str):
    df=pd.read_csv(path)
    X_df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce")
    y_df = df.iloc[:, 2].apply(pd.to_numeric, errors="coerce")
    X = X_df.to_numpy()
    y = y_df.to_numpy()
    return X, y


# 1) 加载并归一化
X, y = load_data("Fish_data.csv")
X = MinMaxScaler().fit_transform(X)

# 2) 多组 epoch 迭代次数
EPOCH_LIST = [100, 200, 300, 500, 1000]

results = []
models = {}

# 3) 循环训练并评估
for epochs in EPOCH_LIST:
    print(f"\n===== Training with {epochs} epochs =====")
    clf = LogisticRegression(solver="saga", max_iter=epochs, random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    results.append((epochs, acc))
    models[epochs] = clf  # 保存模型
    print(f"\n===== {epochs} epochs =====")
    print(classification_report(y, y_pred))


# 4) 找出准确率最高的模型
best_epochs, best_acc = max(results, key=lambda x: x[1])
best_model = models[best_epochs]

print(f"\nBest model: {best_epochs} epochs, accuracy = {best_acc:.4f}")

# 5) 提取权重和偏置
weights = best_model.coef_[0]
bias = best_model.intercept_[0]

print("Weights:", weights)
print("Bias:", bias)

w1 = 3.5431793
w2 = -2.86608717
b = -0.4326884556738937

# 画出散点图
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Paired", edgecolors="k")

# 决策边界：解 w1*x1 + w2*x2 + b = 0
x1_vals = np.linspace(min(X[:,0]), max(X[:,0]), 100)
x2_vals = -(w1*x1_vals + b) / w2   # x2 对应的直线

plt.plot(x1_vals, x2_vals, "r-", label="Decision Boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()


