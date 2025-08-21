import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ===== 配置 =====
CSV_PATH   = "heat_influx_north_south.csv"        # 你的数据文件
FEATURE_COLS = ["South", "North"]   # 两个输入
TARGET_COL   = "HeatFlux"               # 热流量（连续值）
EPOCHS = 200
LR = 0.5
SEED = 42

# ===== 读取与归一化 =====
df = pd.read_csv(CSV_PATH)
X_raw = df[FEATURE_COLS].to_numpy(dtype=float)            # (n,2)
y_raw = df[TARGET_COL].to_numpy(dtype=float).reshape(-1,1)# (n,1)

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()          # 目标缩放到 [0,1] 以匹配 Sigmoid
X = x_scaler.fit_transform(X_raw)  # (n,2)
y = y_scaler.fit_transform(y_raw).ravel()  # (n,)

# ===== 单神经元 + Sigmoid =====
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# ===== 把一次训练与评估封装成函数：保证每次都用相同初始权重 =====
def train_and_eval(num_epochs:int, lr:float, seed:int):
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.1, size=X.shape[1])  # 重新初始化，保证可比
    b = 0.0

    idx = np.arange(len(y))
    for _ in range(num_epochs):
        rng.shuffle(idx)
        for i in idx:
            xi, yi = X[i], y[i]
            z = np.dot(w, xi) + b
            yhat = sigmoid(z)
            g = (yhat - yi) * yhat * (1.0 - yhat)  # dMSE/dz
            w -= lr * g * xi
            b -= lr * g

    # 评估（原始量纲）
    yhat_scaled = sigmoid(X @ w + b)
    y_true = y_scaler.inverse_transform(y.reshape(-1,1)).ravel()
    y_pred = y_scaler.inverse_transform(yhat_scaled.reshape(-1,1)).ravel()
    mse_orig = mean_squared_error(y_true, y_pred)
    r2_orig  = r2_score(y_true, y_pred)
    return w, b, mse_orig, r2_orig

# ===== 跑多组 epoch，并打印表格 =====
epochs_list = [200, 400, 600, 800, 1000]
results = []
for ep in epochs_list:
    w_ep, b_ep, mse_ep, r2_ep = train_and_eval(ep, LR, SEED)
    # results.append((ep, mse_ep, r2_ep))
    results.append((ep, mse_ep, r2_ep, w_ep.copy(), b_ep))
    print(f"epochs={ep} -> MSE(orig)={mse_ep:.6f}, R2(orig)={r2_ep:.6f}")

# 可选：挑选最优模型（以 MSE 最小为准）
best_ep, best_mse, best_r2, best_w, best_b = min(results, key=lambda t: t[1])
print("\nBest by MSE:")
print(f"epochs={best_ep}, MSE(orig)={best_mse:.6f}, R2(orig)={best_r2:.6f}")

# print("Best weights (normalized inputs) [South_norm, North_norm]:", best_w)
# print("Best bias (normalized inputs):", best_b)
# print(
#     "Network (normalized): y_hat = sigmoid("
#     f"{best_w[0]:.6f} * South_norm + {best_w[1]:.6f} * North_norm + {best_b:.6f})"
# )

w_orig = best_w * x_scaler.scale_
b_orig = best_b + np.dot(best_w, x_scaler.min_)
sy = y_scaler.scale_[0]
my = y_scaler.min_[0]
w_final = w_orig / sy
b_final = (b_orig / sy) - (my / sy)
print("Weights in original input units [South, North]:", w_orig)
print("Bias in original input units:", b_orig)
print(
    "Clean formula in ORIGINAL units:\n"
    f"y_hat = sigmoid({w_final[0]:.6f} * South + {w_final[1]:.6f} * North + {b_final:.6f})"
)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 原始输入
south = df["South"].to_numpy()
north = df["North"].to_numpy()
heatflux = df["HeatFlux"].to_numpy()

# 预测平面 (网格)
south_lin = np.linspace(south.min(), south.max(), 50)
north_lin = np.linspace(north.min(), north.max(), 50)
S, N = np.meshgrid(south_lin, north_lin)
Z_sig = sigmoid(w_orig[0] * S + w_orig[1] * N + b_orig)

# 把预测值拉回原始HeatFlux单位
Z = y_scaler.inverse_transform(Z_sig.reshape(-1,1)).reshape(S.shape)

# --- 画图 ---
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")

# 真实点
ax.scatter(south, north, heatflux, c="blue", label="True data") # type: ignore

# 预测平面
ax.plot_surface(S, N, Z, color="red", alpha=0.5, label="Network function")

ax.set_xlabel("South")
ax.set_ylabel("North")
ax.set_zlabel("HeatFlux")
plt.title("3D plot: True data + Network function")
plt.show()

# ===== 3D: 预测点 vs 真实点 =====
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 逐样本预测（原始单位）
z_pts = w_orig[0] * south + w_orig[1] * north + b_orig
yhat_sig_pts = sigmoid(z_pts)
yhat_pts = y_scaler.inverse_transform(yhat_sig_pts.reshape(-1,1)).ravel()

fig2 = plt.figure(figsize=(8,6))
ax2 = fig2.add_subplot(111, projection="3d")
ax2.scatter(south, north, heatflux, s=40, label="Target (true)",  c="blue") # type: ignore
ax2.scatter(south, north, yhat_pts, s=40, label="Predicted",      c="red", alpha=0.8) # type: ignore

ax2.set_xlabel("South")
ax2.set_ylabel("North")
ax2.set_zlabel("HeatFlux")
ax2.set_title("3D plot: Predicted vs Target")
ax2.legend(loc="best")
plt.show()