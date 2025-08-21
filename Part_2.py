import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 1) 读数据（列名：HeatFlux, South, North）
df = pd.read_csv("heat_influx_north_south.csv")  # 列名: HeatFlux, South, North
X = df[["South", "North"]].to_numpy(dtype=float)
y = df["HeatFlux"].to_numpy(dtype=float).reshape(-1, 1)

print("总行数:", len(df))
print("去重后行数:", len(df.drop_duplicates()))
print("完全重复行数:", df.duplicated().sum())

epochs_list = [100, 150, 200, 250, 300]
results = []

# 2) 流水线：X 与 y 都做 0–1 归一化；回归器用 SGD，100 epochs
for ep in epochs_list:
    model = Pipeline([
        ("x_scaler", MinMaxScaler()),
        ("reg", TransformedTargetRegressor(
            regressor=SGDRegressor(
                max_iter=ep, tol=None,            # 跑满100个epoch
                learning_rate="constant", eta0=0.01,
                penalty=None, shuffle=True, random_state=42
            ),
            transformer=MinMaxScaler()            # y 也归一化到[0,1]
        ))
    ])

    # 3) 训练与评估（在全数据上）
    model.fit(X, y)
    y_pred = model.predict(X)                     # 已自动反归一化回原单位
    mse = mean_squared_error(y, y_pred)
    r2  = r2_score(y, y_pred)
    results.append((ep, mse, r2, model))  # 多存一个已训练好的 model

print("Epochs |     MSE |   R2    | Regressor")
for ep, mse, r2, model in results:
    reg = model.named_steps["reg"].regressor_
    print(f"{ep:6d} | {mse:7.4f} | {r2:.4f}  | max_iter={reg.max_iter}, lr={reg.learning_rate}")

# 4) 找出最佳的 epochs
best_ep, best_mse, best_r2, best_model = min(results, key=lambda t: t[1])
print(f"Best by MSE: epochs={best_ep}, MSE={best_mse:.4f}, R2={best_r2:.4f}")

reg = best_model.named_steps["reg"].regressor_
print(reg.coef_, reg.intercept_)

# equation y^​=0.429⋅South−0.875⋅North+0.808

# ---- 3D plot: data points + fitted plane ----
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")

# —— 让三个轴按数据跨度等比例显示（避免Z轴把图“压扁/拉长”）
# xrange = X[:,0].max() - X[:,0].min()      # South
# yrange = X[:,1].max() - X[:,1].min()      # North
# zrange = y.max() - y.min()                # HeatFlux（若 y 是 (n,1)，也可用 y.ravel().max() - y.ravel().min()
# ax.set_box_aspect((xrange, yrange, zrange))
# ax.set_box_aspect((1, 1, 2))


# 1) 画真实数据点
ax.scatter(X[:,0], X[:,1], y.ravel(), c="blue", marker="o", label="Data")  # type: ignore


# 2) 生成 South 和 North 的网格
south_range = np.linspace(X[:,0].min(), X[:,0].max(), 30)
north_range = np.linspace(X[:,1].min(), X[:,1].max(), 30)
south_grid, north_grid = np.meshgrid(south_range, north_range)

# 3) 用模型预测对应的 HeatFlux
grid_points = np.c_[south_grid.ravel(), north_grid.ravel()]
y_pred_grid = best_model.predict(grid_points).reshape(south_grid.shape)

# 4) 画预测平面
ax.plot_surface(south_grid, north_grid, y_pred_grid,
                color="orange", alpha=0.5, label="Model")

# ax.set_zlim(np.asarray(y).min(), np.asarray(y).max())  # 确保上下界正确
# if ax.get_zticks()[0] > ax.get_zticks()[-1]: # type: ignore
#     ax.invert_zaxis()  # 若刻度是从大到小，翻转回来

ax.set_box_aspect((1, 1, 0.7))                       # 0.25~0.5 之间都可，0.35更贴近人眼
zmin, zmax = float(np.min(y)), float(np.max(y))
ax.set_zlim(zmin, zmax)                                # 先固定范围
ax.invert_zaxis()                                      # 强制把 低→高 设为向上（有些后端默认反着）
ax.view_init(elev=22, azim=-55)  


# 5) 标签 & 图例
ax.set_xlabel("South")
ax.set_ylabel("North")
ax.set_zlabel("HeatFlux")
plt.title("Data vs. Linear Neuron Prediction")

# ---- 真实值 vs 预测值 (散点对比) ----
y_pred = best_model.predict(X)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# 真实值 (蓝色圆点)
ax.scatter(X[:,0], X[:,1], y.ravel(), c='b', label='Target', alpha=0.6) # type: ignore

# 预测值 (红色三角)
ax.scatter(X[:,0], X[:,1], y_pred.ravel(), c='r', marker='^', label='Predicted', alpha=0.6)

ax.set_xlabel("South")
ax.set_ylabel("North")
ax.set_zlabel("HeatFlux")
ax.legend()
plt.title("Target vs Predicted HeatFlux")
plt.show()