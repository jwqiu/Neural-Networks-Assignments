import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# configuration
CSV_PATH = "Heat_Influx_insulation_east_south_north.csv"  # 放到脚本同目录即可
FEATURE_COLS = ["Insulation", "East", "South", "North"]   # 四个输入
TARGET_COL   = "HeatFlux"                                 # 输出

# load the dataset and extract the features and target
df = pd.read_csv(CSV_PATH)
X_raw = df[FEATURE_COLS].to_numpy(dtype=float)       # (n,4)
y_raw = df[[TARGET_COL]].to_numpy(dtype=float)       # (n,1) 保持二维


import matplotlib.pyplot as plt

# plot the original data pointt with scatter plots and  observe the relationship between the target and each feature
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
for ax, col in zip(axes.ravel(), ["Insulation", "East", "South", "North"]):
    ax.scatter(df[col], df[TARGET_COL])
    ax.set_xlabel(col)
    ax.set_ylabel("HeatFlux")
    ax.set_title(f"HeatFlux vs {col}")
fig.suptitle("Target vs Inputs")
fig.tight_layout()
plt.show()

# main observations:
# Insulation has a positive relationship with HeatFlux
# North has the strongest negative relationship with HeatFlux
# South and East have weaker relationships with HeatFlux, however, South shows a slight positive relationship with HeatFlux only below 36.

# split the dataset into training, validation, and test sets
idx = np.arange(len(df))
train_idx, tmp_idx = train_test_split(idx, test_size=0.4, random_state=42, shuffle=True)
val_idx,   test_idx = train_test_split(tmp_idx, test_size=0.5, random_state=42, shuffle=True)

X_train_raw, y_train_raw = X_raw[train_idx], y_raw[train_idx]
X_val_raw,   y_val_raw   = X_raw[val_idx],   y_raw[val_idx]
X_test_raw,  y_test_raw  = X_raw[test_idx],  y_raw[test_idx]

x_scaler = MinMaxScaler().fit(X_train_raw)
y_scaler = MinMaxScaler().fit(y_train_raw)

X_train = x_scaler.transform(X_train_raw)
X_val   = x_scaler.transform(X_val_raw)
X_test  = x_scaler.transform(X_test_raw)

y_train = y_scaler.transform(y_train_raw).ravel()
y_val   = y_scaler.transform(y_val_raw).ravel()
y_test  = y_scaler.transform(y_test_raw).ravel()

from sklearn.neural_network import MLPRegressor
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score

# create a function to build a multi-layer model where you can set the number of hidden neurons and activation function
def make_mlp(H:int, activation:str) -> MLPRegressor:
    """创建 4–H–1 的三层网络；隐藏层激活由 activation 指定（'logistic' 或 'relu'）"""
    return MLPRegressor(
        hidden_layer_sizes=(H,),
        activation=activation,        # 'logistic' (sigmoid) 或 'relu' # type: ignore
        solver="sgd",
        learning_rate="constant",
        learning_rate_init=0.5,       # lr=0.5
        batch_size=1,                 # 纯 SGD
        max_iter=1000,                # 1000 epochs
        early_stopping=False,
        n_iter_no_change=100000,
        tol=0.0,
        shuffle=True,
        random_state=42               # 固定种子，便于可比
    )

# evaluate a trained model on a given split and compute MSE and R2 in the original HeatFlux units
def eval_split(model, Xs, y_raw_split, name):
    """在一个数据切分上评估 MSE/R2（输出为原始 HeatFlux 单位）"""
    yhat_s = model.predict(Xs).reshape(-1, 1)             # 预测在[0,1]标度
    y_pred = y_scaler.inverse_transform(yhat_s).ravel()   # 反归一化到原始单位
    y_true = y_raw_split.ravel()
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"{name:>12} -> MSE={mse:.4f}, R2={r2:.4f}")
    return mse, r2

# train mlp models with the specified activation function and different hidden layer sizes
# evaluate each model on train/val/test/all sets and print the results
def run_sweep(activation:str, hidden_list:list[int]):
    """
    传入激活函数与隐藏神经元列表，依次训练并在 train/val/test/ALL 上评估。
    返回 (best_model, rows)，其中 rows 便于生成表格/保存。
    选优准则：优先 test RMSE 最小，然后 test R2 最大，平手再看 ALL（RMSE 小、R2 大）。
    """
    print(f"\nSweep activation={activation}, H = {hidden_list} (SGD lr=0.5, epochs=1000)")
    rows = []  # (H, r2_tr, r2_va, r2_te, r2_all, rmse_tr, rmse_va, rmse_te, rmse_all, model)

    for H in hidden_list:
        model = make_mlp(H, activation).fit(X_train, y_train)

        mse_tr, r2_tr = eval_split(model, X_train, y_train_raw, f"train(H={H})")
        mse_va, r2_va = eval_split(model, X_val,   y_val_raw,   f"val(H={H})")
        mse_te, r2_te = eval_split(model, X_test,  y_test_raw,  f"test(H={H})")

        X_all = x_scaler.transform(X_raw)
        mse_all, r2_all = eval_split(model, X_all, y_raw,       f"ALL(H={H})")

        rows.append((
            H, r2_tr, r2_va, r2_te, r2_all,
            sqrt(mse_tr), sqrt(mse_va), sqrt(mse_te), sqrt(mse_all),
            model
        ))

    # 打印汇总表
    print("\nSummary (original HeatFlux units)")
    print(" H |  R2_train  R2_val  R2_test  R2_all |  RMSE_train  RMSE_val  RMSE_test  RMSE_all")
    for H, r2_tr, r2_va, r2_te, r2_all, rmse_tr, rmse_va, rmse_te, rmse_all, _ in rows:
        print(f"{H:>2} |   {r2_tr:7.3f}  {r2_va:6.3f}  {r2_te:7.3f}  {r2_all:6.3f} |"
              f"     {rmse_tr:8.3f}  {rmse_va:8.3f}  {rmse_te:9.3f}  {rmse_all:9.3f}")

    # 选最佳（test 优先）：(test RMSE 最小, -test R2 最大, ALL RMSE 最小, -ALL R2 最大)
    best = min(rows, key=lambda t: (t[7], -t[3], t[8], -t[4]))
    H_best, best_model = best[0], best[-1]
    print(f"\nSelected best by TEST (RMSE then R2): H={H_best}, activation={activation}")
    return best_model, rows

# get the predicted values in the original HeatFlux units 
def predict_original_units(model, X_raw_new: np.ndarray) -> np.ndarray:
    """给任意原始单位的新样本做预测（返回原始单位 HeatFlux）"""
    Xs = x_scaler.transform(X_raw_new)
    yhat_s = model.predict(Xs).reshape(-1, 1)
    yhat   = y_scaler.inverse_transform(yhat_s).ravel()
    return yhat

# run model sweeps for logistic and ReLU activations with different hidden layer sizes, select the best models, and make predictions
best_logistic, rows_logistic = run_sweep("logistic", [1, 2, 3, 4, 5])
best_relu,     rows_relu     = run_sweep("relu",     [1, 5, 10, 20])

y_true_test   = y_test_raw.ravel()
y_pred_log    = predict_original_units(best_logistic, X_test_raw)
y_pred_relu   = predict_original_units(best_relu,     X_test_raw)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

# left plot:target vs predicted heatflux on the test set using the best logistic model 
axes[0].plot(y_true_test, label="Target",    marker="o")
axes[0].plot(y_pred_log,  label="Predicted", marker="x")
axes[0].set_title("Best Logistic (TEST)")
axes[0].set_xlabel("Test sample index")
axes[0].set_ylabel("HeatFlux")
axes[0].legend()

# right plot:target vs predicted heatflux on the test set using the best ReLU model
axes[1].plot(y_true_test, label="Target",    marker="o")
axes[1].plot(y_pred_relu, label="Predicted", marker="x")
axes[1].set_title("Best ReLU (TEST)")
axes[1].set_xlabel("Test sample index")
axes[1].legend()

fig.suptitle("Target vs Predicted on TEST")
fig.tight_layout()
plt.show()