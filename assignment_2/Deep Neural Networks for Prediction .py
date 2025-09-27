import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# load data
df=pd.read_csv("assignment_2/Body_Fat.csv")

# calculate the correlation between BodyFat and input variables
# identify the top correlated variables
corr=df.corr(numeric_only=True)
bodyfat_corr=corr["BodyFat"].drop("BodyFat").sort_values(ascending=False)
print(bodyfat_corr)

# visualize how the top correlated variables relate to BodyFat

# Use scatter plot to visualize the relationship between Abdomen and BodyFat
plt.scatter(df["Abdomen"], df["BodyFat"])
plt.xlabel("Abdomen")
plt.ylabel("BodyFat")
plt.title("Abdomen vs BodyFat")
plt.show()

# Use bar plot to visualize the relationship between Weight and BodyFat
sns.barplot(x=pd.qcut(df["Weight"], 5), y=df["BodyFat"])  # Divide Weight into 5 groups
plt.xticks(rotation=30)
plt.title("Average BodyFat by Weight Group")
plt.show()

# normalize the data and split into training, validation, and test sets
X=df.drop("BodyFat", axis=1)
y=df["BodyFat"]

scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X)

X_train, X_temp, y_train, y_temp=train_test_split(X_scaled, y, test_size=0.4, random_state=40)
X_val, X_test, y_val, y_test=train_test_split(X_temp, y_temp, test_size=0.5, random_state=40)


# 1. 构建模型
model = keras.Sequential([
    layers.Input(shape=(14,)),                  # 输入层：14个特征
    layers.Dense(10, activation="relu"),        # 隐藏层1：10个神经元，ReLU
    layers.Dense(10, activation="relu"),        # 隐藏层2：10个神经元，ReLU
    layers.Dense(1, activation="linear")        # 输出层：线性激活
])

# 2. 编译模型 (SGD + MSE + R² 作为指标)
model.compile(
    optimizer=keras.optimizers.SGD(),   # 默认学习率和动量
    loss="mse",
    metrics=[keras.metrics.RootMeanSquaredError()]  # 可以先用 RMSE 作为监控
)

# 3. 训练模型
history = model.fit(
    X_train, y_train,
    epochs=200,
    validation_data=(X_val, y_val),
    verbose=1
)

# 4. 评估函数
def evaluate_model(model, X, y, dataset_name):
    preds = model.predict(X).flatten()
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"{dataset_name} -> MSE: {mse:.4f}, R2: {r2:.4f}")
    return mse, r2

# 5. 分别计算训练/验证/测试/整体集的表现
evaluate_model(model, X_train, y_train, "Training")
evaluate_model(model, X_val, y_val, "Validation")
evaluate_model(model, X_test, y_test, "Test")

# 整个数据集
X_all = np.vstack([X_train, X_val, X_test])
y_all = np.hstack([y_train, y_val, y_test])
evaluate_model(model, X_all, y_all, "Whole dataset")