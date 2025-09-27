import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# load the dataset and extract the south and north columns as the feature matrix X, and the heatflux column as the target vector y
df = pd.read_csv("heat_influx_north_south.csv")  
X = df[["South", "North"]].to_numpy(dtype=float)
y = df["HeatFlux"].to_numpy(dtype=float).reshape(-1, 1)

# print the basic information about the dataset
print("Total num of rows:", len(df))
print("number of unique rows after removing duplicates:", len(df.drop_duplicates()))
print("number of fully duplicated rows:", df.duplicated().sum())

epochs_list = [100, 150, 200, 250, 300]
results = []

# Train the model with different epochs, evaluate MSE and R², save the results, and print them one by one
for ep in epochs_list:
    model = Pipeline([
        ("x_scaler", MinMaxScaler()),
        ("reg", TransformedTargetRegressor(
            regressor=SGDRegressor(
                max_iter=ep, tol=None,            
                learning_rate="constant", eta0=0.01,
                penalty=None, shuffle=True, random_state=42
            ),
            transformer=MinMaxScaler()           
        ))
    ])

    model.fit(X, y)
    y_pred = model.predict(X)                     
    mse = mean_squared_error(y, y_pred)
    r2  = r2_score(y, y_pred)
    results.append((ep, mse, r2, model))  

print("Epochs |     MSE |   R2    | Regressor")
for ep, mse, r2, model in results:
    reg = model.named_steps["reg"].regressor_
    print(f"{ep:6d} | {mse:7.4f} | {r2:.4f}  | max_iter={reg.max_iter}, lr={reg.learning_rate}")


# find the best model based on MSE, extract the weights and bias, and write the final equation
# equation y^​=0.429⋅South−0.875⋅North+0.808
best_ep, best_mse, best_r2, best_model = min(results, key=lambda t: t[1])
print(f"Best by MSE: epochs={best_ep}, MSE={best_mse:.4f}, R2={best_r2:.4f}")
reg = best_model.named_steps["reg"].regressor_
print(reg.coef_, reg.intercept_)


# Part1，plot the original data point together with the fitted plane
fig = plt.figure(figsize=(8,6))

# create a 3D scatter plot of the original data points
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:,0], X[:,1], y.ravel(), c="blue", marker="o", label="Data")  # type: ignore

south_range = np.linspace(X[:,0].min(), X[:,0].max(), 30)
north_range = np.linspace(X[:,1].min(), X[:,1].max(), 30)
south_grid, north_grid = np.meshgrid(south_range, north_range)

# use the best model to predict the heatflux values on the grid
grid_points = np.c_[south_grid.ravel(), north_grid.ravel()]
y_pred_grid = best_model.predict(grid_points).reshape(south_grid.shape)

ax.plot_surface(south_grid, north_grid, y_pred_grid,
                color="orange", alpha=0.5, label="Model")

ax.set_box_aspect((1, 1, 0.7))                       # 0.25~0.5 之间都可，0.35更贴近人眼
zmin, zmax = float(np.min(y)), float(np.max(y))
ax.set_zlim(zmin, zmax)                                # 先固定范围
ax.invert_zaxis()                                      # 强制把 低→高 设为向上（有些后端默认反着）
ax.view_init(elev=22, azim=-55)  

ax.set_xlabel("South")
ax.set_ylabel("North")
ax.set_zlabel("HeatFlux")
plt.title("Data vs. Linear Neuron Prediction")


# Part2，compare the target values and model predictions in 3D
y_pred = best_model.predict(X)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], y.ravel(), c='b', label='Target', alpha=0.6) # type: ignore
ax.scatter(X[:,0], X[:,1], y_pred.ravel(), c='r', marker='^', label='Predicted', alpha=0.6)

ax.set_xlabel("South")
ax.set_ylabel("North")
ax.set_zlabel("HeatFlux")
ax.legend()
plt.title("Target vs Predicted HeatFlux")
plt.show()