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
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import seaborn as sns
from SALib.sample import saltelli
from SALib.analyze import sobol


# load data
df=pd.read_csv("assignment_2/Body_Fat.csv")

# calculate the correlation between BodyFat and input variables
# identify the top correlated variables
corr=df.corr(numeric_only=True)
bodyfat_corr=corr["BodyFat"].drop("BodyFat").sort_values(ascending=False)
selected_features = bodyfat_corr[bodyfat_corr.abs() > 0.4]
print("Selected features:\n", selected_features)

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


def build_model(activation_fn):
    
    tf.random.set_seed(40)

    # build the model
    model = keras.Sequential([
        layers.Input(shape=(14,)),
        layers.Dense(10, activation=activation_fn),
        layers.Dense(10, activation=activation_fn),
        layers.Dense(1, activation="linear")
    ], name=f"model_{activation_fn}")

    # compile the model
    model.compile(
        optimizer=keras.optimizers.SGD(),  
        loss="mse",
    )

    # set up model checkpointing to save the best model during training
    checkpoint = ModelCheckpoint(
        f"best_model_{activation_fn}.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=0
    )

    print(f"Training {activation_fn} model...")

    # start training
    history = model.fit(
        X_train, y_train,
        epochs=200,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint],
        verbose=0
    )
    
    print(f"Training of {activation_fn} model completed.")

    # load the best model and return
    best_model = load_model(f"best_model_{activation_fn}.keras")

    return best_model 


# define a function to evaluate the model
def evaluate_model(model, X, y, dataset_name):

    model_name = model.name
    
    preds = model.predict(X).flatten()
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"{model_name}_on_{dataset_name} -> MSE: {mse:.4f}, R2: {r2:.4f}")

    return mse, r2

X_all = np.vstack([X_train, X_val, X_test])
y_all = np.hstack([y_train, y_val, y_test])

# build, train, and evaluate models with different activation functions
for activation in ["relu", "sigmoid", "tanh", "softmax"]:

    best_model = build_model(activation)

    evaluate_model(best_model, X_train, y_train, "Training")
    evaluate_model(best_model, X_val, y_val, "Validation")
    evaluate_model(best_model, X_test, y_test, "Test")
    evaluate_model(best_model, X_all, y_all, "Whole dataset")
    print("-" * 50)


def build_model_with_different_optimizers(optimizer, activation_fn="sigmoid",trial_name="trial",X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val):

    tf.random.set_seed(40)

    input_dim = X_train.shape[1]  

    # build the model
    model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(10, activation=activation_fn),
            layers.Dense(10, activation=activation_fn),
            layers.Dense(1, activation="linear")
        ], name=f"model_{trial_name}")
    
    # compile the model
    model.compile(
        optimizer=optimizer,  
        loss="mse",
    )
    
    # set up model checkpointing to save the best model during training
    checkpoint = ModelCheckpoint(
        f"best_model_{trial_name}.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=0
    )

    # start training
    print(f"Training {trial_name} model with optimizer {optimizer}...")
    history = model.fit(
        X_train, y_train,
        epochs=200,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint],  
        verbose=0
    )
    print(f"Training of {trial_name} model completed.")

    best_model = load_model(f"best_model_{trial_name}.keras")

    return best_model


# try different optimizers
trials = [

    ("E", keras.optimizers.SGD(learning_rate=0.01, momentum=0.1)),
    ("F", keras.optimizers.SGD(learning_rate=0.1, momentum=0.1)),
    ("G", keras.optimizers.SGD(learning_rate=0.1, momentum=0.5)),
    ("H", keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)),
    ("I", keras.optimizers.SGD(learning_rate=0.5, momentum=0.9)),
    ("J", keras.optimizers.Adagrad(learning_rate=0.1)),
    ("K", keras.optimizers.Adam(learning_rate=0.1)),
    ("L", keras.optimizers.Adadelta(learning_rate=0.1)),

]

# the best performing optimizer is SGD with learning rate 0.1 and momentum 0.5 (trial "G")
for trial_name, optimizer in trials:

    print(f"Starting trial {trial_name} with optimizer {optimizer}...")
    best_model = build_model_with_different_optimizers(optimizer, activation_fn="sigmoid", trial_name=trial_name)

    evaluate_model(best_model, X_train, y_train, "Training")
    evaluate_model(best_model, X_val, y_val, "Validation")
    evaluate_model(best_model, X_test, y_test, "Test")
    evaluate_model(best_model, X_all, y_all, "Whole dataset")
    print("-" * 50)


# plot heatmap of correlation matrix for selected features(correlation coefficients with BodyFat > 0.4 or < -0.4)
corr_bodyfat = corr[["BodyFat"]].loc[selected_features.index]
plt.figure(figsize=(4,6))
sns.heatmap(
    corr_bodyfat,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True
)
plt.title("Correlation with BodyFat (Selected Features)", fontsize=14)
plt.show()


# reduce the dataset to only include selected features and retrain the best model, compare the results
selected_cols = selected_features.index.tolist()
X_reduced = df[selected_cols].values
y = df["BodyFat"].values

X_reduced_scaled = scaler.fit_transform(X_reduced)

X_train_reduced, X_temp_reduced, y_train_reduced, y_temp_reduced = train_test_split(
    X_reduced_scaled, y, test_size=0.4, random_state=40
)
X_val_reduced, X_test_reduced, y_val_reduced, y_test_reduced = train_test_split(
    X_temp_reduced, y_temp_reduced, test_size=0.5, random_state=40
)

best_model_reduced = build_model_with_different_optimizers(optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.5), activation_fn="sigmoid", trial_name="G", X_train=X_train_reduced, y_train=y_train_reduced, X_val=X_val_reduced, y_val=y_val_reduced)
evaluate_model(best_model_reduced, X_train_reduced, y_train_reduced, "Training")
evaluate_model(best_model_reduced, X_val_reduced, y_val_reduced, "Validation")
evaluate_model(best_model_reduced, X_test_reduced, y_test_reduced, "Test")

X_all_reduced = np.vstack([X_train_reduced, X_val_reduced, X_test_reduced])
y_all_reduced = np.hstack([y_train_reduced, y_val_reduced, y_test_reduced])
evaluate_model(best_model_reduced, X_all_reduced, y_all_reduced, "Whole dataset")
print("completed retraining G model on reduced dataset")
print("-" * 50)


def build_model_with_different_layers(num_hidden_layers,trial_name="trial",X_train=X_train_reduced, y_train=y_train_reduced, X_val=X_val_reduced, y_val=y_val_reduced):

    tf.random.set_seed(40)

    input_dim = X_train.shape[1]  

    # build the model

    model = keras.Sequential(name=f"model_{trial_name}")
    model.add(layers.Input(shape=(input_dim,)))

    for _ in range(num_hidden_layers):
        model.add(layers.Dense(10,activation="sigmoid"))
    model.add(layers.Dense(1, activation="linear"))

    # compile the model
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.5),  
        loss="mse",
    )
    
    # set up model checkpointing to save the best model during training
    checkpoint = ModelCheckpoint(
        f"best_model_{trial_name}.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=0
    )

    # start training
    print(f"Training {trial_name} model with {num_hidden_layers} hidden layers...")
    history = model.fit(
        X_train, y_train,
        epochs=200,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint],  
        verbose=0
    )
    print(f"Training of {trial_name} model completed.")

    best_model = load_model(f"best_model_{trial_name}.keras")

    return best_model

# try different number of hidden layers, run and evaluate the models
# the best performing number of layers is 2
for layers_count in [2,3,4,5,6]:

    best_model_layers = build_model_with_different_layers(layers_count, trial_name=f"layers_{layers_count}", X_train=X_train_reduced, y_train=y_train_reduced, X_val=X_val_reduced, y_val=y_val_reduced)
    evaluate_model(best_model_layers, X_train_reduced, y_train_reduced, "Training")
    evaluate_model(best_model_layers, X_val_reduced, y_val_reduced, "Validation")
    evaluate_model(best_model_layers, X_test_reduced, y_test_reduced, "Test")
    X_all_reduced = np.vstack([X_train_reduced, X_val_reduced, X_test_reduced])
    y_all_reduced = np.hstack([y_train_reduced, y_val_reduced, y_test_reduced])
    evaluate_model(best_model_layers, X_all_reduced, y_all_reduced, "Whole dataset")
    print("-" * 50)


# the best model for question vii and xi
best_model_for_vii = build_model_with_different_optimizers(optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.5), activation_fn="sigmoid", trial_name="best_model_in_vii")
best_model_for_xi = build_model_with_different_layers(2, trial_name="best_model_in_xi")

# perform Sobol sensitivity analysis on the best models in (vii) and (xi)
def sobol_sensitivity_analysis(model, X_train, feature_names, N=1024):
    
    # define the problem for Sobol analysis: number of variables, their ranges, names
    problem = {
        "num_vars": X_train.shape[1],
        "names": feature_names,
        "bounds": [[X_train[:, i].min(), X_train[:, i].max()] for i in range(X_train.shape[1])]
    }

    # generate samples using Saltelli's method
    param_values = saltelli.sample(problem, N, calc_second_order=False)

    # use the model to predict outputs for the generated samples above
    Y = model.predict(param_values, verbose=0).flatten()

    # perform Sobol sensitivity analysis
    Si = sobol.analyze(problem, Y, calc_second_order=False)

    # print the results
    print("\nSobol Total-order indices (ST):")
    for name, st in zip(feature_names, Si["ST"]):
        print(f"{name}: {st:.4f}")

    return Si

# perform Sobol analysis on the best model from question (vii) using all features
all_feature_names = X.columns.tolist()
Si_vii = sobol_sensitivity_analysis(
    best_model_for_vii,
    X_train, 
    all_feature_names, 
    N=1024
)

# perform Sobol analysis on the best model from question (xi) using reduced features
Si_xi = sobol_sensitivity_analysis(
    best_model_for_xi,
    X_train_reduced, 
    selected_cols, 
    N=1024
)

# the overall best model is the one from question (xi), which is best_model_for_xi

y_all_pred = best_model_for_xi.predict(X_all_reduced).flatten()
y_test_pred = best_model_for_xi.predict(X_test_reduced).flatten()

mse_all = mean_squared_error(y_all, y_all_pred)
r2_all = r2_score(y_all, y_all_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"All Data (reduced features) -> MSE: {mse_all:.4f}, R²: {r2_all:.4f}")
print(f"Test Data (reduced features) -> MSE: {mse_test:.4f}, R²: {r2_test:.4f}")

# plot True vs Pred and Error distribution
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

axes[0].plot(y_all, label="True BodyFat", marker="o", linestyle="--", alpha=0.7)
axes[0].plot(y_all_pred, label="Predicted BodyFat", marker="x", linestyle="-", alpha=0.7)
axes[0].set_ylabel("BodyFat")
axes[0].set_title(
    f"True vs Predicted BodyFat\n"
    f"All Data -> MSE: {mse_all:.4f}, R²: {r2_all:.4f}\n"
    f"Test Data -> MSE: {mse_test:.4f}, R²: {r2_test:.4f}"
)
axes[0].legend()
axes[0].grid(True)

axes[1].plot(y_all - y_all_pred, label="Error (True - Pred)", linestyle=":", color="red")
axes[1].set_xlabel("Sample Index")
axes[1].set_ylabel("Error")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()