import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import random, numpy as np
from sklearn.metrics import confusion_matrix
import shap


random.seed(20)
np.random.seed(20)
tf.random.set_seed(20)

# Load and preprocess the dataset
df=pd.read_csv("assignment_2/HeartDisease.csv")
df["target"]=df["target"].replace({2:1,3:1,4:1})
X=df.drop("target",axis=1)
y=df["target"]

# manually specify numeric and categorical columns
numeric_cols=["age", "restbps", "chol", "thalach", "oldpeak"]
categorical_cols=["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# encode categorical variables and scale numeric variables
X_keep_first=pd.get_dummies(X, columns=categorical_cols, drop_first=False)
scaler=MinMaxScaler()
X_keep_first[numeric_cols]=scaler.fit_transform(X_keep_first[numeric_cols])

# calculate and display correlations with the target variable
df_corr=pd.concat([X_keep_first, y], axis=1)
corr=df_corr.corr(numeric_only=True)
target_corr=corr[["target"]].drop("target").sort_values(by="target", ascending=False)
print(target_corr)

# visualize the correlations
plt.figure(figsize=(12, 8))
sns.heatmap(target_corr,annot=True,cmap="coolwarm", center=0)
plt.title("Correlation of Inputs with Target")
plt.show()

# prepare data for training, validation, and testing, using drop_first encoding
X_drop_first=pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_drop_first[numeric_cols]=scaler.fit_transform(X_drop_first[numeric_cols])

X_train, X_temp, y_train, y_temp = train_test_split(X_drop_first, y, test_size=0.4, random_state=20)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=20)

def build_model(optimizer_name="adam", learning_rate=0.001, num_hidden_layers=2, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val):

    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))

    for _ in range(num_hidden_layers):
        model.add(layers.Dense(10, activation="relu"))

    model.add(layers.Dense(1, activation="sigmoid"))

    optimizers={
        "adam": keras.optimizers.Adam(learning_rate=learning_rate),
        "adagrad": keras.optimizers.Adagrad(learning_rate=learning_rate),
        "adadelta": keras.optimizers.Adadelta(learning_rate=learning_rate),
        "adamax": keras.optimizers.Adamax(learning_rate=learning_rate),
    }

    if optimizer_name.lower() not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    model.compile(
        optimizer=optimizers[optimizer_name.lower()],
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint=ModelCheckpoint(
        f"best_model_{optimizer_name}_{learning_rate}_{num_hidden_layers}.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=0
    )

    print(f"Training model with optimizer={optimizer_name}, learning rate={learning_rate}, num_hidden_layers={num_hidden_layers}")

    history=model.fit(
        X_train,y_train,
        epochs=100,
        validation_data=(X_val,y_val),
        callbacks=[checkpoint],
        verbose=0
    )

    print("Training complete.")

    best_model = load_model(f"best_model_{optimizer_name}_{learning_rate}_{num_hidden_layers}.keras")

    return best_model


def evaluate_model(model, X, y, dataset):
    loss, accuracy = model.evaluate(X, y, verbose=0)
    print(f"{dataset} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return loss, accuracy

trials=[0.001,0.01,0.1,0.5]

for lr in trials:
    model=build_model(optimizer_name="adam", learning_rate=lr)
    evaluate_model(model, X_train, y_train, "Training")
    evaluate_model(model, X_val, y_val, "Validation")
    evaluate_model(model, X_test, y_test, "Test")
    X_all=pd.concat([X_train, X_val, X_test])
    y_all=pd.concat([y_train, y_val, y_test])
    evaluate_model(model, X_all, y_all, "All Data")
    print("-"*50)

# the best model is the one with learning rate 0.01

optimizers=[ "adadelta", "adagrad", "adamax"]

for opt in optimizers:
    for lr in trials:
        model=build_model(optimizer_name=opt, learning_rate=lr)
        evaluate_model(model, X_train, y_train, "Training")
        evaluate_model(model, X_val, y_val, "Validation")
        evaluate_model(model, X_test, y_test, "Test")
        X_all=pd.concat([X_train, X_val, X_test])
        y_all=pd.concat([y_train, y_val, y_test])
        evaluate_model(model, X_all, y_all, "All Data")
        print("-"*50)

# the best model so far is still the one with optimizer adam and learning rate 0.01

# select features with correlation > 0.4 or < -0.4
selected_features = target_corr[abs(target_corr["target"]) > 0.4].index.tolist()
selected_features = [col for col in selected_features if col in X_drop_first.columns]
print("Selected features:", selected_features)

# prepare reduced dataset
X_reduced=X_drop_first[selected_features]
X_train_reduced, X_temp_reduced, y_train_reduced, y_temp_reduced = train_test_split(X_reduced, y, test_size=0.4, random_state=20)
X_val_reduced, X_test_reduced, y_val_reduced, y_test_reduced = train_test_split(X_temp_reduced, y_temp_reduced, test_size=0.5, random_state=20)

# train and evaluate the best model so far on the reduced dataset
model=build_model(optimizer_name="adam", learning_rate=0.01, X_train=X_train_reduced, y_train=y_train_reduced, X_val=X_val_reduced, y_val=y_val_reduced)
evaluate_model(model, X_train_reduced, y_train_reduced, "Training")
evaluate_model(model, X_val_reduced, y_val_reduced, "Validation")
evaluate_model(model, X_test_reduced, y_test_reduced, "Test")
X_all_reduced=pd.concat([X_train_reduced, X_val_reduced, X_test_reduced])
y_all_reduced=pd.concat([y_train_reduced, y_val_reduced, y_test_reduced])
evaluate_model(model, X_all_reduced, y_all_reduced, "All Data")
print("-"*50)

# experiment with different number of hidden layers
hidden_layers_trials=[2,3,4,5,6]
for hl in hidden_layers_trials:
    model=build_model(optimizer_name="adam", learning_rate=0.01, num_hidden_layers=hl, X_train=X_train_reduced, y_train=y_train_reduced, X_val=X_val_reduced, y_val=y_val_reduced)
    evaluate_model(model, X_train_reduced, y_train_reduced, "Training")
    evaluate_model(model, X_val_reduced, y_val_reduced, "Validation")
    evaluate_model(model, X_test_reduced, y_test_reduced, "Test")
    X_all_reduced=pd.concat([X_train_reduced, X_val_reduced, X_test_reduced])
    y_all_reduced=pd.concat([y_train_reduced, y_val_reduced, y_test_reduced])
    evaluate_model(model, X_all_reduced, y_all_reduced, "All Data")
    print("-"*50)

# the best model is the one with 2 hidden layers, display its confusion matrix on the test set
model=build_model(optimizer_name="adam", learning_rate=0.01, num_hidden_layers=2, X_train=X_train_reduced, y_train=y_train_reduced, X_val=X_val_reduced, y_val=y_val_reduced)
y_pred = (model.predict(X_test_reduced) > 0.5).astype("int")
print(confusion_matrix(y_test_reduced, y_pred))

best_model_all_inputs=build_model(optimizer_name="adam", learning_rate=0.01, num_hidden_layers=2, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
best_model_reduced_inputs=build_model(optimizer_name="adam", learning_rate=0.01, num_hidden_layers=2, X_train=X_train_reduced, y_train=y_train_reduced, X_val=X_val_reduced, y_val=y_val_reduced)

# define a function to run SHAP analysis
def run_shap_analysis(model, X, dataset_name="Test"):
    X = X.astype(float)

    print(f"Running SHAP analysis on {dataset_name} dataset...")
    background = shap.sample(X, 50)

    def pred_fn(data):
        p = np.asarray(model.predict(data, verbose=0))
        return p[:, 1] if (p.ndim == 2 and p.shape[1] > 1) else p.ravel()

    explainer = shap.KernelExplainer(pred_fn, background)
    shap_values = explainer.shap_values(X, nsamples=100)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap.summary_plot(shap_values, X, plot_type="dot", show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)                 
    fig.subplots_adjust(top=0.88)            
    plt.title(f"SHAP Summary (dot) - {dataset_name}", pad=10)  
    plt.tight_layout()                        

    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(top=0.88)
    plt.title(f"SHAP Summary (bar) - {dataset_name}", pad=10)
    plt.tight_layout()
    plt.show()


print("X_test_reduced:",  X_test_reduced.shape,  list(X_test_reduced.columns))
print("X_test:",  X_test.shape,  list(X_test.columns))
run_shap_analysis(best_model_all_inputs, X_test, dataset_name="Test with All Inputs")
run_shap_analysis(best_model_reduced_inputs, X_test_reduced, dataset_name="Test with Reduced Inputs")