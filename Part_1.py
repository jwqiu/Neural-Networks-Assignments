import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

# Load data from a csv file(given by path)
# Use the first two columns as the feature matrix X and the third column as the label vector y
# Return both X and y in NumPy array format
def load_data(path:str):
    df=pd.read_csv(path)
    X_df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce")
    y_df = df.iloc[:, 2].apply(pd.to_numeric, errors="coerce")
    X = X_df.to_numpy()
    y = y_df.to_numpy()
    return X, y


def main():

    # Load the dataset from the specified path and split it into features(X) and labels(y)
    # Build a pipeline that first standardizes the feautres and then trains a Perceptron model
    # Classifier is trained with 100 epochs and no early stopping
    # Fit the pipeline on the data, evaluate its accuracy on the training set
    # print the training accuracy score
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

    # create a new figure, plot the data points as a scatter plot
    # class 0 samples(canadian) are blue, class 1 samples(alaskan) are red
    # set axis labels for the two features
    plt.figure(figsize=(6,5))
    plt.scatter(X[y==0, 0], X[y==0, 1], c="blue", label="Canadian_0")
    plt.scatter(X[y==1, 0], X[y==1, 1], c="red", label="Alaskan_1")
    plt.xlabel("RingDiam_fresh_water")
    plt.ylabel("RingDiam_salt_water")


    # compute the decision boundary in the original feature space:
    # retrieve weights(w,b) learned in the standardized space
    # transform them back to the original scale(undo standardization)
    # generate x-values, solve for corresponding y-values, and plot the line
    # print the boundary equation in slope-intercept form((y = m*x + q))
    scaler = clf.named_steps["standardscaler"]
    perce  = clf.named_steps["perceptron"]
    w = perce.coef_[0]
    b = perce.intercept_[0]
    A = w / scaler.scale_  
    c = b - np.sum(w * scaler.mean_ / scaler.scale_)
    x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
    y_vals = -(A[0] * x_vals + c) / A[1]
    plt.plot(x_vals, y_vals, "k--", label="Decision boundary")
    m = -A[0] / A[1]
    q = -c / A[1]
    print(f"Decision boundary in original space: y = {m:.4f} * x + {q:.4f}")


    plt.legend()
    # plt.title("Perceptron Classification Boundary")
    plt.show()


if __name__ == "__main__":
    main()
