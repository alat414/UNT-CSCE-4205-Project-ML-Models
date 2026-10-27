import pandas as pd

X_train = pd.read_csv("X_test.txt", sep=r"\s+", header=None)

print(X_train.head())