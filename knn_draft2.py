import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from pathlib import Path

current_folder = Path(__file__).resolve().parent

X_train = np.loadtxt(current_folder / 'X_train.txt')
y_train = np.loadtxt(current_folder / 'y_train.txt')

X_test = np.loadtxt(current_folder / 'X_test.txt')
y_test = np.loadtxt(current_folder / 'y_test.txt')

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

