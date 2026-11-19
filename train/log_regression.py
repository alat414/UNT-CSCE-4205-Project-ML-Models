from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import sklearn
from pathlib import Path
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score

# Path Setup
currentFolder = Path(__file__).parent #train folder
mainFolder = currentFolder.parent #Root folder (UCI Dataset)
testFolder = mainFolder / "test" #test folder

# Function Definitions
def load_data(dataFolder, mainFolder, feature_names, label_map):
    # Load x data
    XData = pd.read_csv(dataFolder/ f"X_{dataFolder.name}.txt", sep=r"\s+", header=None)
    XData.columns = feature_names

    #Load y data
    YRaw = pd.read_csv(dataFolder/ f"y_{dataFolder.name}.txt", sep=r"\s+", header=None).values.ravel()
    subject = pd.read_csv(dataFolder/ f"subject_{dataFolder.name}.txt", sep=r"\s+", header=None).values.ravel()

    # Map Labels
    yLabels = pd.Series(YRaw).map(label_map)

    # Add to dataframe for tracking purposes
    XData['Subject'] = subject
    XData['Activity'] = yLabels

    return XData

# Get Feature Names
features = pd.read_csv(mainFolder / "features.txt", sep=r"\s+", header=None)
feature_names = features[1].values.tolist()

seen = {}
unique_names = []

for name in feature_names:
    if name in seen:
        seen[name] += 1
        unique_names.append(f"{name}_{seen[name]}")
    else:
        seen[name] = 0
        unique_names.append(name)

# Get Label Mapping
activity_labels = pd.read_csv(mainFolder / "activity_labels.txt", sep=r"\s+", header=None) 

label_to_activity = dict(zip(activity_labels.iloc[:, 0], activity_labels.iloc[:, 1]))

# Load, Train, and Test Data
print("Loading training data...")
train_dataframe = load_data(currentFolder, mainFolder, unique_names, label_to_activity)

print("Loading test data...")
test_dataframe = load_data(testFolder, mainFolder, unique_names, label_to_activity)

# Training data
X = train_dataframe.drop(columns=['Subject', 'Activity'])
y = train_dataframe['Activity']

# Testing data
X_t = test_dataframe.drop(columns=['Subject', 'Activity'])
y_t = test_dataframe['Activity']

# Split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale training set for sensitivity
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Make predictions about test set
y_pred = model.predict(X_test)
print(" Predicted Labels:")
print(y_pred)

# Model evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(model,X_test,y_test,cmap='Blues')
plt.title("Confusion Matrix for Multiclass Logistic Regression")
plt.show()