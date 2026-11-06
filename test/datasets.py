import pandas as pd
import numpy as np
import sklearn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Path Setup

currentFolder = Path(__file__).parent 
mainFolder = currentFolder.parent
testFolder = mainFolder / "test"

# Function Definitions

def load_data(dataFolder, mainFolder, feature_names, label_map):
    # Load x data
    XData = pd.read_csv(testFolder/ f"X_test.txt", sep=r"\s+", header=None)
    XData.columns = feature_names

    #Load y data
    YRaw = pd.read_csv(testFolder/ f"y_test.txt", sep=r"\s+", header=None).values.ravel()
    subject = pd.read_csv(testFolder/ f"subject_train.txt", sep=r"\s+", header=None).values.ravel()

    # Map Labels
    yLabels = pd.Series(YRaw).map(label_map)

    # Add to dataframe for tracking purposes
    XData['Subject'] = subject
    XData['Activity'] = YRaw

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

activity_labels = pd.read_csv(mainFolder / "activity_labels.txt", sep=r"\s+", header=None, names=["label_id", "activity_names"])
label_to_activity = dict(zip(activity_labels["label_id"], activity_labels["activity_name"]))

# Load, Train, and Test Data

print("Loading training data")
train_dataframe = load_data(currentFolder, mainFolder, unique_names, label_to_activity)

print("Loading training data")
test_dataframe = load_data(testFolder, mainFolder, unique_names, label_to_activity)

# Separate Features and Labels

# Display only engineered features
X_test = test_dataframe.drop(columns=['Subject', 'Activity'])
y_test = test_dataframe['Activity']

print(f"\nTraining Set Shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

print("\nSubjects in Test Set:", sorted(test_dataframe['Subject'].unique()))
print("Subjects in Test Set:", sorted(test_dataframe['Subject'].unique()))

# Preprocessing Training Data sets
scalar = StandardScaler() # function used for standardization

X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

# Train Random Forest Model
print("\nTraining Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

#Evaluate on Unseen Subjects
YPrediction = model.predict(X_test_scaled)

print("\n" + "="*60)
print("Evaluation on Unseen Test Subjects (22-30)")
print("="*60)
print(f"Accuracy: {accuracy_score(y_test,YPrediction):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, YPrediction))