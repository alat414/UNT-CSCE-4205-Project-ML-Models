import pandas as pd
import numpy as np
import sklearn
from pathlib import Path
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
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

# Separate Features and Labels
X_train = train_dataframe.drop(columns=['Subject', 'Activity'])
y_train = train_dataframe['Activity']

# Display only engineered features
X_test = test_dataframe.drop(columns=['Subject', 'Activity'])
y_test = test_dataframe['Activity']

print(f"\nTraining Set Shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

print("\nSubjects in Training Set:", sorted(train_dataframe['Subject'].unique()))
print("Subjects in Test Set:", sorted(test_dataframe['Subject'].unique()))

# Preprocessing Training Data sets
scalar = StandardScaler() # function used for standardization

X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

baseline_randomForest = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_randomForest.fit(X_train_scaled, y_train)

# Train Random Forest Model
#print("\nTraining Random Forest...")
#model.fit(X_train_scaled, y_train)

topFeatures = SelectFromModel(baseline_randomForest, threshold='median', prefit=True) # Median retains the top half total engineered features
X_train_selected = topFeatures.transform(X_train_scaled)
X_test_selected = topFeatures.transform(X_test_scaled)

randomForest_selected = RandomForestClassifier(n_estimators=100, random_state=42)
randomForest_selected.fit(X_train_selected, y_train)

#Evaluate on Unseen Subjects
YPrediction = randomForest_selected.predict(X_test_selected)

print("\n" + "="*60)
print("Evaluation on Top Subjects")
print("="*60)
print(f"Accuracy with Feature Selection: {accuracy_score(y_test, YPrediction):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, YPrediction))