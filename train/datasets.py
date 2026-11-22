import pandas as pd
import numpy as np
import platform
import psutil
import winreg
import time
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
# Path Setup

currentFolder = Path(__file__).parent #train folder
mainFolder = currentFolder.parent #Root folder (UCI Dataset)
testFolder = mainFolder / "test" #test folder

# Function Definitions

# Hardware Specifications
def get_hardware_info():
    info = {
        "os": f"{platform.system()}{platform.release()}",
        "processor": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu": "None"
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3),2)
    except ImportError:
        pass
    return info

# Gather hardware specifications prior to training
key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                     r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
print(f"{cpu_name, _  }")
hardware = get_hardware_info()
print(f"Hardware Specifications")
print("*" * 20)
for item, item1 in hardware.items():
    print(f"{item:>20}: {item1}")
print("*" * 20)

def load_data(dataFolder, feature_names, label_map):
    # Load x data
    XData = pd.read_csv(dataFolder/ f"X_{dataFolder.name}.txt", sep=r"\s+", header=None)
    XData.columns = feature_names

    #Load y data
    YRaw = pd.read_csv(dataFolder/ f"y_{dataFolder.name}.txt", sep=r"\s+", header=None).values.ravel()
    subject = pd.read_csv(dataFolder/ f"subject_{dataFolder.name}.txt", sep=r"\s+", header=None).values.ravel()

    # Map Labels
    yLabels = pd.Series(YRaw).map(label_map).to_numpy()

    # Joined metadata frame
    meta = pd.DataFrame({"Subject": subject, "Activity": yLabels})

    return pd.concat([XData.reset_index(drop=True), meta], axis=1)

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

# Start train time
start_training_time = time.perf_counter()

# Load, Train, and Test Data

print("Loading training data...")
train_dataframe = load_data(currentFolder, unique_names, label_to_activity)

print("Loading test data...")
test_dataframe = load_data(testFolder, unique_names, label_to_activity)

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

# Get final training time 
final_training_time = time.perf_counter() - start_training_time
print(f"Training time: {final_training_time:.4f} seconds")

# Start prediction time
start_predict_time = time.perf_counter()
# Preprocessing Training and Test Data sets
scaler = StandardScaler() # function used for standardization

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

baseline_randomForest = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
baseline_randomForest.fit(X_train_scaled, y_train)

# Baseline Random Forest Prediction
baseline_prediction_train = baseline_randomForest.predict(X_train_scaled)
baseline_prediction_test = baseline_randomForest.predict(X_test_scaled)
print(f"Baseline Random Forests training set (all {X_train_scaled.shape[1]} features): {accuracy_score(y_train, baseline_prediction_train):.4f}")
print(f"Baseline Random Forests test set (all {X_train_scaled.shape[1]} features): {accuracy_score(y_test, baseline_prediction_test):.4f}")

top_Features = SelectFromModel(baseline_randomForest, threshold='median', prefit=True) # Median retains the top half total engineered features
X_train_selected = top_Features.transform(X_train_scaled)
X_test_selected = top_Features.transform(X_test_scaled)

randomForest_selected = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
randomForest_selected.fit(X_train_selected, y_train)

#Evaluate on Unseen Subjects
Y_Prediction = randomForest_selected.predict(X_test_selected)

print("\n" + "="*60)
print("Evaluation on Top Subjects")
print("="*60)
print(f"Accuracy with Feature Selection: {accuracy_score(y_test, Y_Prediction):.4f}\n")
print(f"Train acc (selected):  {accuracy_score(y_train, randomForest_selected.predict(X_train_selected)):.4f}")
print(f"Features retained: {X_train_selected.shape[1]} / {X_train_scaled.shape[1]}")
print("Classification Report:")
print(classification_report(y_test, Y_Prediction))

# Get final prediction time 
final_predict_time = time.perf_counter() - start_predict_time
print(f"Predicton time: {final_predict_time:.4f} seconds")

# Adding multiple threshold percentages to see which is the most optimal choice
thresholds = ['median', 0.001, 0.002, 0.005]
print("\nThreshold Sweep:")
print(f"{'threshold':>10} {'n_features':>12} {'test_acc':>10}")
for thr in thresholds:
    selector = SelectFromModel(baseline_randomForest, threshold=thr, prefit=True)
    set_XTrain = selector.transform(X_train_scaled)
    set_XTest = selector.transform(X_test_scaled)
    random_Forest_sweep = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    random_Forest_sweep.fit(set_XTrain, y_train)
    print(f"{str(thr):>10} {set_XTrain.shape[1]:>12} {accuracy_score(y_test, random_Forest_sweep.predict(set_XTest)):>10.4f}")

# Feature Importance Plot
importances = pd.Series(baseline_randomForest.feature_importances_, index=X_train.columns)
top20 = importances.nlargest(20)
top20.sort_values().plot.barh(figsize=(8,6))
plt.title("Top 20 Feature Importances for Baseline Random Forests")
plt.show()

# Add Confusion Matrix for quality results
ConfusionMatrixDisplay.from_predictions(y_test, Y_Prediction, xticks_rotation=45, cmap='Blues')
plt.title("Random Forests Confusion Matrix - Test Set")
plt.show()