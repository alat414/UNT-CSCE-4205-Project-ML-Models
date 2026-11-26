import pandas as pd
import matplotlib.pyplot as plt
import time
import platform
import psutil
import winreg
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Path Setup
currentFolder = Path(__file__).parent #train folder
mainFolder = currentFolder.parent #Root folder (UCI Dataset)
testFolder = mainFolder / "test" #test folder

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


# Function Definitions
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

# Load Train, and Test Data
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

# Preprocessing Training and Test Data sets
scaler = StandardScaler() # function used for standardization

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Radial Basis Function Support Vector Machine (RBF SVM) Baseline
print("\nTraining RBF SVM...")
svm_rbf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)

# Get final training time 
final_training_time = time.perf_counter() - start_training_time
print(f"Training time: {final_training_time:.4f} seconds")

# Start prediction time
start_predict_time = time.perf_counter()

svm_train_pred = svm_rbf.predict(X_train_scaled)
svm_test_pred = svm_rbf.predict(X_test_scaled)

# Accuracy and macro F1 results
print(f"\nRBF SVM Train accuracy (all {X_train_scaled.shape[1]} features):"
      f"{accuracy_score(y_train, svm_train_pred):.4f}")
print(f"\nRBF SVM Test acc (all {X_train_scaled.shape[1]} features):"
      f"{accuracy_score(y_test, svm_test_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, svm_test_pred))

print(f"RBF SVM Test  — Accuracy {accuracy_score(y_test, svm_test_pred):.4f} | "
      f"macro-F1 {f1_score(y_test, svm_test_pred, average='macro'):.4f}")

# Per-class F1 (test set)
per_class_f1 = f1_score(y_test, svm_test_pred, average=None,
                        labels=sorted(y_test.unique()))
print("Per-class F1:", {k: round(float(v), 3)
                        for k, v in zip(sorted(y_test.unique()), per_class_f1)})

# Get final prediction time 
final_predict_time = time.perf_counter() - start_predict_time
print(f"Prediction time: {final_predict_time:.4f} seconds")

# Showing all subject test performance (RBF SVM)
test_eval_svm = test_dataframe[['Subject', 'Activity']].copy()
test_eval_svm['pred'] = svm_test_pred

print("\n" + "="*60)
print("Per-Participant Test Performance (RBF SVM)")
print("="*60)
print(f"{'Subject':>8} {'n_samples':>10} {'accuracy':>10} {'macro_F1':>10}")

per_subj_rows_svm = []
for subj, grp in test_eval_svm.groupby('Subject'):
    acc = (grp['Activity'] == grp['pred']).mean()
    f1  = f1_score(grp['Activity'], grp['pred'], average='macro', zero_division=0)
    per_subj_rows_svm.append((subj, len(grp), acc, f1))
    print(f"{subj:>8} {len(grp):>10} {acc:>10.4f} {f1:>10.4f}")

per_subj_df_svm = pd.DataFrame(
    per_subj_rows_svm,
    columns=['Subject', 'n_samples', 'accuracy', 'macro_F1']
)

print("-" * 60)
print(f"{'mean':>8} {'':>10} {per_subj_df_svm['accuracy'].mean():>10.4f} {per_subj_df_svm['macro_F1'].mean():>10.4f}")
print(f"{'std':>8} {'':>10} {per_subj_df_svm['accuracy'].std():>10.4f} {per_subj_df_svm['macro_F1'].std():>10.4f}")
print(f"{'min':>8} {'':>10} {per_subj_df_svm['accuracy'].min():>10.4f} {per_subj_df_svm['macro_F1'].min():>10.4f}")
print(f"{'max':>8} {'':>10} {per_subj_df_svm['accuracy'].max():>10.4f} {per_subj_df_svm['macro_F1'].max():>10.4f}")

worst = per_subj_df_svm.loc[per_subj_df_svm['accuracy'].idxmin()]
best  = per_subj_df_svm.loc[per_subj_df_svm['accuracy'].idxmax()]
print(f"\n  Worst participant: subject {int(worst['Subject'])} (acc {worst['accuracy']:.4f})")
print(f"  Best  participant: subject {int(best['Subject'])}  (acc {best['accuracy']:.4f})")
print(f"  Spread (max-min):  {best['accuracy'] - worst['accuracy']:.4f}")

# Add Confusion Matrix for quality results
ConfusionMatrixDisplay.from_predictions(
    y_test, svm_test_pred, xticks_rotation=45, cmap='Blues'
)
plt.title("RBF SVM Confusion Matrix - Test Set")
plt.tight_layout()
plt.show()

# Hyperparameter Sweep
C_values = [0.1, 1, 10, 100]

print("\nHyperparameter Sweep (C, gamma):")
print(f"{'C':>10} {'gamma':>10} {'test_acc':>10}")
for C in C_values:
    for gamma in ['scale', 0.001, 0.01, 0.1]:
        model = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        model.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_scaled))
        print(f"{C:>6} {str(gamma):>10} {acc:>10.4f}")