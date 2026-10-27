import pandas as pd
from pathlib import Path

current_folder = Path(__file__).parent
main_folder = current_folder.parent

# Load features names
features = pd.read_csv(main_folder / "features.txt", sep=r"\s+", header=None)
feature_names = features[1].values.tolist()

# Make duplicate names unique (e.g., name, name_1, name_2)
seen = {}
unique_names = []
for name in feature_names:
    if name in seen:
        seen[name] += 1
        unique_names.append(f"{name}_{seen[name]}")
    else:
        seen[name] = 0
        unique_names.append(name)


#Load X_train and apply the feature names
X_train = pd.read_csv(current_folder / "X_train.txt", sep=r"\s+", header=None)
X_train.columns = unique_names

# Load labels and subjects
y_train_raw = pd.read_csv(current_folder / "y_train.txt", sep=r"\s+", header=None).values.ravel()
subject_train = pd.read_csv(current_folder / "subject_train.txt", sep=r"\s+", header=None).values.ravel()

# Map numeric activities to names
activity_labels = pd.read_csv(main_folder / "activity_labels.txt", sep=r"\s+", header=None, names=["label_id", "activity_name"])
label_to_activity = dict(zip(activity_labels["label_id"], activity_labels["activity_name"]))
y_train_named = pd.Series(y_train_raw).map(label_to_activity)

#Add subject and activity to df
X_train['Subject'] = subject_train
X_train['Activity'] = y_train_named

print("Combined DataFrame shape:", X_train.shape)
print("\n First five rows and five columns:")
print(X_train.iloc[:10, list(range(3)) + [-2, -1]].to_string())

# View a specific feature (e.g., 'tBodyAcc-mean()-X')
#print(X_train['tBodyAcc-mean()-X'].head())


# Look at the average of all features for each activity
print(X_train.groupby('Activity').mean().iloc[:, :6])