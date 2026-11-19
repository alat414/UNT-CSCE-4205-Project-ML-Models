# RBF SVM on the UCI HAR dataset
# Plan:
#   1. load the data (official split: 21 people for training, 9 different people for testing)
#   2. tune C and gamma with cross-validation, keeping each person's data in one fold
#   3. train the final model and test it once on the 9 test people
#   4. look at the results: macro-F1, accuracy, confusion matrix, per-person scores, timing

import os
import time
import platform
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import (f1_score, accuracy_score, classification_report,
                             confusion_matrix)

SEED = 42
np.random.seed(SEED)

QUICK = False   # set to True to test the script quickly with a tiny grid

DATA_DIR = "UCI HAR Dataset"
activity_names = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
                  "SITTING", "STANDING", "LAYING"]   # labels 1 to 6 in the dataset


# ------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------

# download the dataset if it's not here yet (the UCI zip has another zip inside it)
if not os.path.isdir(DATA_DIR):
    print("Downloading the dataset...")
    url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
    urllib.request.urlretrieve(url, "har.zip")
    zipfile.ZipFile("har.zip").extractall(".")
    zipfile.ZipFile("UCI HAR Dataset.zip").extractall(".")


def load_data(split):
    # X = 561 features per window, y = activity label, subject = which person
    folder = os.path.join(DATA_DIR, split)
    X = np.loadtxt(os.path.join(folder, f"X_{split}.txt"))
    y = np.loadtxt(os.path.join(folder, f"y_{split}.txt"), dtype=int)
    subject = np.loadtxt(os.path.join(folder, f"subject_{split}.txt"), dtype=int)
    return X, y, subject


X_train, y_train, subj_train = load_data("train")
X_test, y_test, subj_test = load_data("test")

# make sure nobody is in both train and test, otherwise the test isn't about new people
assert len(set(subj_train) & set(subj_test)) == 0

print("Train:", X_train.shape, "people:", sorted(set(subj_train.tolist())))
print("Test: ", X_test.shape, "people:", sorted(set(subj_test.tolist())))


# ------------------------------------------------------------
# 2. Tune C and gamma
# ------------------------------------------------------------

# The scaler is inside the pipeline so it only learns mean/std from the training
# part of each fold (no information from the validation people leaks in).
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", cache_size=1000, random_state=SEED)),
])

if QUICK:
    param_grid = {"svm__C": [10, 100], "svm__gamma": [1e-4]}
else:
    # First tried a wider grid (C up to 1000, gamma up to 1e-2). Best gamma was 1e-4,
    # at the low end, so this grid goes lower. C=1000 was removed because it was very slow.
    param_grid = {"svm__C": [1, 10, 100, 300],
                  "svm__gamma": [1e-5, 3e-5, 1e-4, 3e-4]}

# GroupKFold with groups=subject: each fold validates on people the model didn't train on
search = GridSearchCV(model, param_grid,
                      scoring="f1_macro",
                      cv=GroupKFold(n_splits=5),
                      n_jobs=-1,
                      refit=False,   # I train the final model myself below to time it
                      verbose=1)

start = time.time()
search.fit(X_train, y_train, groups=subj_train)
search_time = time.time() - start

# put the CV scores in a table: rows = C, columns = gamma
cv = pd.DataFrame(search.cv_results_)
cv_table = cv.pivot_table(index="param_svm__C", columns="param_svm__gamma",
                          values="mean_test_score")
print("\nMean CV macro-F1 (rows = C, columns = gamma):")
print(cv_table.round(4))

best_C = search.best_params_["svm__C"]
best_gamma = search.best_params_["svm__gamma"]
print(f"\nBest: C = {best_C}, gamma = {best_gamma}, "
      f"CV macro-F1 = {search.best_score_:.4f}")
print(f"Grid search took {search_time / 60:.1f} minutes")

# if the best value is the first or last one in the grid, the real best might be outside it
if best_C in (param_grid["svm__C"][0], param_grid["svm__C"][-1]) and len(param_grid["svm__C"]) > 1:
    print("Note: best C is on the edge of the grid")
if best_gamma in (param_grid["svm__gamma"][0], param_grid["svm__gamma"][-1]) and len(param_grid["svm__gamma"]) > 1:
    print("Note: best gamma is on the edge of the grid")


# ------------------------------------------------------------
# 3. Train the final model and test it (test set is only used here)
# ------------------------------------------------------------

model.set_params(svm__C=best_C, svm__gamma=best_gamma)

start = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start

start = time.time()
y_pred = model.predict(X_test)
predict_time = time.time() - start

test_f1 = f1_score(y_test, y_pred, average="macro")
test_acc = accuracy_score(y_test, y_pred)

print("\n===== Test results (9 new people) =====")
print(f"Macro-F1: {test_f1:.4f}")
print(f"Accuracy: {test_acc:.4f}\n")
print(classification_report(y_test, y_pred, target_names=activity_names, digits=4))


# ------------------------------------------------------------
# 4. Confusion matrix
# ------------------------------------------------------------

def plot_confusion(cm, title, ax):
    # each cell shows the count and the % of that row (where that activity's windows went)
    cm_percent = cm / cm.sum(axis=1, keepdims=True)
    img = ax.imshow(cm_percent, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels(activity_names, rotation=45, ha="right")
    ax.set_yticks(range(6))
    ax.set_yticklabels(activity_names)
    for i in range(6):
        for j in range(6):
            color = "white" if cm_percent[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{cm_percent[i, j]:.0%}",
                    ha="center", va="center", fontsize=8, color=color)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return img


cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
img = plot_confusion(cm, f"RBF SVM - test confusion matrix (macro-F1 {test_f1:.3f})", ax)
fig.colorbar(img, ax=ax, fraction=0.046, label="share of true class")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 5. Results for each test person
# ------------------------------------------------------------

rows = []
for person in sorted(set(subj_test)):
    mask = subj_test == person
    rows.append({
        "person": person,
        "windows": mask.sum(),
        "macro_f1": f1_score(y_test[mask], y_pred[mask], average="macro"),
        "accuracy": accuracy_score(y_test[mask], y_pred[mask]),
    })
per_person = pd.DataFrame(rows)
print("\nPer-person results:")
print(per_person.round(4).to_string(index=False))

# mean, std and a 95% confidence interval (t-distribution, only 9 people)
n = len(per_person)
mean_f1 = per_person["macro_f1"].mean()
std_f1 = per_person["macro_f1"].std()
margin = stats.t.ppf(0.975, n - 1) * std_f1 / np.sqrt(n)
print(f"\nMacro-F1 per person: {mean_f1:.4f} +/- {std_f1:.4f}")
print(f"95% CI: {mean_f1 - margin:.3f} to {mean_f1 + margin:.3f}")
print(f"Lowest: {per_person['macro_f1'].min():.4f}, highest: {per_person['macro_f1'].max():.4f}")

# confusion matrices for the 2 hardest people, side by side, to see what goes wrong for them
worst_two = per_person.nsmallest(2, "macro_f1")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (_, row) in zip(axes, worst_two.iterrows()):
    mask = subj_test == row["person"]
    cm_person = confusion_matrix(y_test[mask], y_pred[mask], labels=[1, 2, 3, 4, 5, 6])
    plot_confusion(cm_person, f"Person {int(row['person'])} (macro-F1 {row['macro_f1']:.3f})", ax)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 6. Timing and hardware (needed for the report)
# ------------------------------------------------------------

n_support = model.named_steps["svm"].n_support_
print("\n===== Timing =====")
print(f"Grid search:      {search_time / 60:.1f} min")
print(f"Final training:   {train_time:.2f} s")
print(f"Prediction:       {predict_time:.3f} s for {len(X_test)} windows "
      f"({1000 * predict_time / len(X_test):.3f} ms per window)")
print(f"Support vectors:  {n_support.sum()} -> {dict(zip(activity_names, n_support.tolist()))}")
print(f"Hardware: {platform.platform()}, {os.cpu_count()} CPUs, "
      f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}")
