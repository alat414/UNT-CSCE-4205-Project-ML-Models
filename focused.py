# Focused study: how sensitive is the RBF SVM to gamma (the kernel width)?
#
# One factor changes: gamma, over a wide log range.
# Everything else stays fixed: C = 100, StandardScaler, all 561 features,
# the official split (21 training people / 9 test people), 5-fold GroupKFold by person, seed 42.

import os
import sys
import time
import platform
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

SEED = 42
np.random.seed(SEED)

QUICK = False          # True = only 3 gamma values, for a fast test run
C_FIXED = 100          # tied for best in our grid search, and faster to train than 300
CHECK_C1 = True        # also draw the CV curve for C = 1 (secondary check, CV only)

if QUICK:
    GAMMAS = [1e-5, 1e-4, 1e-3]
else:
    GAMMAS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

DATA_DIR = "UCI HAR Dataset"
activity_names = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
                  "SITTING", "STANDING", "LAYING"]   # labels 1 to 6
short_names = ["Walk", "Up", "Down", "Sit", "Stand", "Lay"]


# ------------------------------------------------------------
# 1. Load the data 
# ------------------------------------------------------------

if not os.path.isdir(DATA_DIR):
    print("Downloading the dataset...")
    url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
    urllib.request.urlretrieve(url, "har.zip")
    zipfile.ZipFile("har.zip").extractall(".")
    zipfile.ZipFile("UCI HAR Dataset.zip").extractall(".")


def load_data(split):
    folder = os.path.join(DATA_DIR, split)
    X = np.loadtxt(os.path.join(folder, f"X_{split}.txt"))
    y = np.loadtxt(os.path.join(folder, f"y_{split}.txt"), dtype=int)
    subject = np.loadtxt(os.path.join(folder, f"subject_{split}.txt"), dtype=int)
    return X, y, subject


X_train, y_train, subj_train = load_data("train")
X_test, y_test, subj_test = load_data("test")
assert len(set(subj_train) & set(subj_test)) == 0

print("Train:", X_train.shape, "| Test:", X_test.shape)
print(f"Gamma values: {GAMMAS}")
print(f"Fixed: C = {C_FIXED}, StandardScaler, 561 features, 5-fold GroupKFold by person\n")


def make_model(gamma, C):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=C, gamma=gamma, cache_size=1000, random_state=SEED)),
    ])


cv_folds = GroupKFold(n_splits=5)


# ------------------------------------------------------------
# 2. The sweep: for each gamma, cross-validate and then test
# ------------------------------------------------------------

results = []          # one row per gamma
oof_predictions = {}  # out-of-fold CV predictions for every training window, per gamma

sweep_start = time.time()
for gamma in GAMMAS:
    print(f"gamma = {gamma:g} ...", end=" ", flush=True)

    # cross-validation on the training people; keep the fitted fold models so we can
    # also get the out-of-fold predictions (for per-class scores and confusion matrices)
    cv = cross_validate(make_model(gamma, C_FIXED), X_train, y_train,
                        groups=subj_train, cv=cv_folds, scoring="f1_macro",
                        return_train_score=True, return_estimator=True,
                        return_indices=True, n_jobs=-1)

    oof = np.zeros_like(y_train)
    for fold_model, val_idx in zip(cv["estimator"], cv["indices"]["test"]):
        oof[val_idx] = fold_model.predict(X_train[val_idx])
    oof_predictions[gamma] = oof
    per_class_cv = f1_score(y_train, oof, average=None, labels=[1, 2, 3, 4, 5, 6])

    # final model on all training data, then the test set (analysis only)
    model = make_model(gamma, C_FIXED)
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start

    per_person = [f1_score(y_test[subj_test == p], y_pred[subj_test == p], average="macro")
                  for p in sorted(set(subj_test))]

    row = {
        "gamma": gamma,
        "cv_f1": cv["test_score"].mean(),
        "cv_std": cv["test_score"].std(),
        "train_f1": cv["train_score"].mean(),
        "gap": cv["train_score"].mean() - cv["test_score"].mean(),
        "n_sv": int(model.named_steps["svm"].n_support_.sum()),
        "train_s": train_time,
        "pred_ms": 1000 * predict_time / len(X_test),
        "test_f1": f1_score(y_test, y_pred, average="macro"),
        "test_acc": accuracy_score(y_test, y_pred),
        "worst_person_f1": min(per_person),
    }
    for name, score in zip(short_names, per_class_cv):
        row[f"cv_f1_{name}"] = score
    results.append(row)
    print(f"CV macro-F1 = {row['cv_f1']:.4f}, train = {row['train_f1']:.4f}, "
          f"support vectors = {row['n_sv']}")

sweep_time = time.time() - sweep_start
res = pd.DataFrame(results)


# ------------------------------------------------------------
# 3. Optional check: the same CV curve with C = 1
# ------------------------------------------------------------

if CHECK_C1:
    print("\nSecondary check: CV curve with C = 1")
    c1_scores = []
    for gamma in GAMMAS:
        scores = cross_validate(make_model(gamma, 1), X_train, y_train,
                                groups=subj_train, cv=cv_folds,
                                scoring="f1_macro", n_jobs=-1)["test_score"]
        c1_scores.append((scores.mean(), scores.std()))
        print(f"  gamma = {gamma:g}: CV macro-F1 = {scores.mean():.4f}")
    res["cv_f1_C1"] = [m for m, s in c1_scores]
    res["cv_std_C1"] = [s for m, s in c1_scores]


# ------------------------------------------------------------
# 4. Results tables
# ------------------------------------------------------------

pd.set_option("display.width", 160)


def table(columns):
    # round the scores, but print gamma as 1e-05 etc. (rounding would turn it into 0.0000)
    t = res[columns].round(4)
    t["gamma"] = [f"{g:g}" for g in res["gamma"]]
    return t.to_string(index=False)


print("\n===== Sensitivity to gamma (C = 100) =====")
print(table(["gamma", "cv_f1", "cv_std", "train_f1", "gap", "n_sv", "train_s", "pred_ms"]))

print("\nPer-class CV F1 (from out-of-fold predictions on the training people):")
print(table(["gamma"] + [f"cv_f1_{n}" for n in short_names]))

print("\nTest set, for analysis only (not used to choose gamma):")
print(table(["gamma", "test_f1", "test_acc", "worst_person_f1"]))

# how sensitive is it? -> which gammas stay within 0.01 macro-F1 of the best CV score
best = res.loc[res["cv_f1"].idxmax()]
close = res[res["cv_f1"] >= best["cv_f1"] - 0.01]["gamma"]
print(f"\nBest CV gamma: {best['gamma']:g} (CV macro-F1 {best['cv_f1']:.4f})")
print(f"Gammas within 0.01 of the best: {close.min():g} to {close.max():g} "
      f"(a {close.max() / close.min():.0f}x range)")
print(f"Lowest CV macro-F1 in the sweep: {res['cv_f1'].min():.4f} "
      f"at gamma = {res.loc[res['cv_f1'].idxmin(), 'gamma']:g}")
print(f"\nSweep took {sweep_time / 60:.1f} min")


# ------------------------------------------------------------
# 5. Figures
# ------------------------------------------------------------

# Figure 1: train vs CV macro-F1 (the main result)
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(res["gamma"], res["cv_f1"], yerr=res["cv_std"], marker="o", capsize=3,
            label=f"CV (validation people), C = {C_FIXED}")
ax.plot(res["gamma"], res["train_f1"], marker="s", linestyle="--",
        label=f"Training people, C = {C_FIXED}")
if CHECK_C1:
    ax.errorbar(res["gamma"], res["cv_f1_C1"], yerr=res["cv_std_C1"], marker="^",
                capsize=3, alpha=0.6, label="CV, C = 1 (secondary check)")
ax.axvspan(close.min(), close.max(), color="green", alpha=0.1,
           label="within 0.01 of best CV")
ax.set_xscale("log")
ax.set_xlabel("gamma (log scale)")
ax.set_ylabel("macro-F1")
ax.set_title("RBF SVM: sensitivity to gamma")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 2: model size and speed
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(res["gamma"], res["n_sv"], marker="o")
ax1.axhline(len(X_train), color="gray", linestyle=":", label="all training windows")
ax1.set_xscale("log")
ax1.set_xlabel("gamma (log scale)")
ax1.set_ylabel("number of support vectors")
ax1.set_title("Support vectors")
ax1.legend()
ax1.grid(alpha=0.3)
ax2.plot(res["gamma"], res["pred_ms"], marker="o", label="prediction (ms per window)")
ax2.plot(res["gamma"], res["train_s"], marker="s", label="training (s, all data)")
ax2.set_xscale("log")
ax2.set_xlabel("gamma (log scale)")
ax2.set_title("Training and prediction time")
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 3: per-class CV F1 -- which activities break first?
fig, ax = plt.subplots(figsize=(8, 5))
for name, full in zip(short_names, activity_names):
    ax.plot(res["gamma"], res[f"cv_f1_{name}"], marker="o", label=full)
ax.set_xscale("log")
ax.set_xlabel("gamma (log scale)")
ax.set_ylabel("CV F1 (out-of-fold)")
ax.set_title("Per-class F1 vs gamma")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Figure 4: confusion matrices at three settings (smallest, best, largest gamma)
picks = [("smallest gamma", GAMMAS[0]), ("best gamma", best["gamma"]),
         ("largest gamma", GAMMAS[-1])]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (label, gamma) in zip(axes, picks):
    cm = confusion_matrix(y_train, oof_predictions[gamma], labels=[1, 2, 3, 4, 5, 6])
    cm_percent = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cm_percent, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels(short_names)
    ax.set_yticks(range(6))
    ax.set_yticklabels(short_names)
    for i in range(6):
        for j in range(6):
            color = "white" if cm_percent[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{cm_percent[i, j]:.0%}",
                    ha="center", va="center", fontsize=8, color=color)
    f1 = f1_score(y_train, oof_predictions[gamma], average="macro")
    ax.set_title(f"{label}: gamma = {gamma:g}\nCV macro-F1 = {f1:.3f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 6. Hardware
# ------------------------------------------------------------

print(f"\nHardware: {platform.platform()}, {os.cpu_count()} CPUs, "
      f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}")
