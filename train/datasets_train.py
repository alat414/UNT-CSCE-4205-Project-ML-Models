import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

current_folder = Path(__file__).parent
main_folder = current_folder.parent

y_train= pd.read_csv(current_folder / "y_train.txt", sep=r"\s+", header=None).values.ravel()

activity_labels = pd.read_csv(main_folder / "activity_labels.txt", sep=r"\s+", header=None, names=["label_id", "activity_name"])

print("Activity labels mapping:")
print(activity_labels)
print(f"\ny_train shape: {y_train.shape}")
print(f"Unique labels in y_train: {sorted(set(y_train))}")

# Mapping numeric labels to activity names
label_to_activity = dict(zip(activity_labels["label_id"], activity_labels["activity_name"]))
y_activity = pd.Series(y_train).map(label_to_activity)

# Count Samples per activity
counts = y_activity.value_counts().sort_index()
percentages = (counts / counts.sum() * 100).round(2)

summary = pd.DataFrame({
    "Activity": counts.index,
    "Samples": counts.values,
    "Percentage": percentages.values
})

print("\n" + "="*60)
print("Activity Distribution in Training Set")
print("="*60)
print(summary.to_string(index=False))
print(f"\nTotal Samples: {counts.sum()}")
print(f"Number of activities: {len(counts)}")

# Plot Counts and Percentages side by side

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left Counts
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor='black')
axes[0].set_title("Activity Distribution (Counts)", fontsize=13)
axes[0].set_xlabel("Activity")
axes[0].set_ylabel("Number of Samples")
axes[0].tick_params(axis='x', rotation=30)
axes[0].grid(axis='y', alpha=0.3)

# Add count labels on top of bars
for bar, count in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                str(count), ha='center', fontsize=10)

# Right percentages
bars2 = axes[1].bar(counts.index, counts.values, color=colors, edgecolor='black')
axes[1].set_title("Activity Distribution (Percentages)", fontsize=13)
axes[1].set_xlabel("Activity")
axes[1].set_ylabel("Percentage (%)")
axes[1].tick_params(axis='x', rotation=30)
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_ylim(0, max(percentages.values) * 1.2)

# Add percentage labels on top of bars
for bar, pct in zip(bars2, percentages.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{pct}%", ha='center', fontsize=10)

plt.tight_layout()
plt.show()