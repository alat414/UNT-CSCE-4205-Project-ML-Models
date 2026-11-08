import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from matplotlib.colors import ListedColormap

# Path Setup

currentFolder = Path(__file__).parent #train folder
mainFolder = currentFolder.parent #Root folder (UCI Dataset)
testFolder = mainFolder / "test" #test folder


###########################################################################################
# Function Definitions
###########################################################################################
def euclidean_distance(pointOne, pointTwo):
    return np.sqrt(np.sum((np.array(pointOne) - np.array(pointTwo))**2))

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


###########################################################################################
# Get Feature Names
###########################################################################################

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

###########################################################################################
# Get Label Mapping
###########################################################################################
activity_labels = pd.read_csv(mainFolder / "activity_labels.txt", sep=r"\s+", header=None) 

label_to_activity = dict(zip(activity_labels.iloc[:, 0], activity_labels.iloc[:, 1]))

###########################################################################################
# Load, Train, and Test Data
###########################################################################################
print("Loading training data...")
train_dataframe = load_data(currentFolder, mainFolder, unique_names, label_to_activity)

print("Loading test data...")
test_dataframe = load_data(testFolder, mainFolder, unique_names, label_to_activity)

###########################################################################################
# Separate Features and Labels
###########################################################################################
X_train = train_dataframe.drop(columns=['Subject', 'Activity'])
y_train = train_dataframe['Activity']

###########################################################################################
# Display only engineered features
###########################################################################################
X_test = test_dataframe.drop(columns=['Subject', 'Activity'])
y_test = test_dataframe['Activity']

print(f"\nTraining Set Shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

print("\nSubjects in Training Set:", sorted(train_dataframe['Subject'].unique()))
print("Subjects in Test Set:", sorted(test_dataframe['Subject'].unique()))

###########################################################################################
# Preprocessing Training Data sets
###########################################################################################
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

x_set, y_set = X_train, y_train

xOne, xTwo = np.meshgrid(np.arrange(start = x_set[:, 0].min()-1, stop = x_set[:, 0].max()+1, step = 0.01),
                         np.arrange(start = x_set[:, 1].min()-1, stop = x_set[:, 1].max()+1, step = 0.01),)

plt.contourf(xOne, xTwo, knn.predict(np.array([xOne.ravel(), xTwo.ravel()]).T).reshape(xOne.shape),
            alpha =0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(xOne.min(), xOne.max())
plt.ylim(xTwo.min(), xTwo.max())

for i ,j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('blue', 'yellow')(i), label = j))


###########################################################################################
#Project Predicted Subjects
###########################################################################################

plt.title("KNN training set")
plt.xlabel('features')
plt.ylabel('activities')
plt.legend()
plt.show()