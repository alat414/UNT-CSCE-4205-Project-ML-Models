import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


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
print("Scaling the data...")
scaler = StandardScaler()
XTrainScaled = scaler.fit_transform(X_train)
XTestScaled = scaler.transform(X_test)

print("Training KNN k=5")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(XTrainScaled, y_train)

# Evaluate on unseen subjects
YPredicition = knn.predict(XTestScaled)
print(f"\nKNN Test Accuracy: {accuracy_score(y_test, YPredicition):.4f}\n")
print(classification_report(y_test, YPredicition))

# Using PCA for graphic visualization
print("Creating 2D PCA projection for visualization...")
pca = PCA(n_components=2)
XTrain_PCA = pca.fit_transform(XTrainScaled)

xMin, xMax = XTrain_PCA[:, 0].min() - 1, XTrain_PCA[:, 0].max() + 1
yMin, yMax = XTrain_PCA[:, 1].min() - 1, XTrain_PCA[:, 1].max() + 1
xGrid, yGrid = np.meshgrid(np.arange(xMin, xMax, 0.1),
                          np.arange(yMin, yMax, 0.1))

# Training new KNN 
knn2D = KNeighborsClassifier(n_neighbors=5)
knn2D.fit(XTrain_PCA, y_train) 

#Predict on the meshgrid
graph2D = knn2D.predict(np.c_[xGrid.ravel(), yGrid.ravel()])
graph2D = graph2D.reshape(xGrid.shape)

# Convert strings to numbers for matplotlib
y_train_cat = y_train.astype('category')
y_train_codes = y_train_cat.cat.codes

# Convert meshgrid predictions to the same numbers
graph2D_codes = pd.Categorical(graph2D.ravel(), categories=y_train_cat.cat.categories).codes
graph2D_codes = graph2D_codes.reshape(xGrid.shape)

# Color codes
colors = ListedColormap(['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#c2c2f0', '#ffb3e6'])

# Plotting
plt.figure(figsize=(10, 8))
plt.contourf(xGrid, yGrid, graph2D_codes, alpha=0.4, cmap=colors)
scatter = plt.scatter(XTrain_PCA[:, 0], XTrain_PCA[:, 1],
                      c=y_train_codes, cmap=colors, edgecolors='k', s=30)

plt.title("KNN Decision Boundary (PCA reduced to 2D)")
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')

plt.legend(handles=scatter.legend_elements()[0], labels=list(y_train_cat.cat.categories), loc='best')
plt.show()