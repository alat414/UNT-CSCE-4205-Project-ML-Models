Synopsis:
The objective of this project is to use machine learning method that can recognize the activity of a 
participant whose data was not used for training. 

The experiment had classified basic human physical activites using a smartphone accelerometer 
and gyroscope recordings. 

The UCI Human Activity Recognition dataset contains over 10,000 windows of data from 30 participants, 6 activity classes,
and 561 engineered features, and corresponding preprocessed sensor sequences. The represenataions of this data is shown
by tabular and sequential methods.

Gustavo Alatriste – K Nearest Neighbors, Documentation and Coding
Aerith Brieden – Random Forests, Testing
Atlas Infante – Logarithmic Classifiers, Testing
Ngoc Yen Nhi Ha- RBF Kernel Support Vector Machine, Presentation Slides

ML Methods/Pipelines:
For this project, we will use the K Nearest Neighbors, Random Forests, Logarithmic
Classifiers, and the RBF Kernel Support Vector Machine methods to experiment on the
given datasets and related features.
Moreover, the RBF method will be our focused selection for a more in-depth study. Given
the dataset’s size and variety including over 7k recordings of human movement and over
560 engineered features extracted, the RBF Kernel SVM allows for accurate measurements
and predictions.

Focused Experimental Question:
How sensitive is the selected model to one important hyperparameter or modeling choice?

Data Splitting:
Using the test dataset, we can evaluate how well the chosen models can predict the
engineered features on unseen subjects.
However, to create more accurate predictions without data leakage (our predictions being
“too” right), performing cross validation on the training dataset can help us achieve this
objective.

In this project, the dataset will be tested across all models using the same fixed set and
splits. As the project continues, having varying splits on either the training or test datasets
will be necessary for more complex machine learning models.

Control/Fixed Factors:
In the RBF-Kernal SVM, all factors include the data points and the gamma variable, γ, the
parameter that says how quickly similarity falls off.
K(x,x') = e^ (- γ ||x – x'||^2)
x,x' are the chosen data points; γ is the gamma variable, e log function. Hence, the control
factor is the gamma variable, and the fixed ones would be the dataset points because
those don’t change in value.

In the KNN method, the controlled factor would be the number of neighbors used for
plotting new points. An exact value has not been decided yet, but it is expected to be
between 10 and 20.
In the Random Forests method, a controlled factor would be the number of decision trees
made. Similarly to the KNN method, an exact value for the generated number of trees is not
yet decided but is expected to be in the range of 100 to 200.

Evaluation Metrics:
We'll primarily use macro-F1 to ensure all six activities are weighted equally, preventing
frequent classes from masking errors in minority ones. Along with overall accuracy and
confusion matrices, this helps us catch specific mix-ups—like confusing active walking
with standing. To test true generalization, we'll evaluate strictly on unseen participants,
while also tracking per-person accuracy and computation runtimes.

Future Concepts:
Newer concepts: A concept we may utilize during the project is the kernel regression
model. This can be useful for distinguishing complex engineering features from both the
training and test datasets. It does this by using a kernelized model or a nonparametric
technique where smooth curves are produced using nearby observations and weighing
them based on the kernel function. The kernel trick helps transform the low dimensional
data into either high dimensional or infinite dimensional spaces and performs such model
without computing the data’s coordinates.
Another future concept is the stochastic gradient descent. Unlike a normal gradient
descent function that uses an entire dataset for calculating the loss function, this
approach uses only a selected, randomly chosen training example at each iteration. This
can be very useful to avoid data leakage.

Student-written component:
A custom NumPy implementation of PCA using SVD and a fully customized PyTorch
training/evaluation loop with explicit gradient tracking and validation checkpointing.


Initial Results:
The first table shows the a selected window of a time recording, 2.56 seconds to be exact, versus
a selected number engineered features generated. Each value consists of an average of the several senor readings recored, 
along with the corresponding subject and activity. 
The second table shows three features's averages on all activities peformed. The data roughly shows the mean in each direction
of movement (x, y, and z directions).

<img width="920" height="368" alt="Image" src="https://github.com/user-attachments/assets/6e0a3c8e-d448-4cc7-8e4e-b57f1b43645a"/>