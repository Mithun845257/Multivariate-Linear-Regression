# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
Import Libraries and Load Dataset
Import the required Python libraries and load the California Housing dataset.
### Step2
Define Input and Output Variables
Store the feature matrix in X and target values in y.
### Step3
Split the Dataset
Divide the dataset into training and testing sets using train_test_split().
### Step4
Create and Train the Linear Regression Model
Create a Linear Regression object and train it using the training data.
### Step5
Predict and Plot Residual Errors
Predict the output values, calculate residual errors, and display the residual error graph using Matplotlib.

## Program:
```
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# Load the California Housing dataset
housing = fetch_california_housing()

# Defining feature matrix (X) and response vector (y)
X = housing.data
y = housing.target

# Splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# Create linear regression object
reg = linear_model.LinearRegression()

# Train the model using the training sets
reg.fit(X_train, y_train)

# Regression coefficients
print("Coefficients:\n", reg.coef_)

# Variance score
print("Variance score: {:.2f}".format(reg.score(X_test, y_test)))

# Setting plot style
plt.style.use('fivethirtyeight')

# Plotting residual errors in training data
plt.scatter(
    reg.predict(X_train),
    reg.predict(X_train) - y_train,
    color="green",
    s=10,
    label="Train data"
)

# Plotting residual errors in test data
plt.scatter(
    reg.predict(X_test),
    reg.predict(X_test) - y_test,
    color="blue",
    s=10,
    label="Test data"
)

# Plotting line for zero residual error
plt.hlines(y=0, xmin=0, xmax=6, linewidth=2)

# Plotting legend
plt.legend(loc="upper right")

# Plot title
plt.title("Residual Errors")

# Show plot
plt.show()




```
## Output:
<img width="640" height="551" alt="image" src="https://github.com/user-attachments/assets/6a3ead82-0bf9-47cf-b870-373b572da938" />

### Insert your output

<br>

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
