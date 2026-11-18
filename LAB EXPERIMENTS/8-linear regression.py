
# Linear Regression Program

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dataset
# Study hours and marks
X = np.array([[1], [2], [3], [4], [5],
              [6], [7], [8], [9], [10]])

y = np.array([10, 20, 30, 40, 50,
              60, 70, 80, 90, 100])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Display results
print("Training Data:", X_train.flatten())
print("Testing Data:", X_test.flatten())
print("Actual Marks:", y_test)
print("Predicted Marks:", y_pred)

# Calculate accuracy
accuracy = r2_score(y_test, y_pred)
print("R2 Score:", accuracy)

# Predict marks for 7.5 study hours
new_prediction = model.predict([[7.5]])
print("Predicted Marks for 7.5 hours:", new_prediction)
