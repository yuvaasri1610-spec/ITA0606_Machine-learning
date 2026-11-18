import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create dataset
X = np.array([
    [20], [25], [30], [35], [40],
    [45], [50], [55], [60], [65]
])

y = np.array([
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predict output
y_pred = model.predict(X_test)

# Display results
print("Training Data:", X_train.flatten())
print("Testing Data:", X_test.flatten())
print("Actual Output:", y_test)
print("Predicted Output:", y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy * 100, "%")

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict a new value
new_data = np.array([[48]])
result = model.predict(new_data)

print("New Prediction:", result[0])
