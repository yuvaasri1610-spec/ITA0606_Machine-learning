
# Naive Bayes Algorithm

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Predict the output
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Naive Bayes Classification Results")
print("\nConfusion Matrix:")
print(cm)

print("\nAccuracy:", accuracy * 100, "%")

# Display confusion matrix graph
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)

display.plot(cmap='Blues')
plt.title("Naive Bayes Confusion Matrix")
plt.show()

# Display sample predictions
print("\nSample Predictions:")
for i in range(5):
    print("Actual:", data.target_names[y_test[i]],
          "Predicted:", data.target_names[y_pred[i]])
