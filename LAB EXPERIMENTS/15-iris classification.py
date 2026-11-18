
# Iris Flower Classification using Naive Bayes

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Predict flower type
y_pred = model.predict(X_test)

# Display results
print("Iris Flower Classification")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# Display predictions
print("\nFlower Predictions:")
for i in range(5):
    print("Actual:", data.target_names[y_test[i]],
          "Predicted:", data.target_names[y_pred[i]])
