
# Bank Loan Prediction using Naive Bayes

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create sample bank loan dataset
np.random.seed(42)

n = 500

age = np.random.randint(21, 60, n)
income = np.random.randint(15000, 100000, n)
credit_score = np.random.randint(300, 850, n)
loan_amount = np.random.randint(10000, 200000, n)

# Create loan approval target
loan_status = (
    (income > 35000) &
    (credit_score > 600) &
    (loan_amount < income * 3)
).astype(int)

# Combine input features
X = np.column_stack((age, income, credit_score, loan_amount))
y = loan_status

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Predict loan approval
y_pred = model.predict(X_test)

# Display results
print("Bank Loan Prediction Results")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy:",
      accuracy_score(y_test, y_pred) * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict a new loan application
new_customer = [[30, 60000, 750, 100000]]

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("Loan Status: Approved")
else:
    print("Loan Status: Not Approved")
