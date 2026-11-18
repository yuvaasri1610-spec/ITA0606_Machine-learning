
# Credit Score Classification

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create dataset
data = {
    'Income': [25000, 30000, 45000, 50000, 60000,
               70000, 80000, 90000, 35000, 75000],
    'Age': [22, 25, 30, 35, 40, 45, 50, 55, 28, 48],
    'Loan': [10000, 15000, 20000, 25000, 30000,
             35000, 40000, 45000, 12000, 38000],
    'Credit_Score': ['Low', 'Low', 'Medium', 'Medium',
                     'Good', 'Good', 'Good', 'Good',
                     'Low', 'Good']
}

df = pd.DataFrame(data)

# Input and output
X = df[['Income', 'Age', 'Loan']]
y = df['Credit_Score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Display results
print("Actual Credit Scores:", list(y_test))
print("Predicted Credit Scores:", list(y_pred))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100)

# Predict for a new person
new_person = pd.DataFrame(
    [[65000, 38, 25000]],
    columns=['Income', 'Age', 'Loan']
)

result = model.predict(new_person)
print("New Person Credit Score:", result[0])
