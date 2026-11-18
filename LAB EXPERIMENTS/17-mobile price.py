
# Mobile Price Prediction using Machine Learning

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Create dataset
data = {
    'RAM': [2, 3, 4, 4, 6, 8, 8, 12, 12, 16],
    'Battery': [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5000, 5500, 6000],
    'Memory': [16, 32, 32, 64, 64, 128, 128, 128, 256, 256],
    'Screen': [4.5, 5.0, 5.2, 5.5, 5.8, 6.0, 6.2, 6.5, 6.7, 6.9],
    'Price_Range': [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
}

# Convert into DataFrame
df = pd.DataFrame(data)

# Select input and output
X = df[['RAM', 'Battery', 'Memory', 'Screen']]
y = df['Price_Range']

# Create and train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Predict price range for a new mobile
new_mobile = [[8, 5000, 128, 6.5]]

prediction = model.predict(new_mobile)

# Display result
print("Mobile Price Prediction")
print("-----------------------")
print("RAM:", new_mobile[0][0], "GB")
print("Battery:", new_mobile[0][1], "mAh")
print("Memory:", new_mobile[0][2], "GB")
print("Screen Size:", new_mobile[0][3], "inches")

print("Predicted Price Range:", prediction[0])

# Display price category
price = {
    0: "Low Cost",
    1: "Medium Cost",
    2: "High Cost",
    3: "Very High Cost"
}

print("Mobile Price Category:", price[prediction[0]])
