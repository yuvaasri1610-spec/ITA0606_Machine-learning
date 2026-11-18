
# Future Sales Prediction using Linear Regression

import pandas as pd
from sklearn.linear_model import LinearRegression

# Create sales dataset
data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Sales': [10000, 12000, 15000, 18000, 20000,
              22000, 25000, 28000, 30000, 33000]
}

# Convert into DataFrame
df = pd.DataFrame(data)

# Select input and output
X = df[['Month']]
y = df['Sales']

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict future sales
future_months = pd.DataFrame({'Month': [11, 12, 13]})

predictions = model.predict(future_months)

# Display results
print("Future Sales Prediction")
print("-----------------------")

for month, sales in zip(future_months['Month'], predictions):
    print("Month", month, "Predicted Sales: Rs.",
          round(sales, 2))
