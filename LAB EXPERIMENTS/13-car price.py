
# Car Price Prediction Model

import pandas as pd
from sklearn.linear_model import LinearRegression

# Create dataset
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'Mileage': [50000, 45000, 40000, 35000, 30000, 25000, 20000, 15000],
    'Engine': [1000, 1200, 1200, 1500, 1500, 1800, 1800, 2000],
    'Price': [300000, 350000, 400000, 450000, 500000, 600000, 700000, 800000]
}

# Convert into DataFrame
df = pd.DataFrame(data)

# Input features and target
X = df[['Year', 'Mileage', 'Engine']]
y = df['Price']

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predict car price
new_car = [[2023, 10000, 2000]]
prediction = model.predict(new_car)

# Display result
print("Car Price Prediction")
print("--------------------")
print("Year:", new_car[0][0])
print("Mileage:", new_car[0][1])
print("Engine:", new_car[0][2])
print("Predicted Car Price: Rs.", round(prediction[0], 2))
