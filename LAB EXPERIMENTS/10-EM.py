
# Expectation Maximization Algorithm

import numpy as np
from sklearn.mixture import GaussianMixture

# Create dataset
X = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [8, 8],
    [9, 8],
    [10, 9]
])

# Create Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=42)

# Train the model using EM algorithm
gmm.fit(X)

# Predict clusters
labels = gmm.predict(X)

# Display results
print("Dataset:")
print(X)

print("\nCluster Labels:")
print(labels)

print("\nCluster Means:")
print(np.round(gmm.means_, 2))
