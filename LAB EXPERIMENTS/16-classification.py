
# Compare Different Classification Algorithms

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Classification algorithms
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAlgorithm:", name)
    print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=iris.target_names,
        zero_division=0
    ))
