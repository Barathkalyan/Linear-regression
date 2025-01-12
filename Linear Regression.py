import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert target to binary classification (0 vs not-0)
y_bin = np.where(y == 0, 0, 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.25, random_state=42)

# Logistic Regression Model
model = LogisticRegression(C=0.5, max_iter=200, solver="liblinear", penalty="l1")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print metrics
print(f"Accuracy: {acc * 100:.2f} %")
print(f"Confusion Matrix:\n{conf}")
print(f"Classification Report:\n{class_report}")

# Visualization - Reduce dimensions to 2 using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Split the reduced dataset
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
    X_reduced, y_bin, test_size=0.25, random_state=42
)

# Fit logistic regression again on reduced data
model.fit(X_train_red, y_train_red)
y_pred_red = model.predict(X_test_red)

# Plot decision boundary
x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, probs, alpha=0.8, cmap="coolwarm")
plt.scatter(X_test_red[:, 0], X_test_red[:, 1], c=y_test_red, edgecolor="k", cmap="coolwarm")
plt.xlabel("Feature 1 (PCA)")
plt.ylabel("Feature 2 (PCA)")
plt.title("Logistic Regression Decision Boundary")
plt.colorbar(label="Probability of Class 1")
plt.show()
