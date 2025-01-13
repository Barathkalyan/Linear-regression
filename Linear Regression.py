import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Visualization - Reduce dimensions to 2 using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Split the reduced dataset
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
    X_reduced, y_bin, test_size=0.25, random_state=42
)

# Fit linear regression again on reduced data
model.fit(X_train_red, y_train_red)
y_pred_red = model.predict(X_test_red)

# Plot regression predictions
x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict(grid).reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, probs, alpha=0.8, cmap="coolwarm")
plt.scatter(X_test_red[:, 0], X_test_red[:, 1], c=y_test_red, edgecolor="k", cmap="coolwarm")
plt.xlabel("Feature 1 (PCA)")
plt.ylabel("Feature 2 (PCA)")
plt.title("Linear Regression Predictions (PCA-reduced features)")
plt.colorbar(label="Predicted Value")
plt.show()
