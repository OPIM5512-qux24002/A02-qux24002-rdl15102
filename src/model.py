from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)

from sklearn.neural_network import MLPRegressor

model = MLPRegressor(
    hidden_layer_sizes=(77,),
    early_stopping=True
)

model.fit(X_train, y_train)

import os
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

y_train_pred = model.predict(X_train)

plt.figure()
plt.scatter(y_train, y_train_pred)
plt.xlabel("actual price ($100k)")
plt.ylabel("predicted price ($100k)")
plt.title("Model captures the trend but predictions are imprecise")
plt.savefig("figures/train_actual_vs_pred.png")

y_test_pred = model.predict(X_test)

plt.figure()
plt.scatter(y_test, y_test_pred)
plt.xlabel("actual price ($100k)")
plt.ylabel("predicted price ($100k)")
plt.title("Test predictions follow the trend with wide spread")
plt.savefig("figures/test_actual_vs_pred.png")
