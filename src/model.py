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
from sklearn.pipeline import make_pipeline # added for creating a pipeline
from sklearn.preprocessing import StandardScaler # added for feature scaling
from sklearn.metrics import mean_squared_error, r2_score # added for evaluation metrics


model = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(77,),
        early_stopping=True,
        random_state=42,
    ),
)

model.fit(X_train, y_train)

import os
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
test_rmse = mean_squared_error(y_test, y_test_pred) ** 0.5

print(f"Train R2: {train_r2:.3f}  RMSE: {train_rmse:.3f}")
print(f"Test  R2: {test_r2:.3f}  RMSE: {test_rmse:.3f}")

plt.figure()
plt.scatter(y_train, y_train_pred)
plt.xlabel("actual price ($100k)")
plt.ylabel("predicted price ($100k)")
plt.title("Model captures the trend but predictions are imprecise")
plt.savefig("figures/train_actual_vs_pred.png")



plt.figure()
plt.scatter(y_test, y_test_pred)
plt.xlabel("actual price ($100k)")
plt.ylabel("predicted price ($100k)")
plt.title("Test predictions follow the trend with wide spread")
plt.savefig("figures/test_actual_vs_pred.png")
