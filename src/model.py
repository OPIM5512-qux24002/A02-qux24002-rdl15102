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

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
test_rmse = mean_squared_error(y_test, y_test_pred) ** 0.5


#Code for plotting
import os
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)


def plot_actual_vs_pred(actual, predicted, split_name, r2, rmse, path):
    plt.figure(figsize=(6, 6))
    plt.scatter(actual, predicted, alpha=0.2, s=10)
    lo, hi = actual.min(), actual.max()
    plt.plot([lo, hi], [lo, hi], color="red", linestyle="--", label="perfect prediction")
    plt.xlabel("actual price ($100k)")
    plt.ylabel("predicted price ($100k)")
    plt.title(f"{split_name}: actual vs predicted (R2 = {r2:.3f}, RMSE = {rmse:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


plot_actual_vs_pred(y_train, y_train_pred, "Train", train_r2, train_rmse,
                    "figures/train_actual_vs_pred.png")
plot_actual_vs_pred(y_test, y_test_pred, "Test", test_r2, test_rmse,
                    "figures/test_actual_vs_pred.png")