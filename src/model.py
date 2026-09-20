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
