from sklearn.neural_network import MLPRegressor

model = MLPRegressor(
    hidden_layer_sizes=(77,),
    early_stopping=True
)

model.fit(X_train, y_train)
