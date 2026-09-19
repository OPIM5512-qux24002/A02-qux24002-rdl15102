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
