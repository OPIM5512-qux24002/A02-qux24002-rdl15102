# A02 Ping Pong: California Housing MLP

Predicts median house value in California from eight census features using a neural
network.

## Partners

- Raymond Larson
- Davide Trani

## How to run

```bash
pip install -r src/requirements.txt
python src/model.py
```


## What the script does

1. Loads the California Housing dataset from scikit-learn
2. Splits 80/20 into train and test 
3. Standardizes the features with `StandardScaler` inside a pipeline
4. Trains an `MLPRegressor` with one hidden layer of 77 units and
   `early_stopping=True`
5. Prints R2 and RMSE for train and test
6. Saves two actual-vs-predicted plots:
   - `figures/train_actual_vs_pred.png`
   - `figures/test_actual_vs_pred.png`

## Results

Test R2 is about 0.76 so there is room for improvement with this model. 
