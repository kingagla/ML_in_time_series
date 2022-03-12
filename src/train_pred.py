import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from src import time_counter


@time_counter
def train_series(model, objective_fun, n_trials, y, x=None, **model_params):
    study = optuna.create_study()  # Create a new study.
    study.optimize(lambda trial: objective_fun(trial, y, x),
                   n_trials=n_trials)  # Invoke optimization of the objective function.
    params = study.best_params

    model_params.update(params)
    model.set_params(**model_params)
    model.fit(X=x, y=y)

    return model


def predict_series(model, x=None, fh=None):
    if isinstance(model, (AutoARIMA, ExponentialSmoothing)):
        prediction = model.predict(fh)

    elif isinstance(model, RandomForestRegressor):
        y_pred = model.predict(x.values.reshape(-1, 1))
        prediction = pd.Series(data=y_pred, index=fh)

    return prediction
