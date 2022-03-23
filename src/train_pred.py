import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
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


def predict_series(model, x=None, fh=None, type_=None):
    if isinstance(model, (AutoARIMA, ExponentialSmoothing)):
        prediction = model.predict(fh)

    elif isinstance(model, RandomForestRegressor) and type_ == 'univariate':
        prediction = []
        for i in range(x.shape[0] // 12):
            if i != 0:
                x.iloc[12 * i: 12 * (i + 1), -1] = y_pred
            y_pred = model.predict(x[12 * i: 12 * (i + 1)])
            pred = pd.Series(data=y_pred, index=fh[12 * i: 12 * (i + 1)])
            prediction.append(pred)
        prediction = pd.concat(prediction)

    elif isinstance(model, (RandomForestRegressor, SVR)) and type_ == 'multivariate':
        cols_to_use = [col for col in x.columns if 'lag' not in col]
        features = x[cols_to_use]
        lags = x[[col for col in x.columns if 'lag' in col]]
        n_pred = len(fh)
        y_pred = []
        for i in range(n_pred):
            feat = np.concatenate((features.iloc[i, :].values, y_pred[::-1], lags.iloc[0, :].values))[
                   :(lags.shape[1] + features.shape[1])]
            pred = model.predict(feat.reshape(1, -1))
            y_pred.append(pred[0])
        prediction = pd.Series(data=y_pred, index=fh)

    else:
        raise TypeError("Use one of (RandomForestRegressor, AutoARIMA, ExponentialSmoothing) model type. "
                        "If RandomForestRegressor used please specify type_: {univariate, multivariate} ")

    return prediction
