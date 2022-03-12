import warnings

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, \
    mean_absolute_percentage_error
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing

import objectives
import train_pred
from plots import plot_with_zoom

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def read_and_prep_data_univariate(filepath, predicted_column):
    df = pd.read_csv(filepath)
    df.columns = [col.lower() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.index.freq = 'M'
    df = df[[predicted_column]]

    for i in range(1, 13):
        df[f'lag_{i}'] = df[predicted_column].shift(i)

    return df[['lag_12', predicted_column]].dropna()


def main():
    filepath = '../data/wig20_m.csv'
    predicted_column = 'close'
    df = read_and_prep_data_univariate(filepath, predicted_column)
    x_train, y_train = df.iloc[:-12, 0], df.iloc[:-12, 1]
    x_test, y_test = df.iloc[-12:, 0], df.iloc[-12:, 1]

    # define models
    hw_model = train_pred.train_series(ExponentialSmoothing(), objectives.objective_hw, 100, y_train, sp=12)
    arima_model = train_pred.train_series(AutoARIMA(), objectives.objective_arima, 2, y_train, sp=12)
    rf_model = train_pred.train_series(RandomForestRegressor(), objectives.objective_rf, 100, y_train.values,
                                       x=x_train.values.reshape(-1, 1))

    fh = pd.date_range(y_test.index[0], y_test.index[-1], freq='M')
    models = [hw_model, arima_model, rf_model]
    col_names = ['HW', 'ARIMA', 'RF']
    labels = ['actuals', 'Holt-Winters', 'ARIMA', 'Random Forest']
    for model, col_name in zip(models, col_names):
        prediction = train_pred.predict_series(model, x=x_test, fh=fh)
        df.loc[y_test.index, col_name] = prediction

        print('-' * 50, col_name, '-' * 50)
        for error_name in [mean_squared_error, mean_absolute_error, mean_squared_log_error,
                           mean_absolute_percentage_error]:
            error = error_name(y_test, prediction)
            print(col_name, '- error -', error)

    col_names.insert(0, predicted_column)
    plot_with_zoom(df, y_test.index, **dict(zip(col_names, labels)))


if __name__ == '__main__':
    main()
