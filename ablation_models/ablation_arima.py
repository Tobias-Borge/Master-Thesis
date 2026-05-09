import numpy as np

from ARIMA import fit_auto_arima_and_forecast


def forecast_arima_standalone(train_series, test_horizon):
    train_series = np.asarray(train_series, dtype=float).ravel()
    h = int(test_horizon)
    fc, order = fit_auto_arima_and_forecast(train_series, h)
    return np.asarray(fc, dtype=float).ravel(), order

