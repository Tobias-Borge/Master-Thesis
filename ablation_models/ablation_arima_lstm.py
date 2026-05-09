import numpy as np

from ARIMA import forecast_arima_lstm_low_imf


def forecast_arima_residual_model(train_series,test_series,seq_len,hidden_size,epochs,lr,model_type="LSTM",skip_if_iid=True,forecast_days_ahead=1,residual_target_step=1,output_horizon=1,label="ARIMA-LSTM",):
    out = forecast_arima_lstm_low_imf(
        series_train=np.asarray(train_series, dtype=float).ravel(),
        series_test=np.asarray(test_series, dtype=float).ravel(),
        res_seq_len=int(seq_len),
        res_hidden_size=int(hidden_size),
        res_epochs=int(epochs),
        res_lr=float(lr),
        skip_lstm_if_iid=bool(skip_if_iid),
        res_model_type=str(model_type),
        forecast_days_ahead=int(forecast_days_ahead),
        residual_target_step=int(residual_target_step),
        output_horizon=int(output_horizon),
        imf_label=str(label),
    )
    return out

