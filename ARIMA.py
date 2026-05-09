# ARIMA on MEMD IMFs (clean baseline)

import os
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
try:
    from pmdarima import auto_arima as _pmd_auto_arima
    PMDARIMA_AVAILABLE = True
except Exception:
    _pmd_auto_arima = None
    PMDARIMA_AVAILABLE = False

from LSTM import LSTM
from MLP import MLP

# Function to fit the auto ARIMA model and forecast
def fit_auto_arima_and_forecast(series_train, horizon):
    series_train = np.asarray(series_train, dtype=float)
    if PMDARIMA_AVAILABLE:
        model = _pmd_auto_arima(
            series_train,
            start_p=0,
            start_q=0,
            max_p=3,
            max_q=3,
            d=None,
            seasonal=False,
            information_criterion="aic",
            stepwise=True,
            error_action="ignore",
            suppress_warnings=True,
        )
        best_order = model.order
        fc = model.predict(n_periods=horizon)
        return np.asarray(fc, dtype=float), best_order

    # Fallback: small AIC grid search with statsmodels only
    best_order = None
    best_aic = np.inf
    for p in range(0, 4):
        for q in range(0, 4):
            try:
                res = ARIMA(
                    series_train,
                    order=(p, 0, q),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
                if np.isfinite(res.aic) and res.aic < best_aic:
                    best_aic = float(res.aic)
                    best_order = (p, 0, q)
            except Exception:
                continue

    if best_order is None:
        best_order = (1, 0, 0)
    res = ARIMA(
        series_train,
        order=best_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    fc = res.forecast(steps=horizon)
    return np.asarray(fc, dtype=float), best_order


# Function to do the rolling one step ARIMA forecast
def arima_rolling_one_step_forecast(
    series_train,
    series_test,
    fitted_model,
    use_diff=False,
    use_observed_update=True,
    output_horizon=1,
    return_horizon_matrix=False,
    rollout_horizon=None,
):
    history_raw = list(np.asarray(series_train, dtype=float))
    series_test = np.asarray(series_test, dtype=float)
    preds = []
    out_h = int(max(1, output_horizon))
    horizon_rows = []
    current_model = fitted_model
    roll_h = int(max(1, rollout_horizon)) if rollout_horizon is not None else out_h

    for y_true in series_test:
        history_arr = np.asarray(history_raw, dtype=float)
        y_proc_vec = np.asarray(current_model.forecast(steps=out_h), dtype=float).reshape(-1)
        y_hat_proc = float(y_proc_vec[0])
        if use_diff:
            y_hat = history_arr[-1] + y_hat_proc
            y_obs_proc = float(y_true - history_arr[-1])
            y_raw_vec = history_arr[-1] + np.cumsum(y_proc_vec)
        else:
            y_hat = y_hat_proc
            y_obs_proc = float(y_true)
            y_raw_vec = y_proc_vec.copy()

        preds.append(y_hat)
        if roll_h == out_h:
            horizon_rows.append(y_raw_vec[:out_h])
        else:
            temp_model = current_model
            last_raw = float(history_arr[-1])
            row = []
            for _k in range(roll_h):
                yk_proc = float(np.asarray(temp_model.forecast(steps=1), dtype=float).reshape(-1)[0])
                if use_diff:
                    yk_raw = last_raw + yk_proc
                    last_raw = float(yk_raw)
                else:
                    yk_raw = yk_proc
                row.append(float(yk_raw))
                temp_model = temp_model.append([float(yk_proc)], refit=False)
            horizon_rows.append(np.asarray(row, dtype=float))
        # Walk-forward update
        y_update_proc = y_obs_proc if use_observed_update else float(y_hat_proc)
        current_model = current_model.append([y_update_proc], refit=False)
        history_raw.append(float(y_true))

    if return_horizon_matrix:
        return np.asarray(preds, dtype=float), np.asarray(horizon_rows, dtype=float)
    return np.asarray(preds, dtype=float)


# Function to return the ADF p-value for the stationarity test
def adf_pvalue(series):
    series = np.asarray(series, dtype=float)
    if len(series) < 20:
        return np.nan
    try:
        return float(adfuller(series, autolag="AIC")[1])
    except Exception:
        return np.nan


# Function to preprocess the stationary series
def preprocess_stationary(series_train, alpha=0.05):
    series_train = np.asarray(series_train, dtype=float)
    p_raw = adf_pvalue(series_train)
    use_diff = np.isfinite(p_raw) and p_raw > alpha

    if use_diff:
        series_proc = np.diff(series_train)
        p_proc = adf_pvalue(series_proc)
    else:
        series_proc = series_train.copy()
        p_proc = p_raw

    meta = {
        "use_diff": use_diff,
        "p_raw": p_raw,
        "p_proc": p_proc,
        "last_raw": float(series_train[-1]),
    }
    return series_proc, meta


# Function to invert the preprocessing forecast
def invert_preprocessing_forecast(forecast_proc, meta):
    forecast_proc = np.asarray(forecast_proc, dtype=float)
    if not meta["use_diff"]:
        return forecast_proc
    return meta["last_raw"] + np.cumsum(forecast_proc)


# Function to reconstruct the fitted values to the raw scale
def reconstruct_fitted_to_raw(fitted_proc, raw_train, meta):
    fitted_proc = np.asarray(fitted_proc, dtype=float)
    raw_train = np.asarray(raw_train, dtype=float)
    if not meta["use_diff"]:
        # align length with raw train
        fitted = fitted_proc[-len(raw_train):]
        actual = raw_train[-len(fitted):]
        return fitted, actual

    # First-difference case
    fitted_raw = raw_train[:-1] + fitted_proc
    actual_raw = raw_train[1:]
    return fitted_raw, actual_raw


# Function to test the residual IID
def residual_iid_test_ljungbox(residuals, alpha=0.05, lags=10):
    residuals = np.asarray(residuals, dtype=float)
    if len(residuals) < max(20, lags + 5):
        return False, np.nan
    try:
        lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
        pval = float(lb["lb_pvalue"].iloc[-1])
        return pval > alpha, pval
    except Exception:
        return False, np.nan


# Function to z-score scale the train series
def zscore_scale_train(series):
    series = np.asarray(series, dtype=float)
    mu = float(np.mean(series))
    sigma = float(np.std(series))
    if sigma == 0.0:
        sigma = 1.0
    scaled = (series - mu) / sigma
    return scaled, mu, sigma


# Function to build the residual model
def _build_residual_model(model_type, input_size, hidden_size, learning_rate):
    model_key = str(model_type).upper()
    if model_key == "MLP":
        return MLP(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=1,
            learning_rate=learning_rate,
        )
    return LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=1,
        learning_rate=learning_rate,
    )


# Function to do the rolling residual forecast
def lstm_residual_forecast_rolling(
    residual_train,
    residual_test,
    seq_len=50,
    hidden_size=32,
    epochs=10,
    learning_rate=5e-4,
    model_type="LSTM",
    use_observed_update=True,
    target_step=1,
    output_horizon=1,
    return_horizon_matrix=False,
    rollout_horizon=None,
    model_cache_path=None,
    force_retrain_model=False,
    save_trained_model=True,
):
    residual_train = np.asarray(residual_train, dtype=float)
    residual_test = np.asarray(residual_test, dtype=float)
    if len(residual_train) <= seq_len + 1:
        return np.zeros_like(residual_test, dtype=float)

    # Scale residuals with train stats only
    residual_train_scaled, mu, sigma = zscore_scale_train(residual_train)
    residual_test_scaled = (residual_test - mu) / sigma

    X = []
    y = []
    h = int(max(1, target_step))
    n_pairs = len(residual_train_scaled) - seq_len - h + 1
    for t in range(max(0, n_pairs)):
        X.append(residual_train_scaled[t : t + seq_len].reshape(seq_len, 1))
        y.append([[residual_train_scaled[t + seq_len + h - 1]]])
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    model = None
    trained_now = False
    model_key = str(model_type).upper()
    model_cls = MLP if model_key == "MLP" else LSTM
    if (
        model_cache_path is not None
        and (not force_retrain_model)
        and os.path.exists(model_cache_path)
    ):
        try:
            model = model_cls.load(model_cache_path)
            print(f"Loaded pretrained low-frequency residual model: {model_cache_path}")
        except Exception as ex:
            print(
                f"Pretrained low-frequency residual load failed ({model_cache_path}), retraining: {ex}"
            )
            model = None

    if model is None:
        model = _build_residual_model(
            model_type=model_type,
            input_size=1,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
        )
        for _ in range(epochs):
            for i in range(len(X)):
                model.train_step(X[i], y[i])
        trained_now = True
        if trained_now and model_cache_path is not None and save_trained_model:
            try:
                model.save(model_cache_path)
            except Exception as ex:
                print(f"Low-frequency residual model save skipped ({model_cache_path}): {ex}")

    # Rolling residual forecasts
    history = list(residual_train_scaled)
    preds_scaled = []
    out_h = int(max(1, output_horizon))
    horizon_rows_scaled = []
    roll_h = int(max(1, rollout_horizon)) if rollout_horizon is not None else out_h
    for r_true_scaled in residual_test_scaled:
        window = np.asarray(history[-seq_len:], dtype=float).reshape(seq_len, 1)
        cache = model.forward(window)
        h_last = cache[-1][0]
        r_vec_scaled = (model.Why @ h_last + model.by).flatten()
        if r_vec_scaled.size < out_h:
            pad_v = float(r_vec_scaled[-1]) if r_vec_scaled.size > 0 else 0.0
            r_vec_scaled = np.concatenate([r_vec_scaled, np.full(out_h - r_vec_scaled.size, pad_v, dtype=float)])
        r_hat_scaled = float(r_vec_scaled[0])
        preds_scaled.append(r_hat_scaled)

        if roll_h == out_h:
            horizon_rows_scaled.append(r_vec_scaled[:out_h])
        else:
            temp_hist = list(history)
            row = []
            for _k in range(roll_h):
                w = np.asarray(temp_hist[-seq_len:], dtype=float).reshape(seq_len, 1)
                c = model.forward(w)
                rv = (model.Why @ c[-1][0] + model.by).flatten()
                rk = float(rv[0]) if rv.size > 0 else 0.0
                row.append(float(rk))
                temp_hist.append(float(rk))
            horizon_rows_scaled.append(np.asarray(row, dtype=float))

        history.append(float(r_true_scaled) if use_observed_update else float(r_hat_scaled))

    preds_scaled = np.asarray(preds_scaled, dtype=float)
    preds = preds_scaled * sigma + mu
    if return_horizon_matrix:
        hmat = np.asarray(horizon_rows_scaled, dtype=float) * sigma + mu
        return preds, hmat
    return preds


# Function to forecast the ARIMA-LSTM model
def forecast_arima_lstm_low_imf(series_train,series_test,res_seq_len=50,res_hidden_size=32,res_epochs=10,res_lr=5e-4,skip_lstm_if_iid=True,ljung_alpha=0.05,ljung_lags=10,adf_alpha=0.05,imf_label="IMF",res_model_type="LSTM",forecast_days_ahead=1,residual_target_step=1,output_horizon=1,residual_model_cache_path=None,force_retrain_model=False,save_trained_model=True):
    series_train = np.asarray(series_train, dtype=float)
    series_test = np.asarray(series_test, dtype=float)
    horizon = len(series_test)
    use_observed_update = True
    out_h = int(max(1, output_horizon))
    roll_h = int(max(1, forecast_days_ahead))

    try:
        series_proc, pre_meta = preprocess_stationary(series_train, alpha=adf_alpha)
        use_diff = pre_meta["use_diff"]
        p_raw = pre_meta["p_raw"]
        p_proc = pre_meta["p_proc"]
        print(
            f"{imf_label}: ADF p(raw)={p_raw:.4f}, "
            f"preprocess={'diff1' if use_diff else 'none'}, ADF p(proc)={p_proc:.4f}"
        )

        _, order = fit_auto_arima_and_forecast(series_proc, horizon=1)
        sm_model = ARIMA(series_proc, order=order).fit()

        fc, arima_hmat = arima_rolling_one_step_forecast(
            series_train=series_train,
            series_test=series_test,
            fitted_model=sm_model,
            use_diff=use_diff,
            use_observed_update=use_observed_update,
            output_horizon=out_h,
            return_horizon_matrix=True,
            rollout_horizon=roll_h,
        )

        fitted_proc = np.asarray(sm_model.fittedvalues, dtype=float)
        fitted_raw, actual_aligned = reconstruct_fitted_to_raw(
            fitted_proc, series_train, pre_meta
        )
        train_resid = np.asarray(actual_aligned - fitted_raw, dtype=float)
        test_resid = np.asarray(series_test - fc, dtype=float)

        iid_like, lb_pvalue = residual_iid_test_ljungbox(
            train_resid, alpha=ljung_alpha, lags=ljung_lags
        )

        ss_res = np.sum((series_test - fc) ** 2)
        ss_tot = np.sum((series_test - series_test.mean()) ** 2) + 1e-12
        r2_arima = 1.0 - ss_res / ss_tot
        print(f"{imf_label}: auto-ARIMA order={order}, rolling 1-step test R^2={r2_arima:.4f}")
        res_model_name = str(res_model_type).upper()
        print(
            f"{imf_label}: Ljung-Box p={lb_pvalue:.4f} -> "
            f"{'IID-like (skip residual model)' if iid_like else f'structure (use {res_model_name})'}"
        )

        if skip_lstm_if_iid and iid_like:
            resid_fc = np.zeros(horizon, dtype=float)
        else:
            resid_fc, resid_hmat = lstm_residual_forecast_rolling(
                residual_train=train_resid,
                residual_test=test_resid,
                seq_len=res_seq_len,
                hidden_size=res_hidden_size,
                epochs=res_epochs,
                learning_rate=res_lr,
                model_type=res_model_type,
                use_observed_update=use_observed_update,
                target_step=residual_target_step,
                output_horizon=out_h,
                return_horizon_matrix=True,
                rollout_horizon=roll_h,
                model_cache_path=residual_model_cache_path,
                force_retrain_model=force_retrain_model,
                save_trained_model=save_trained_model,
            )
        if skip_lstm_if_iid and iid_like:
            resid_hmat = np.zeros((horizon, roll_h), dtype=float)

        combined_fc = fc + resid_fc
        # Align horizon-matrix widths for combination
        target_w = max(int(roll_h), int(out_h))
        if arima_hmat.ndim == 1:
            arima_hmat = arima_hmat.reshape(-1, 1)
        if resid_hmat.ndim == 1:
            resid_hmat = resid_hmat.reshape(-1, 1)
        if arima_hmat.shape[1] < target_w:
            last_col = (
                arima_hmat[:, [-1]]
                if arima_hmat.shape[1] > 0
                else np.zeros((arima_hmat.shape[0], 1), dtype=float)
            )
            arima_hmat = np.concatenate(
                [arima_hmat, np.repeat(last_col, target_w - arima_hmat.shape[1], axis=1)],
                axis=1,
            )
        if resid_hmat.shape[1] < target_w:
            last_col = (
                resid_hmat[:, [-1]]
                if resid_hmat.shape[1] > 0
                else np.zeros((resid_hmat.shape[0], 1), dtype=float)
            )
            resid_hmat = np.concatenate(
                [resid_hmat, np.repeat(last_col, target_w - resid_hmat.shape[1], axis=1)],
                axis=1,
            )
        combined_hmat = np.asarray(arima_hmat[:, :target_w], dtype=float) + np.asarray(
            resid_hmat[:, :target_w], dtype=float
        )
        ss_res_c = np.sum((series_test - combined_fc) ** 2)
        r2_comb = 1.0 - ss_res_c / ss_tot
        print(f"{imf_label}: ARIMA+{res_model_name} 1-step walk-forward test R^2={r2_comb:.4f}")

        return {
            "combined": combined_fc,
            "combined_hmat": combined_hmat,
            "arima_only": fc,
            "resid_lstm": resid_fc,
            "order": order,
            "iid_like": iid_like,
            "lb_pvalue": lb_pvalue,
            "r2_arima": r2_arima,
            "r2_combined": r2_comb,
            "train_resid": train_resid,
            "test_resid": test_resid,
        }
    except Exception as e:
        print(f"{imf_label}: ARIMA-LSTM failed -> {e}")
        z = np.zeros(horizon, dtype=float)
        return {
            "combined": z,
            "combined_hmat": np.zeros((horizon, int(max(1, output_horizon))), dtype=float),
            "arima_only": z,
            "resid_lstm": z,
            "order": None,
            "iid_like": True,
            "lb_pvalue": np.nan,
            "r2_arima": np.nan,
            "r2_combined": np.nan,
            "train_resid": z,
            "test_resid": z,
        }

