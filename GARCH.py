
import numpy as np
from arch import arch_model


# Function to fit the best GARCH model with fallback
def _fit_best_garch_with_fallback(series,order=(1, 1),distributions=("normal", "t"),search_orders=((1, 1), (1, 2), (2, 1), (2, 2)),fallback_order=(1, 1),fallback_dist="normal"):

    best_res = None
    best_aic = np.inf
    best_order = None
    best_dist = None

    tried = []
    # Keep deterministic order and remove duplicates while preserving order.
    candidates = []
    for spec in [tuple(order)] + [tuple(o) for o in search_orders]:
        if spec not in candidates:
            candidates.append(spec)
    dists = []
    for d in [fallback_dist] + list(distributions):
        if d not in dists:
            dists.append(d)

    for p, q in candidates:
        for dist in dists:
            tried.append((p, q, dist))
            try:
                model = arch_model(series, vol="Garch", p=int(p), q=int(q), dist=str(dist))
                res = model.fit(disp="off")
                aic = float(getattr(res, "aic", np.inf))
                if np.isfinite(aic) and aic < best_aic:
                    best_aic = aic
                    best_res = res
                    best_order = (int(p), int(q))
                    best_dist = str(dist)
            except Exception:
                continue

    if best_res is not None:
        return best_res, best_order, best_dist, False

    # Last-resort fallback.
    fp, fq = fallback_order
    try:
        model = arch_model(series, vol="Garch", p=int(fp), q=int(fq), dist=str(fallback_dist))
        res = model.fit(disp="off")
        print(
            f"GARCH fit failed for searched candidates {tried}; "
            f"fell back to default GARCH({int(fp)},{int(fq)}) dist='{fallback_dist}'."
        )
        return res, (int(fp), int(fq)), str(fallback_dist), True
    except Exception as ex:
        raise RuntimeError(
            f"GARCH fit failed for all candidates and fallback GARCH({int(fp)},{int(fq)}) "
            f"dist='{fallback_dist}' also failed: {ex}"
        )


# Function to fit a GARCH model to a 1D IMF series and forecast variance.
def garch_forecast(imf, forecast_length=10, order=(1, 1)):
    imf = np.asarray(imf, dtype=float)
    res, _ord, _dist, _fb = _fit_best_garch_with_fallback(imf, order=order)
    fc = res.forecast(horizon=forecast_length)
    var_fc = fc.variance.values[-1]
    return np.asarray(var_fc)

# Function to fit GARCH on train close IMF and return a full volatility feature series:
def fit_garch_volatility_feature(close_train, horizon, order=(1, 1), rescale=100.0):
    close_train = np.asarray(close_train, dtype=float)
    if len(close_train) < 5:
        return np.zeros(len(close_train) + horizon, dtype=float)
    scaled_train = close_train * float(rescale)
    res, selected_order, selected_dist, used_fallback = _fit_best_garch_with_fallback(
        scaled_train, order=order
    )
    if used_fallback:
        print(
            f"GARCH feature path: using fallback order={selected_order}, dist='{selected_dist}'."
        )

    # In-sample conditional volatility on training segment
    vol_train = np.asarray(res.conditional_volatility, dtype=float)
    # Multi-step out-of-sample variance forecast, converted to volatility
    var_fc = np.asarray(res.forecast(horizon=horizon).variance.values[-1], dtype=float)
    vol_fc = np.sqrt(np.maximum(var_fc, 0.0))

    # Undo scaling so feature matches IMF magnitude better.
    vol_full = np.concatenate([vol_train, vol_fc]) / float(rescale)

    # Standardize with train stats only to keep scale stable for LSTM.
    mu = float(np.mean(vol_full[: len(close_train)]))
    sigma = float(np.std(vol_full[: len(close_train)]))
    if sigma == 0.0:
        sigma = 1.0
    return (vol_full - mu) / sigma

# Function to do the strict 1-step walk-forward forecast for the High frequency LSTM.
def hf_walk_forward_forecast(
    model,
    imf_k,
    T_train,
    horizon,
    seq_len,
    close_idx,
    garch_feature_full,
    use_observed_update=True,
    output_horizon=1,
    return_horizon_matrix=False,
    rollout_horizon=None,
):

    history = imf_k[:, :T_train].T.copy()
    preds = []
    out_h = int(max(1, output_horizon))
    horizon_rows = []
    roll_h = int(max(1, rollout_horizon)) if rollout_horizon is not None else out_h
    n_total = imf_k.shape[1]
    # Forecast horizon steps ahead
    for step in range(horizon):
        if history.shape[0] < seq_len:
            break
        # Current time step
        current_t = T_train + step
        # Start and end of the GARCH feature window
        g_start = current_t - seq_len
        g_end = current_t
        # If the window is out of bounds, break
        if g_start < 0 or g_end > len(garch_feature_full):
            break
        # Window of data available up to current step
        window_core = history[-seq_len:, :]
        garch_feat = garch_feature_full[g_start:g_end].reshape(seq_len, 1)
        window = np.concatenate([window_core, garch_feat], axis=1)
        # Forward pass through the LSTM
        cache = model.forward(window)
        h_last = cache[-1][0]
        # Predicted close for this step
        y_vec = (model.Why @ h_last + model.by).flatten()
        if y_vec.size < out_h:
            pad_v = float(y_vec[-1]) if y_vec.size > 0 else 0.0
            y_vec = np.concatenate([y_vec, np.full(out_h - y_vec.size, pad_v, dtype=float)])
        y_hat = float(y_vec[0])
        preds.append(y_hat)
        if roll_h == out_h:
            horizon_rows.append(np.asarray(y_vec[:out_h], dtype=float))
        else:
            # Recursive multi-step rollout from current history (does not use future observations).
            temp_hist = history.copy()
            row = []
            for k in range(roll_h):
                ct = T_train + step + k
                gs = ct - seq_len
                ge = ct
                if gs < 0 or ge > len(garch_feature_full):
                    break
                w_core = temp_hist[-seq_len:, :]
                g_feat = np.asarray(garch_feature_full[gs:ge], dtype=float).reshape(seq_len, 1)
                w = np.concatenate([w_core, g_feat], axis=1)
                c = model.forward(w)
                h_last_k = c[-1][0]
                yk_vec = (model.Why @ h_last_k + model.by).flatten()
                yk = float(yk_vec[0]) if yk_vec.size > 0 else 0.0
                row.append(float(yk))
                next_row_k = temp_hist[-1].copy()
                next_row_k[close_idx] = yk
                temp_hist = np.vstack([temp_hist, next_row_k])
            if len(row) < roll_h:
                pad_v = float(row[-1]) if len(row) > 0 else float(y_hat)
                row = list(row) + [pad_v] * (roll_h - len(row))
            horizon_rows.append(np.asarray(row, dtype=float))

        # Target index for the next step
        target_idx = T_train + step
        if use_observed_update and target_idx < n_total:
            # Walk-forward update with newly observed market data at this timestamp
            next_row = imf_k[:, target_idx].copy()
        else:
            # Fully recursive fallback when no observation update is used
            next_row = history[-1].copy()
            next_row[close_idx] = y_hat

        history = np.vstack([history, next_row])

    if return_horizon_matrix:
        return np.asarray(preds, dtype=float), np.asarray(horizon_rows, dtype=float)
    return np.asarray(preds, dtype=float)

# Function to do the walk-forward LSTM on the close channel with the GARCH feature
def walk_forward_close_garch_lstm(model,close_channel_full,T_train,horizon,seq_len,garch_feature_full,use_observed_update=True,output_horizon=1,return_horizon_matrix=False):
    close_channel_full = np.asarray(close_channel_full, dtype=float).ravel()
    history = close_channel_full[:T_train].reshape(-1, 1).copy()
    preds = []
    out_h = int(max(1, output_horizon))
    horizon_rows = []
    n_total = len(close_channel_full)
    for step in range(int(horizon)):
        if history.shape[0] < int(seq_len):
            break
        current_t = int(T_train) + step
        g_start = current_t - int(seq_len)
        g_end = current_t
        if g_start < 0 or g_end > len(garch_feature_full):
            break
        window_close = history[-int(seq_len) :, :]
        garch_feat = np.asarray(garch_feature_full[g_start:g_end], dtype=float).reshape(int(seq_len), 1)
        window = np.concatenate([window_close, garch_feat], axis=1)
        cache = model.forward(window)
        y_vec = (model.Why @ cache[-1][0] + model.by).flatten()
        if y_vec.size < out_h:
            pad_v = float(y_vec[-1]) if y_vec.size > 0 else 0.0
            y_vec = np.concatenate([y_vec, np.full(out_h - y_vec.size, pad_v, dtype=float)])
        y_hat = float(y_vec[0])
        preds.append(y_hat)
        horizon_rows.append(np.asarray(y_vec[:out_h], dtype=float))
        target_idx = int(T_train) + step
        if use_observed_update and target_idx < n_total:
            next_row = np.array([[float(close_channel_full[target_idx])]], dtype=float)
        else:
            next_row = np.array([[y_hat]], dtype=float)
        history = np.vstack([history, next_row])
    if return_horizon_matrix:
        return np.asarray(preds, dtype=float), np.asarray(horizon_rows, dtype=float)
    return np.asarray(preds, dtype=float)


