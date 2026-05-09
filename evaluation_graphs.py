##### The evaluation plots that we need to compare the hybrids #####

import re
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
except Exception:
    plot_acf = None
    plot_pacf = None

# assigning colors to the strategies
THESIS_COLOR_MPC = "#1f77b4"
THESIS_COLOR_MPC_TERMINAL = "#17becf"
THESIS_COLOR_MPC_SHARPE = "#8c564b"
THESIS_COLOR_EW = "#2ca02c"
THESIS_COLOR_BH = "#ff7f0e"
THESIS_COLOR_MARKOWITZ = "#9467bd"
THESIS_COLOR_MEAN_MARKER = "#d62728"

# Function to get the color for the strategy label
def thesis_color_for_strategy_label(label):
    s = str(label).lower()
    if "terminal wealth" in s:
        return THESIS_COLOR_MPC_TERMINAL
    if "sharpe-like" in s or "sharpe like" in s or "sharpe_like" in s:
        return THESIS_COLOR_MPC_SHARPE
    if "mean-variance" in s or "mean_variance" in s:
        return THESIS_COLOR_MPC
    if "mpc" in s:
        return THESIS_COLOR_MPC
    if "equal weight" in s or s.startswith("ew") or " ew" in s:
        return THESIS_COLOR_EW
    if "buy" in s and "hold" in s:
        return THESIS_COLOR_BH
    if "markowitz" in s or "mvo" in s:
        return THESIS_COLOR_MARKOWITZ
    return None

# Function to convert the array to a 1D array
def _as_1d(x, name="array"):
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr

# Function to check if the two arrays have the same length
def _check_same_len(a, b, a_name="a", b_name="b"):
    if len(a) != len(b):
        raise ValueError(f"{a_name} and {b_name} must have same length. Got {len(a)} and {len(b)}.")


# Function to compute the forecast metrics
def _forecast_metrics(forecast, real_value):
    y_hat = _as_1d(forecast, "forecast")
    y = _as_1d(real_value, "real_value")
    _check_same_len(y_hat, y, "forecast", "real_value")

    err = y_hat - y
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)
    mae = float(np.mean(np.abs(err)))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


# Function to plot the forecast error
def _forecast_error_plot(forecast, real_value, model_name="Model", save_path=None, show=True):
    y_hat = _as_1d(forecast, "forecast")
    y = _as_1d(real_value, "real_value")
    _check_same_len(y_hat, y, "forecast", "real_value")
    e = y_hat - y
    m = _forecast_metrics(y_hat, y)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(
        y,
        label="Actual",
        color="black",
        linewidth=1.0,
        linestyle="--",
        alpha=0.95,
    )
    axes[0].plot(y_hat, label="Forecast", color="tab:blue", alpha=0.8)
    axes[0].set_title(
        f"{model_name} forecast vs actual | "
        f"R²={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MSE={m['MSE']:.4e}"
    )
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(e, color="tab:red", label="Tracking error (forecast - actual)")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Error")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return m


# Function to plot the forecasting error for the LSTM Hybrid
def forecasting_error_LSTM_Hybrid(forecast, real_value, save_path=None, show=True):
    return _forecast_error_plot(
        forecast=forecast,
        real_value=real_value,
        model_name="LSTM Hybrid",
        save_path=save_path,
        show=show,
    )


# Function to plot the cumulative portfolio return
def cumulative_portfolio_return(cumulative_returns, hybrids=None, save_path=None, show=True):
    if isinstance(cumulative_returns, dict):
        series_dict = {k: _as_1d(v, k) for k, v in cumulative_returns.items()}
    else:
        arr = np.asarray(cumulative_returns, dtype=float)
        if arr.ndim != 2:
            raise ValueError("cumulative_returns must be dict or 2D array.")
        if hybrids is None:
            hybrids = [f"Model {i+1}" for i in range(arr.shape[0])]
        series_dict = {name: arr[i] for i, name in enumerate(hybrids)}

    fig, ax = plt.subplots(figsize=(12, 4))
    for name, curve in series_dict.items():
        c = thesis_color_for_strategy_label(name)
        ax.plot(curve, label=name, color=c, linewidth=1.3, alpha=0.9)
    ax.set_title("Cumulative portfolio return")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative value")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# Function to plot the allocation plot
def allocation_plot(stocks_shares, ETF_shares=None, labels=None, save_path=None, show=True, x=None):
    stocks = np.asarray(stocks_shares, dtype=float)
    if stocks.ndim == 1:
        stocks = stocks.reshape(-1, 1)
    if stocks.ndim != 2:
        raise ValueError("stocks_shares must be 1D or 2D array.")

    comps = [stocks]
    comp_labels = []
    if labels is None:
        comp_labels = [f"Stock {i+1}" for i in range(stocks.shape[1])]
    else:
        comp_labels = list(labels)

    if ETF_shares is not None:
        etf = np.asarray(ETF_shares, dtype=float).reshape(-1, 1)
        if etf.shape[0] != stocks.shape[0]:
            raise ValueError("ETF_shares length must match number of time steps in stocks_shares.")
        comps.append(etf)
        comp_labels.append("ETF")

    all_alloc = np.concatenate(comps, axis=1)
    n_steps = all_alloc.shape[0]
    if x is None:
        x_plot = np.arange(n_steps)
    else:
        x_plot = np.asarray(x)
        if len(x_plot) != n_steps:
            raise ValueError("x must have same length as time dimension of stocks_shares.")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stackplot(x_plot, all_alloc.T, labels=comp_labels, alpha=0.8)
    ax.set_title("Portfolio allocation over time")
    ax.set_xlabel("Time step" if x is None else "Date")
    ax.set_ylabel("Weight share")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# Function to plot the allocation concentration plot
def allocation_concentration_plot(weights,labels=None,x=None,save_path=None,show=True,):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2:
        raise ValueError("weights must be 2D (time_steps, n_assets).")
    n_steps, n_assets = w.shape
    if labels is None:
        labels = [f"Asset {i+1}" for i in range(n_assets)]
    else:
        labels = list(labels)
        if len(labels) != n_assets:
            raise ValueError("labels length must match number of assets.")

    if x is None:
        x_plot = np.arange(n_steps)
    else:
        x_plot = np.asarray(x)
        if len(x_plot) != n_steps:
            raise ValueError("x must match number of time steps.")

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.get_cmap("tab20")
    for j in range(n_assets):
        c = cmap((j % 20) / 19.0 if n_assets > 1 else 0.0)
        ax.plot(x_plot, w[:, j], label=labels[j], color=c, linewidth=1.0, alpha=0.9)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Portfolio weight (share)")
    ax.set_xlabel("Date" if x is not None else "Time step")
    ax.set_title("Holdings by ticker over time")
    ax.grid(alpha=0.3)
    ncol = 4 if n_assets > 8 else min(3, max(1, n_assets))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, fontsize=8)
    bottom_pad = 0.28 if n_assets > 12 else (0.22 if n_assets > 6 else 0.16)
    fig.tight_layout(rect=(0.0, bottom_pad, 1.0, 1.0))
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# Function to plot the allocation stackplot top k other
def allocation_stackplot_top_k_other(weights,labels=None,top_k=5,x=None,save_path=None,show=True,):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2:
        raise ValueError("weights must be 2D (time_steps, n_assets).")
    n_steps, n_assets = w.shape
    if labels is None:
        labels = [f"Asset {i+1}" for i in range(n_assets)]
    else:
        labels = list(labels)
        if len(labels) != n_assets:
            raise ValueError("labels length must match number of assets.")

    if x is None:
        x_plot = np.arange(n_steps)
    else:
        x_plot = np.asarray(x)
        if len(x_plot) != n_steps:
            raise ValueError("x must match number of time steps.")

    k = min(int(top_k), n_assets)
    mean_w = np.mean(w, axis=0)
    top_idx = np.argsort(mean_w)[-k:][::-1]
    w_top = w[:, top_idx]
    other = np.clip(1.0 - np.sum(w_top, axis=1), 0.0, 1.0)
    stack_layers = [w_top[:, j] for j in range(k)]
    stack_labels = [labels[i] for i in top_idx]
    if k < n_assets:
        stack_layers.append(other)
        stack_labels.append("Other")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stackplot(x_plot, stack_layers, labels=stack_labels, alpha=0.85)
    ax.set_title(f"Portfolio weights (top {k} by mean weight + Other)")
    ax.set_ylabel("Weight share")
    ax.set_xlabel("Date" if x is not None else "Time step")
    ax.legend(loc="upper left", ncol=min(3, len(stack_labels)))
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# Function to compute the forecast error metrics
def compute_forecast_error_metrics(actual, predicted):
    y = _as_1d(actual, "actual")
    y_hat = _as_1d(predicted, "predicted")
    _check_same_len(y, y_hat, "actual", "predicted")
    err = y_hat - y
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / (np.abs(y) + 1e-12)) * 100.0)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


# Function to plot the RQ1 accuracy timeseries
def rq1_accuracy_timeseries(actual,forecasts_by_name,x=None,title="Actual vs predicted (test)",ylabel="Close",save_path=None,show=True):
    y = _as_1d(actual, "actual")
    if x is None:
        x_plot = np.arange(len(y))
    else:
        x_plot = np.asarray(x)
        if len(x_plot) != len(y):
            raise ValueError("x must match length of actual.")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(
        x_plot,
        y,
        color="black",
        linewidth=1.0,
        linestyle="--",
        alpha=0.95,
        label="Actual",
        zorder=6,
    )
    cmap = plt.get_cmap("tab10")
    i = 0
    for name, series in forecasts_by_name.items():
        if series is None:
            continue
        v = np.asarray(series, dtype=float).reshape(-1)
        if len(v) != len(y):
            continue
        if not np.all(np.isfinite(v)):
            continue
        ax.plot(x_plot, v, label=name, color=cmap(i % 10), alpha=0.85, linewidth=1.2)
        i += 1
    ax.set_title(title)
    ax.set_xlabel("Date" if x is not None else "Time step")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# Function to plot the RQ1 prediction scatter
def rq1_prediction_scatter(actual,predicted,model_name="Model",save_path=None,show=True):
    y = _as_1d(actual, "actual")
    y_hat = _as_1d(predicted, "predicted")
    _check_same_len(y, y_hat, "actual", "predicted")
    m = compute_forecast_error_metrics(y, y_hat)

    lo = float(min(y.min(), y_hat.min()))
    hi = float(max(y.max(), y_hat.max()))
    pad = (hi - lo) * 0.05 + 1e-9
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y, y_hat, s=12, alpha=0.45, edgecolors="none", c="tab:blue")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="Perfect (y = x)")
    # OLS: predicted ~ a + b * actual
    b, a = np.polyfit(y, y_hat, 1)
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, a + b * xs, color="tab:orange", linewidth=1.5, label=f"OLS fit (slope={b:.3f})")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model_name}: predicted vs actual")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return m


def rq1_metric_table(metric_by_model,title="Forecast accuracy metrics table",save_path=None,show=True,):
    if not isinstance(metric_by_model, dict) or len(metric_by_model) == 0:
        raise ValueError("metric_by_model must be a non-empty dict.")

    models = list(metric_by_model.keys())
    rows = []
    for m in models:
        d = metric_by_model[m]
        rows.append(
            [
                f"{float(d['MSE']):.6g}",
                f"{float(d['RMSE']):.6g}",
                f"{float(d['MAE']):.6g}",
                f"{float(d['MAPE']):.3f}",
                f"{float(d['R2']):.4f}",
            ]
        )

    col_labels = ["MSE", "RMSE", "MAE", "MAPE (%)", "R2"]
    fig_h = max(4.0, 0.5 * len(models) + 2.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        rowLabels=models,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.3)
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def rq1_ticker_metric_table(metric_by_ticker,title="Per-ticker forecast accuracy (Hybrid)",save_path=None,show=True):
    if not isinstance(metric_by_ticker, dict) or len(metric_by_ticker) == 0:
        raise ValueError("metric_by_ticker must be a non-empty dict.")

    tickers = list(metric_by_ticker.keys())
    rows = []
    for tkr in tickers:
        d = metric_by_ticker[tkr]
        rows.append(
            [
                f"{float(d['MSE']):.6g}",
                f"{float(d['RMSE']):.6g}",
                f"{float(d['MAE']):.6g}",
                f"{float(d['MAPE']):.3f}",
                f"{float(d['R2']):.4f}",
            ]
        )

    col_labels = ["MSE", "RMSE", "MAE", "MAPE (%)", "R2"]
    fig_h = max(4.0, 0.42 * len(tickers) + 2.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        rowLabels=tickers,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.25)
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def rq1_ablation_parameter_table(model_params_by_name,title="Ablation study parameter settings",save_path=None,show=True,):
    if not isinstance(model_params_by_name, dict) or len(model_params_by_name) == 0:
        raise ValueError("model_params_by_name must be a non-empty dict.")

    rows = []
    for model_name, params in model_params_by_name.items():
        if isinstance(params, dict):
            txt = "\n".join([f"{k}={v}" for k, v in params.items()])
        else:
            txt = str(params)
        rows.append([str(model_name), txt])

    fig_h = max(4.5, 0.55 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Model", "Parameter settings"],
        loc="center",
        cellLoc="left",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.45)
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _loss_series(actual, predicted, loss="mse"):
    y = _as_1d(actual, "actual")
    y_hat = _as_1d(predicted, "predicted")
    _check_same_len(y, y_hat, "actual", "predicted")
    e = y_hat - y
    key = str(loss).lower()
    if key == "mse":
        return e ** 2
    if key == "mae":
        return np.abs(e)
    raise ValueError("loss must be 'mse' or 'mae'.")


def diebold_mariano_test(actual, pred_ref, pred_cmp, loss="mse", max_lag=None):
    l_ref = _loss_series(actual, pred_ref, loss=loss)
    l_cmp = _loss_series(actual, pred_cmp, loss=loss)
    d = np.asarray(l_cmp - l_ref, dtype=float).ravel()
    T = int(d.size)
    if T < 5:
        return {"dm_stat": np.nan, "p_two_sided": np.nan, "p_ref_better": np.nan, "mean_diff": np.nan}

    if max_lag is None:
        max_lag = int(max(1, round(T ** (1.0 / 3.0))))
    max_lag = int(max(0, min(max_lag, T - 1)))

    d0 = d - np.mean(d)
    gamma0 = float(np.dot(d0, d0) / T)
    lr_var = gamma0
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(d0[lag:], d0[:-lag]) / T)
        w = 1.0 - lag / (max_lag + 1.0)
        lr_var += 2.0 * w * cov
    lr_var = max(lr_var, 1e-12)
    se_mean = math.sqrt(lr_var / T)
    mean_diff = float(np.mean(d))
    dm_stat = float(mean_diff / se_mean)
    p_two = float(math.erfc(abs(dm_stat) / math.sqrt(2.0)))
    # one-sided: H1 reference better => mean_diff > 0
    phi = 0.5 * (1.0 + math.erf(dm_stat / math.sqrt(2.0)))
    p_ref_better = float(max(0.0, min(1.0, 1.0 - phi)))
    return {
        "dm_stat": dm_stat,
        "p_two_sided": p_two,
        "p_ref_better": p_ref_better,
        "mean_diff": mean_diff,
        "n": T,
        "max_lag": max_lag,
    }


def superior_predictive_ability_test(
    actual,
    pred_ref,
    preds_comp_by_name,
    loss="mse",
    n_boot=500,
    block_len=10,
    random_seed=42,
):
    if not isinstance(preds_comp_by_name, dict) or len(preds_comp_by_name) == 0:
        return {"spa_stat": np.nan, "p_value": np.nan, "n_models": 0, "n_obs": 0}
    l_ref = _loss_series(actual, pred_ref, loss=loss)
    comps = []
    for _, pred_i in preds_comp_by_name.items():
        li = _loss_series(actual, pred_i, loss=loss)
        comps.append(li - l_ref)
    D = np.asarray(comps, dtype=float)  # (m, T)
    m, T = D.shape
    if T < 5:
        return {"spa_stat": np.nan, "p_value": np.nan, "n_models": m, "n_obs": T}

    means = np.mean(D, axis=1)
    spa_stat = float(np.sqrt(T) * np.max(means))

    # Center under null and use circular block bootstrap for dependence.
    Dc = D - means[:, None]
    rng = np.random.default_rng(int(random_seed))
    L = int(max(1, min(block_len, T)))
    boot_stats = np.zeros(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        idx = np.empty(T, dtype=int)
        pos = 0
        while pos < T:
            s = int(rng.integers(0, T))
            take = min(L, T - pos)
            idx[pos : pos + take] = (s + np.arange(take)) % T
            pos += take
        Db = Dc[:, idx]
        mb = np.mean(Db, axis=1)
        boot_stats[b] = float(np.sqrt(T) * np.max(mb))
    p_val = float(np.mean(boot_stats >= spa_stat))
    return {"spa_stat": spa_stat, "p_value": p_val, "n_models": m, "n_obs": T}


def rq1_significance_table(dm_rows, spa_rows, title="Forecast significance tests (DM + SPA)", save_path=None, show=True):
    rows = []
    for r in dm_rows:
        rows.append(
            [
                "DM",
                str(r.get("loss", "")),
                str(r.get("model", "")),
                f"{float(r.get('mean_diff', np.nan)):.4g}",
                f"{float(r.get('dm_stat', np.nan)):.4f}",
                f"{float(r.get('p_ref_better', np.nan)):.4g}",
            ]
        )
    for r in spa_rows:
        rows.append(
            [
                "SPA",
                str(r.get("loss", "")),
                f"{int(r.get('n_models', 0))} benchmarks",
                "",
                f"{float(r.get('spa_stat', np.nan)):.4f}",
                f"{float(r.get('p_value', np.nan)):.4g}",
            ]
        )
    if len(rows) == 0:
        return
    fig_h = max(4.8, 0.4 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Test", "Loss", "Compared model(s)", "Mean diff", "Statistic", "p-value"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.8)
    tbl.scale(1.0, 1.25)
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# Function to plot the RQ2 MEMD decomposition
def rq2_memd_decomposition(time_index,original_series,imfs,residue,channel_label="Close (normalized)",imfs_channel_idx=0,save_path=None,show=True,):
    orig = np.asarray(original_series, dtype=float).reshape(-1)
    res = np.asarray(residue, dtype=float).reshape(-1)
    im = np.asarray(imfs, dtype=float)
    if im.ndim == 3:
        im = im[:, int(imfs_channel_idx), :]
    if im.ndim != 2:
        raise ValueError("imfs must be 2D (n_imfs, n_samples) after channel selection.")
    n_imfs, T = im.shape
    if len(orig) != T or len(res) != T:
        raise ValueError("original_series, residue, and IMF width must share length T.")

    if time_index is None:
        x_plot = np.arange(T)
        xlab = "Time step"
    else:
        x_plot = np.asarray(time_index)
        if len(x_plot) != T:
            raise ValueError("time_index must match series length.")
        xlab = "Date"

    n_panels = 1 + n_imfs + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.0 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    axes[0].plot(x_plot, orig, color="black", linewidth=1.2)
    axes[0].set_title(f"Original {channel_label}")
    axes[0].grid(alpha=0.3)
    for i in range(n_imfs):
        ax = axes[1 + i]
        ax.plot(x_plot, im[i], linewidth=1.0, color="tab:blue")
        ax.set_title(f"IMF {i + 1}")
        ax.grid(alpha=0.3)
    axes[-1].plot(x_plot, res, color="tab:red", linewidth=1.2)
    axes[-1].set_title("Residue (monotonic trend)")
    axes[-1].set_xlabel(xlab)
    fig.suptitle("MEMD decomposition (divide-and-conquer)", fontsize=13, y=1.01)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# Function to plot the RQ2 ablation sequential heatmap
def rq2_ablation_sequential_heatmap(models_ordered,metrics_by_model,title="Marginal contribution (sequential vs previous model)",save_path=None,show=True,):
    order = [m for m in models_ordered if m in metrics_by_model]
    if len(order) < 2:
        raise ValueError("Need at least two model names present in metrics_by_model.")

    R, C = len(order) - 1, 4
    M = np.zeros((R, C), dtype=float)
    row_labels = []
    col_labels = ["ΔRMSE %", "ΔMAE %", "ΔMAPE %", "Δ R²"]
    for i in range(1, len(order)):
        a = metrics_by_model[order[i - 1]]
        b = metrics_by_model[order[i]]
        M[i - 1, 0] = 100.0 * (float(a["RMSE"]) - float(b["RMSE"])) / (float(a["RMSE"]) + 1e-12)
        M[i - 1, 1] = 100.0 * (float(a["MAE"]) - float(b["MAE"])) / (float(a["MAE"]) + 1e-12)
        M[i - 1, 2] = 100.0 * (float(a["MAPE"]) - float(b["MAPE"])) / (float(a["MAPE"]) + 1e-12)
        M[i - 1, 3] = float(b["R2"]) - float(a["R2"])
        row_labels.append(f"{order[i - 1][:22]}\n→ {order[i][:22]}")

    M_norm = np.zeros_like(M, dtype=float)
    for c in range(C):
        col = M[:, c]
        s = float(np.max(np.abs(col)) + 1e-12)
        M_norm[:, c] = col / s

    fig, ax = plt.subplots(figsize=(9.0, max(3.5, 1.1 * R)))
    im = ax.imshow(M_norm, aspect="auto", cmap="RdYlGn", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(C))
    ax.set_xticklabels(col_labels, rotation=15, ha="right")
    ax.set_yticks(np.arange(R))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title + "\n(green = improvement; color normalized per column; text = raw value)")
    for r in range(R):
        for c in range(C):
            if c < 3:
                txt = f"{M[r, c]:.1f}%"
            else:
                txt = f"{M[r, c]:.4f}"
            ax.text(c, r, txt, ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalized gain (−1…1 per column)")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# Function to plot the RQ2 residual ACF / PACF
def rq2_residual_acf_pacf(residual_by_stage,nlags=40,title="Residual ACF / PACF",save_path=None,show=True,):
    if plot_acf is None or plot_pacf is None:
        raise RuntimeError("statsmodels tsaplots (plot_acf, plot_pacf) required for rq2_residual_acf_pacf.")

    names = list(residual_by_stage.keys())
    if len(names) == 0:
        raise ValueError("residual_by_stage must be non-empty.")

    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 5.5), squeeze=False)
    for j, name in enumerate(names):
        x = np.asarray(residual_by_stage[name], dtype=float).ravel()
        x = x[np.isfinite(x)]
        if len(x) < 8:
            axes[0, j].set_title(f"{name[:35]}\n(too few points)")
            axes[0, j].text(0.5, 0.5, "n<8", ha="center", va="center", transform=axes[0, j].transAxes)
            axes[1, j].set_visible(False)
            continue
        use_lags = min(int(nlags), max(5, len(x) // 4))
        plot_acf(x, ax=axes[0, j], lags=use_lags, title=f"ACF\n{name[:40]}")
        try:
            plot_pacf(x, ax=axes[1, j], lags=use_lags, method="ywm")
        except Exception:
            plot_pacf(x, ax=axes[1, j], lags=use_lags, method="ols")
        axes[1, j].set_title("PACF")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# Function to compute the Markowitz long-only max Sharpe weights
def markowitz_longonly_max_sharpe_weights(mu, Sigma, ridge_frac=1e-4):
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)
    if Sigma.shape != (n, n):
        raise ValueError("Sigma must be square and match mu length.")
    tr = float(np.trace(Sigma))
    Sigma = Sigma + (ridge_frac * tr / max(n, 1)) * np.eye(n)

    def neg_sharpe(wv):
        wv = np.clip(wv, 0.0, 1.0)
        s = float(np.sum(wv))
        if s < 1e-15:
            return 1e9
        wv = wv / s
        mp = float(wv @ mu)
        v = float(wv @ Sigma @ wv) + 1e-12
        return -mp / np.sqrt(v)

    x0 = np.ones(n, dtype=float) / n
    bounds = [(0.0, 1.0)] * n
    cons = ({"type": "eq", "fun": lambda wv: float(np.sum(wv) - 1.0)},)
    res = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    w = np.clip(np.asarray(res.x, dtype=float), 0.0, 1.0)
    s = float(np.sum(w))
    if s < 1e-15:
        return np.ones(n, dtype=float) / n
    return w / s


# Function to plot the RQ3 cumulative wealth curves
def rq3_cumulative_wealth_curves(curves_by_name,x=None,title="Cumulative wealth (net portfolio value)",ylabel="Portfolio value ($)",save_path=None,show=True):
    if not isinstance(curves_by_name, dict) or len(curves_by_name) == 0:
        raise ValueError("curves_by_name must be a non-empty dict.")
    series = {k: _as_1d(v, k) for k, v in curves_by_name.items()}
    L = len(next(iter(series.values())))
    for k, v in series.items():
        if len(v) != L:
            raise ValueError(f"All curves must have same length; mismatch at {k}.")

    if x is None:
        x_plot = np.arange(L)
        xlab = "Time step"
    else:
        x_plot = np.asarray(x)
        if len(x_plot) != L:
            raise ValueError("x must match curve length.")
        xlab = "Date"

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for name, curve in series.items():
        c = thesis_color_for_strategy_label(name)
        ax.plot(x_plot, curve, label=name, color=c, linewidth=1.3, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# Function to plot the RQ3 bootstrap terminal profit histogram
def rq3_bootstrap_terminal_profit_histogram(step_returns_by_strategy,initial_cash=1000.0,n_sims=500,random_seed=0,title="Bootstrap distribution of terminal profit",save_path=None,show=True):
    if not isinstance(step_returns_by_strategy, dict) or len(step_returns_by_strategy) == 0:
        raise ValueError("step_returns_by_strategy must be a non-empty dict.")

    rng = np.random.default_rng(int(random_seed))
    T = len(next(iter(step_returns_by_strategy.values())))
    for k, v in step_returns_by_strategy.items():
        if len(np.asarray(v).ravel()) != T:
            raise ValueError(f"All step-return series must have length {T}; got {k}.")

    n_strat = len(step_returns_by_strategy)
    names = list(step_returns_by_strategy.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    for i, name in enumerate(names):
        if i >= len(axes):
            break
        ax = axes[i]
        r = np.asarray(step_returns_by_strategy[name], dtype=float).ravel()
        finals = np.zeros(int(n_sims), dtype=float)
        for b in range(int(n_sims)):
            idx = rng.integers(0, T, size=T)
            step = r[idx]
            w = float(initial_cash) * float(np.prod(1.0 + step))
            finals[b] = w - float(initial_cash)
        face = thesis_color_for_strategy_label(name) or THESIS_COLOR_MPC
        ax.hist(finals, bins=35, color=face, alpha=0.75, edgecolor="white")
        ax.axvline(
            float(np.mean(finals)),
            color=THESIS_COLOR_MEAN_MARKER,
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {np.mean(finals):.1f}",
        )
        std = float(np.std(finals))
        ax.set_title(f"{name}\nmean={np.mean(finals):.1f}, std={std:.1f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Terminal profit ($)")
        ax.set_ylabel("Count")
    for j in range(len(names), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# Function to compute the block bootstrap return paths
def block_bootstrap_return_paths(real_ret, exp_ret, rng, block_len, n_steps=None):
    real_ret = np.asarray(real_ret, dtype=float)
    exp_ret = np.asarray(exp_ret, dtype=float)
    if real_ret.shape != exp_ret.shape:
        raise ValueError("real_ret and exp_ret must have the same shape.")
    if real_ret.ndim != 2:
        raise ValueError("Returns must be 2D (time, n_assets).")
    T, _n = real_ret.shape
    if T < 1:
        raise ValueError("Need at least one return row.")
    if n_steps is None:
        n_steps = T
    block_len = int(max(1, min(int(block_len), T)))
    out_r = np.zeros((int(n_steps), _n), dtype=float)
    out_e = np.zeros((int(n_steps), _n), dtype=float)
    pos = 0
    while pos < int(n_steps):
        if T <= block_len:
            start = 0
            seg_len = T
        else:
            start = int(rng.integers(0, T - block_len + 1))
            seg_len = block_len
        need = int(n_steps) - pos
        take = min(seg_len, need)
        out_r[pos : pos + take] = real_ret[start : start + take]
        out_e[pos : pos + take] = exp_ret[start : start + take]
        pos += take
    return out_r, out_e

# Function to compute the block bootstrap horizon return paths
def block_bootstrap_horizon_return_paths(real_ret, exp_ret_horizon, rng, block_len, n_steps=None):
    real_ret = np.asarray(real_ret, dtype=float)
    exp_ret_horizon = np.asarray(exp_ret_horizon, dtype=float)
    if real_ret.ndim != 2:
        raise ValueError("real_ret must be 2D (time, n_assets).")
    if exp_ret_horizon.ndim != 3:
        raise ValueError("exp_ret_horizon must be 3D (time, horizon, n_assets).")
    T, n = real_ret.shape
    if exp_ret_horizon.shape[0] != T or exp_ret_horizon.shape[2] != n:
        raise ValueError("exp_ret_horizon must align with real_ret on time and assets.")
    if T < 1:
        raise ValueError("Need at least one return row.")
    if n_steps is None:
        n_steps = T
    block_len = int(max(1, min(int(block_len), T)))
    H = int(exp_ret_horizon.shape[1])
    out_r = np.zeros((int(n_steps), n), dtype=float)
    out_eh = np.zeros((int(n_steps), H, n), dtype=float)
    pos = 0
    while pos < int(n_steps):
        if T <= block_len:
            start = 0
            seg_len = T
        else:
            start = int(rng.integers(0, T - block_len + 1))
            seg_len = block_len
        need = int(n_steps) - pos
        take = min(seg_len, need)
        out_r[pos : pos + take] = real_ret[start : start + take]
        out_eh[pos : pos + take] = exp_ret_horizon[start : start + take]
        pos += take
    return out_r, out_eh


# Function to compute the annualized Sharpe step returns
def annualized_sharpe_step_returns(step_ret, ann_factor=252):
    r = np.asarray(step_ret, dtype=float).ravel()
    if r.size < 2:
        return float("nan")
    s = float(np.std(r, ddof=1))
    if s < 1e-18:
        return float("nan")
    return float(np.sqrt(float(ann_factor)) * float(np.mean(r)) / s)


# Function to plot the RQ4 transaction cost sensitivity lines
def rq4_transaction_cost_sensitivity_lines(bps_grid,total_returns,sharpe_ratios,title="MPC performance vs transaction-penalty scale",xlabel="Transaction-penalty index (basis points; see caption)",footnote=None,save_path=None,show=True):
    bps_grid = np.asarray(bps_grid, dtype=float).ravel()
    tr = np.asarray(total_returns, dtype=float).ravel()
    sh = np.asarray(sharpe_ratios, dtype=float).ravel()
    if not (len(bps_grid) == len(tr) == len(sh)):
        raise ValueError("bps_grid, total_returns, sharpe_ratios must have the same length.")

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    c_tr = THESIS_COLOR_MPC
    c_sh = "#6c6c6c"
    ax1.plot(bps_grid, tr * 100.0, "o-", color=c_tr, linewidth=2.0, markersize=6, label="Total return (%)")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Total return (%)", color=c_tr)
    ax1.tick_params(axis="y", labelcolor=c_tr)

    ax2 = ax1.twinx()
    ax2.plot(bps_grid, sh, "s--", color=c_sh, linewidth=1.8, markersize=5, alpha=0.95, label="Annualized Sharpe")
    ax2.set_ylabel("Annualized Sharpe", color=c_sh)
    ax2.tick_params(axis="y", labelcolor=c_sh)

    ax1.set_title(title)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    if footnote:
        fig.text(0.5, 0.02, footnote, ha="center", fontsize=7.5, wrap=True)
        fig.subplots_adjust(bottom=0.20)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# Function to plot the RQ4 underwater drawdown
def rq4_underwater_drawdown(curves_by_name,x=None,title="Underwater (drawdown) curve",save_path=None,show=True):
    if not isinstance(curves_by_name, dict) or len(curves_by_name) == 0:
        raise ValueError("curves_by_name must be a non-empty dict of strategy name -> equity series.")

    series = {k: _as_1d(v, k) for k, v in curves_by_name.items()}
    L = len(next(iter(series.values())))
    for k, v in series.items():
        if len(v) != L:
            raise ValueError(f"All equity curves must have length {L}; mismatch at {k}.")

    if x is None:
        x_plot = np.arange(L)
        xlab = "Time step"
    else:
        x_plot = np.asarray(x)
        if len(x_plot) != L:
            raise ValueError("x must match equity curve length.")
        xlab = "Date"

    fig, ax = plt.subplots(figsize=(12, 4.2))
    cmap = plt.get_cmap("tab10")
    for i, (name, eq) in enumerate(series.items()):
        cummax = np.maximum.accumulate(eq)
        dd_pct = (eq / (cummax + 1e-15) - 1.0) * 100.0
        col = thesis_color_for_strategy_label(name)
        if col is None:
            col = cmap(i % 10)
        ax.plot(x_plot, dd_pct, color=col, linewidth=1.2, label=name, alpha=0.92)
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Drawdown from own running peak (%)")
    ax.legend(loc="lower left", fontsize=8, ncol=2 if len(series) > 3 else 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

# Function to plot the RQ4 drawdown statistics table
def rq4_drawdown_stats_table(curves_by_name, title="RQ4: Drawdown statistics by strategy", save_path=None, show=True):
    if not isinstance(curves_by_name, dict) or len(curves_by_name) == 0:
        raise ValueError("curves_by_name must be a non-empty dict.")

    rows = []
    labels = []
    for name, curve in curves_by_name.items():
        eq = _as_1d(curve, name)
        peak = np.maximum.accumulate(eq)
        dd_pct = (eq / (peak + 1e-15) - 1.0) * 100.0
        max_dd = float(np.min(dd_pct))
        mean_dd = float(np.mean(dd_pct))
        end_dd = float(dd_pct[-1])
        uw_frac = float(np.mean(dd_pct < 0.0) * 100.0)
        labels.append(str(name))
        rows.append(
            [
                f"{max_dd:.2f}",
                f"{mean_dd:.2f}",
                f"{end_dd:.2f}",
                f"{uw_frac:.1f}",
            ]
        )

    fig_h = max(4.2, 0.45 * len(labels) + 2.0)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        rowLabels=labels,
        colLabels=["Max DD (%)", "Mean DD (%)", "End DD (%)", "Time underwater (%)"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.25)
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# Function to plot the RQ4 horizon profit violin
def rq4_horizon_profit_violin(profits_by_horizon_label,title="Distribution of terminal profit by MPC horizon (block-bootstrap paths)",highlight_horizon=None,save_path=None,show=True):
    if not isinstance(profits_by_horizon_label, dict) or len(profits_by_horizon_label) == 0:
        raise ValueError("profits_by_horizon_label must be a non-empty dict.")

    def _horizon_sort_key(k):
        s = str(k)
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else 0

    names = sorted(profits_by_horizon_label.keys(), key=_horizon_sort_key)
    data = [_as_1d(profits_by_horizon_label[n], n) for n in names]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    pos = np.arange(1, len(names) + 1, dtype=float)
    parts = ax.violinplot(data, positions=pos, showmeans=True, showextrema=True, widths=0.65)
    for pc in parts["bodies"]:
        pc.set_facecolor(THESIS_COLOR_MPC)
        pc.set_alpha(0.55)
        pc.set_edgecolor("#0b3a5e")
    ax.set_xticks(pos)
    ax.set_xticklabels(names)
    ax.set_xlabel("MPC look-ahead horizon (steps)")
    ax.set_ylabel("Terminal profit ($)")
    ax.set_title(title)
    if highlight_horizon is not None:
        hi = int(highlight_horizon)
        key = f"H = {hi}"
        if key in names:
            j = names.index(key)
            ax.axvline(
                float(pos[j]),
                color=THESIS_COLOR_MEAN_MARKER,
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                label=f"Selected H = {hi}",
            )
            ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

