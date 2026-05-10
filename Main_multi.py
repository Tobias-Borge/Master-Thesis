
import warnings
import os
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from MEMD import download_and_prepare_data, memd, plot_direction_hypersphere_3d
USE_TF_MODELS = False
if USE_TF_MODELS:
    from LSTM_tf import LSTM
    from MLP_tf import MLP
else:
    from LSTM import LSTM
    from MLP import MLP
from ARIMA import forecast_arima_lstm_low_imf
from GARCH import fit_garch_volatility_feature, hf_walk_forward_forecast, walk_forward_close_garch_lstm
from MPC import TradingMPC, run_mpc_backtest, run_mpc_backtest_returns, run_mpc_backtest_with_horizon_returns
import evaluation_graphs
from helper_function import sample_entropy
from ablation_models.ablation_arima import forecast_arima_standalone
from ablation_models.ablation_memd_lstm import forecast_memd_lstm_sum
from ablation_models.ablation_memd_mlp import forecast_memd_mlp_sum
from ablation_models.ablation_hybrid_pipeline import combine_hybrid_pipeline
from ablation_models.ablation_memd_arima_garch_mlp import combine_memd_arima_garch_mlp

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ==============================
# Ticker list
# ==============================
ticker_list = [
    "SPY",   # US large cap (S&P 500)
    "VEA",   # Developed ex-US equity
    "VWO",   # Emerging equity
    "AGG",   # US aggregate bonds
    "BNDX",  # International bonds
    "VGIT",  # Intermediate US Treasuries
    "VTIP",  # Short TIPS
    "GLD",   # Gold
    "VNQ",   # US REITs
    "PDBC",  # Broad commodities 
    "AAPL",  # Tech
    "MSFT",  # Tech
    "NVDA",  # Semiconductors/AI
    "JPM",   # Financials
    "UNH",   # Healthcare
    "XOM",   # Energy
    "PG",    # Consumer staples
    "CAT",   # Industrials
    "NEE",   # Utilities
    "AMZN",  # Consumer discretionary
]

# Single ticker
# ticker_list = ["JPM","AAPL"]

# Global parameters
PERIOD = "10y"
TRAIN_RATIO = 0.7
MPC_HORIZON = 5
# Forecasting strategy: train/evaluate 1-step models, and build the MPC horizon path by
# recursive 1-step rollout (errors accumulate with horizon).
FORECAST_DAYS_AHEAD = int(MPC_HORIZON)  # rollout width for horizon-matrices used by MPC
FORECAST_HORIZON_TRAINING_MODE = "recursive"
FORECAST_TRAIN_TARGET_STEP = 1
FORECAST_OUTPUT_HORIZON = 1

# MEMD parameters
MEMD_MAX_IMFS = None
MEMD_MAX_SIFT_ITER = 350
MEMD_SD_THRESHOLD = 0.05
MEMD_N_DIRECTIONS = 512
MEMD_ENVELOPE_METHOD = "cubic"

# IMF grouping parameters
SE_M = 2
SE_R_RATIO = 0.2
SE_DOWNSAMPLE_STEP = 5

# GARCH parameters
GARCH_ORDER = (1, 1)
GARCH_RESCALE = 100.0

# LSTM (high-frequency IMFs) parameters
HF_SEQ_LEN = 10
HF_HIDDEN_SIZE = 128
HF_EPOCHS = 40
HF_MODEL_TYPE = "LSTM"  # "LSTM" or "MLP"

# ARIMA-LSTM (low-frequency IMFs) parameters
RES_SEQ_LEN = 350
RES_HIDDEN_SIZE = 64
RES_EPOCHS = 40
RES_LR = 0.00005
# LSTM (high-frequency IMFs) parameters
RES_MODEL_TYPE = HF_MODEL_TYPE  # "LSTM" or "MLP"
SKIP_LSTM_IF_IID = True
# MPC parameters
MULTI_INITIAL_CASH = 1000.0
# GARCH + LSTM parameters
# Keep strict walk-forward updates for 1-step accuracy; horizon rows are built via
# separate recursive rollout inside the forecast functions.
HF_USE_OBSERVED_WALK_FORWARD_UPDATE = True

# Portfolio: MPC (long-only, fully invested)
# MPC parameters
# MPC_HORIZON is set above (independent of model output size).
MPC_RISK_AVERSION = 0.5
MPC_TRADE_COST = 0.05  # legacy alias, maps to impact_cost if MPC_IMPACT_COST is None
MPC_LINEAR_COST = 0.0005
MPC_IMPACT_COST = None
MPC_FIXED_TICKET_COST = 0.5
MPC_FIXED_TRADE_EPS = 1e-3
MPC_MAX_TURNOVER_L1 = None  # e.g. 0.8 to cap sum_i |Δw_i| per step; None = no cap
MPC_MIN_TRADE_THRESHOLD = 1e-4

# RQ1 — forecasting accuracy: ARIMA, LSTM/MLP close-only, MEMD-LSTM/MLP, GARCH-LSTM, ARIMA-LSTM, MEMD-ARIMA-GARCH-MLP, hybrid
RQ1_PLOTS = True
# Run expensive ablation-model branch for one reference ticker only.
RQ1_ABLATION_TICKER = "AAPL"
RQ1_SIGNIFICANCE_TESTS = True
RQ1_DM_MAX_LAG = None
RQ1_SPA_BOOT = 500
RQ1_SPA_BLOCK_LEN = 10
RQ1_SPA_SEED = 42
RQ1_TOP_SCATTER_MODELS = 3
# LSTM (close only) parameters
BASELINE_LSTM_SEQ_LEN = 100
BASELINE_LSTM_EPOCHS = 40
BASELINE_LSTM_HIDDEN = 128
BASELINE_LSTM_LR = 1e-5

# MEMD-LSTM ablation: walk-forward LSTM on each IMF (Close) then sum (+ residue forecast); lighter than baseline to limit runtime
MEMD_LSTM_SEQ_LEN = 100
MEMD_LSTM_EPOCHS = 40
MEMD_LSTM_HIDDEN = 128

# MLP (close only) baseline — independent of LSTM baseline hyperparameters
BASELINE_MLP_SEQ_LEN = 100
BASELINE_MLP_EPOCHS = 40
BASELINE_MLP_HIDDEN = 128
BASELINE_MLP_LR = 1e-5

# MEMD-MLP: per-IMF MLP on Close paths (independent of MEMD-LSTM settings)
MEMD_MLP_SEQ_LEN = 100
MEMD_MLP_EPOCHS = 40
MEMD_MLP_HIDDEN = 128
MEMD_MLP_LR = 1e-5

# MEMD-ARIMA-GARCH-MLP hybrid: low IMF ARIMA + MLP residual, HF GARCH + MLP (independent of RES_* / HF_* LSTM hybrid)
HYBRID_MLP_RES_SEQ_LEN = 350
HYBRID_MLP_RES_HIDDEN = 64
HYBRID_MLP_RES_EPOCHS = 40
HYBRID_MLP_RES_LR = 0.00005
HYBRID_MLP_HF_SEQ_LEN = 10
HYBRID_MLP_HF_HIDDEN = 128
HYBRID_MLP_HF_EPOCHS = 40
HYBRID_MLP_HF_LR = 5e-5

# RQ2 — MEMD / ablation / residual diagnostics (uses RQ1 metrics for heatmap when RQ1_PLOTS is on)
RQ2_PLOTS = True
# MEMD ablation parameters
RQ2_ABLATION_ORDER = (
    "ARIMA (standalone)",
    "LSTM (close only)",
    "MLP (close only)",
    "MEMD-LSTM",
    "MEMD-MLP",
    "GARCH-LSTM",
    "ARIMA-LSTM",
    "MEMD-ARIMA-GARCH-MLP",
    "Hybrid (pipeline)",
)
# MEMD residual diagnostics parameters
RQ2_ACF_NLAGS = 40
# Optional: export MEMD decomposition for all OHLCV channels for one ticker only
RQ2_MEMD_ALL_CHANNELS_TICKER = "JPM"
RQ2_MEMD_ALL_CHANNELS = True
# Optional: one thesis figure for MEMD direction set (3D projection) for a single ticker
RQ2_DIRECTION_PLOT_TICKER = "JPM"
RQ2_DIRECTION_PLOT_3D = True

# RQ3 — risk-adjusted portfolio performance (wealth curves, bootstrap)
RQ3_PLOTS = True
# RQ3 bootstrap parameters
RQ3_BOOTSTRAP_SIMS = 500
RQ3_BOOTSTRAP_SEED = 42
# Compare MPC objectives in RQ3 wealth plot
RQ3_COMPARE_MPC_OBJECTIVES = True

# RQ4 — market frictions and realism (cost sweep, drawdown, horizon choice)
RQ4_PLOTS = True
# RQ4 annualization factor parameters
RQ4_ANN_FACTOR = 252  # Sharpe annualization (daily steps)
# RQ4 linear-cost sensitivity grid parameters
RQ4_COST_BPS_GRID = np.linspace(0.0, 100.0, 11)
RQ4_BPS_FOR_BASELINE_PENALTY = 10.0  # x-axis bps at which linear cost equals MPC_LINEAR_COST
# RQ4 impact-cost sensitivity scale (multiplier of baseline impact coefficient)
RQ4_IMPACT_SCALE_GRID = np.linspace(0.0, 3.0, 11)
# RQ4 fixed-ticket sensitivity scale (multiplier of baseline fixed-ticket coefficient)
RQ4_FIXED_SCALE_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=float)
# RQ4 horizon grid parameters
RQ4_HORIZON_GRID = (2, 3, 5, 7, 10)
# RQ4 horizon block multiplier parameters
RQ4_HORIZON_BLOCK_MULT = 3
# RQ4 horizon simulations parameters
RQ4_HORIZON_SIMS = 300
# RQ4 horizon seed parameters
RQ4_HORIZON_SEED = 43

# Report mode: force all thesis RQ sections (RQ1-RQ4) to run.
REPORT_MODE = True
if REPORT_MODE:
    RQ1_PLOTS = True
    RQ2_PLOTS = True
    RQ3_PLOTS = True
    RQ4_PLOTS = True

# Plot display behavior: suppress popups during run, optionally replay saved plots at end.
SHOW_PLOTS_DURING_RUN = False
SHOW_PLOTS_AT_END = False

# Ensure horizon rollouts cover the full RQ4 horizon grid (e.g. 10), even if the
# primary MPC horizon is smaller (e.g. 5). MPC will slice what it needs.
try:
    FORECAST_DAYS_AHEAD = int(max(int(FORECAST_DAYS_AHEAD), int(max(RQ4_HORIZON_GRID))))
except Exception:
    FORECAST_DAYS_AHEAD = int(FORECAST_DAYS_AHEAD)

# Results directory parameters
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
MODEL_CACHE_DIR = "model_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
SAVE_TRAINED_MODELS = True
FORCE_RETRAIN_MODELS = False


# Function to build the results path
def _results_path(filename):
    return os.path.join(RESULTS_DIR, filename)

# Function to build the model cache path
def _model_cache_path(model_key):
    safe = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(model_key))
    return os.path.join(MODEL_CACHE_DIR, f"{safe}.npz")

# Function to build the cache context prefix
def _cache_context_prefix():
    backend = "tf" if USE_TF_MODELS else "np"
    horizon = int(max(1, FORECAST_DAYS_AHEAD))
    mode = str(FORECAST_HORIZON_TRAINING_MODE).lower()
    tstep = int(max(1, FORECAST_TRAIN_TARGET_STEP))
    return f"{backend}_fd{horizon}_{mode}_ts{tstep}"

# Function to build the scoped model key
def _scoped_model_key(model_key):
    return f"{_cache_context_prefix()}__{model_key}"

# Function to build a full scoped cache file path
def _scoped_model_cache_path(model_key):
    return _model_cache_path(_scoped_model_key(model_key))

# Function to try to load the model
def _try_load_model(model_cls, model_key):
    if FORCE_RETRAIN_MODELS:
        return None
    fp = _model_cache_path(_scoped_model_key(model_key))
    if not os.path.exists(fp):
        return None
    try:
        m = model_cls.load(fp)
        print(f"Loaded pretrained model: {fp}")
        return m
    except Exception as ex:
        print(f"Pretrained load failed ({fp}), retraining: {ex}")
        return None

# Function to try to save the model
def _try_save_model(model_obj, model_key):
    if not SAVE_TRAINED_MODELS:
        return
    fp = _model_cache_path(_scoped_model_key(model_key))
    try:
        model_obj.save(fp)
    except Exception as ex:
        print(f"Model save skipped ({fp}): {ex}")

# Function to build the high-frequency model
def _build_hf_model(input_size, hidden_size, output_horizon=1):
    model_type = str(HF_MODEL_TYPE).upper()
    if model_type == "MLP":
        return MLP(input_size=input_size, hidden_size=hidden_size, output_size=int(output_horizon))
    return LSTM(input_size=input_size, hidden_size=hidden_size, output_size=int(output_horizon))


# Function to build the baseline LSTM walk-forward close model
def _baseline_lstm_walk_forward_close(train_norm,test_norm,seq_len,epochs,hidden_size,learning_rate=5e-5,model_cache_key=None,use_observed_update=True,target_step=1,output_horizon=1,
):
    train = np.asarray(train_norm, dtype=float).ravel()
    test = np.asarray(test_norm, dtype=float).ravel()
    Tt, H = len(train), len(test)
    seq_len = int(seq_len)
    if Tt <= seq_len or H == 0 or epochs < 1:
        return None
    out_h = int(max(1, output_horizon))
    model = _try_load_model(LSTM, model_cache_key) if model_cache_key is not None else None
    trained_now = False
    if model is None:
        model = LSTM(1, int(hidden_size), int(out_h), learning_rate=float(learning_rate))
    Xs, ys = [], []
    h = int(max(1, target_step))
    n_pairs = Tt - seq_len - max(h, out_h) + 1
    if n_pairs <= 0:
        return None
    for t in range(max(0, n_pairs)):
        w = train[t : t + seq_len].reshape(seq_len, 1)
        Xs.append(w)
        if out_h > 1:
            ys.append(train[t + seq_len : t + seq_len + out_h].reshape(out_h, 1))
        else:
            ys.append([[train[t + seq_len + h - 1]]])
    if model_cache_key is None or model is None:
        for _ in range(int(epochs)):
            for i in range(len(Xs)):
                model.train_step(Xs[i], ys[i])
        trained_now = True
    if trained_now and model_cache_key is not None:
        _try_save_model(model, model_cache_key)
    full = np.concatenate([train, test]).copy()
    pred = np.zeros(H, dtype=float)
    for h in range(H):
        start = Tt + h - seq_len
        window = full[start : Tt + h].reshape(seq_len, 1)
        cache = model.forward(window)
        y_vec = (model.Why @ cache[-1][0] + model.by).flatten()
        y_hat = float(y_vec[0])
        pred[h] = y_hat
        if not use_observed_update:
            full[Tt + h] = y_hat
    return pred

# Function to run the baseline MLP walk-forward close
def _baseline_mlp_walk_forward_close(train_norm,test_norm,seq_len,epochs,hidden_size,learning_rate=5e-5,model_cache_key=None,use_observed_update=True,target_step=1,output_horizon=1,
):
    train = np.asarray(train_norm, dtype=float).ravel()
    test = np.asarray(test_norm, dtype=float).ravel()
    Tt, H = len(train), len(test)
    seq_len = int(seq_len)
    if Tt <= seq_len or H == 0 or epochs < 1:
        return None
    out_h = int(max(1, output_horizon))
    model = _try_load_model(MLP, model_cache_key) if model_cache_key is not None else None
    trained_now = False
    if model is None:
        model = MLP(1, int(hidden_size), int(out_h), learning_rate=float(learning_rate))
    Xs, ys = [], []
    h = int(max(1, target_step))
    n_pairs = Tt - seq_len - max(h, out_h) + 1
    if n_pairs <= 0:
        return None
    for t in range(max(0, n_pairs)):
        w = train[t : t + seq_len].reshape(seq_len, 1)
        Xs.append(w)
        if out_h > 1:
            ys.append(train[t + seq_len : t + seq_len + out_h].reshape(out_h, 1))
        else:
            ys.append([[train[t + seq_len + h - 1]]])
    if model_cache_key is None or model is None:
        for _ in range(int(epochs)):
            for i in range(len(Xs)):
                model.train_step(Xs[i], ys[i])
        trained_now = True
    if trained_now and model_cache_key is not None:
        _try_save_model(model, model_cache_key)
    full = np.concatenate([train, test]).copy()
    pred = np.zeros(H, dtype=float)
    for h in range(H):
        start = Tt + h - seq_len
        window = full[start : Tt + h].reshape(seq_len, 1)
        cache = model.forward(window)
        y_vec = (model.Why @ cache[-1][0] + model.by).flatten()
        y_hat = float(y_vec[0])
        pred[h] = y_hat
        if not use_observed_update:
            full[Tt + h] = y_hat
    return pred

# Function to run the MEMD-LSTM per IMF forecast sum
def _memd_lstm_per_imf_forecast_sum(IMFs,close_idx,n_imfs,train_slice,test_slice,seq_len,epochs,hidden_size,model_cache_prefix=None,use_observed_update=True,target_step=1,output_horizon=1,
):
    H = int(test_slice.stop - test_slice.start)
    out = np.zeros(H, dtype=float)
    for k in range(int(n_imfs)):
        tr = np.asarray(IMFs[k, close_idx, train_slice], dtype=float).ravel()
        te = np.asarray(IMFs[k, close_idx, test_slice], dtype=float).ravel()
        cache_key = None if model_cache_prefix is None else f"{model_cache_prefix}_imf{k}_sl{seq_len}_ep{epochs}_h{hidden_size}"
        fc = _baseline_lstm_walk_forward_close(
            tr,
            te,
            seq_len,
            epochs,
            hidden_size,
            model_cache_key=cache_key,
            use_observed_update=use_observed_update,
            target_step=target_step,
            output_horizon=output_horizon,
        )
        if fc is None or len(fc) != H:
            return None
        out += fc
    return out

# Function to run the MEMD-MLP per IMF forecast sum
def _memd_mlp_per_imf_forecast_sum(IMFs,close_idx,n_imfs,train_slice,test_slice,seq_len,epochs,hidden_size,learning_rate=5e-5,model_cache_prefix=None,use_observed_update=True,target_step=1,output_horizon=1,
):
    H = int(test_slice.stop - test_slice.start)
    out = np.zeros(H, dtype=float)
    for k in range(int(n_imfs)):
        tr = np.asarray(IMFs[k, close_idx, train_slice], dtype=float).ravel()
        te = np.asarray(IMFs[k, close_idx, test_slice], dtype=float).ravel()
        cache_key = None if model_cache_prefix is None else f"{model_cache_prefix}_imf{k}_sl{seq_len}_ep{epochs}_h{hidden_size}_lr{learning_rate}"
        fc = _baseline_mlp_walk_forward_close(
            tr,
            te,
            seq_len,
            epochs,
            hidden_size,
            learning_rate=learning_rate,
            model_cache_key=cache_key,
            use_observed_update=use_observed_update,
            target_step=target_step,
            output_horizon=output_horizon,
        )
        if fc is None or len(fc) != H:
            return None
        out += fc
    return out


# Function to run the single ticker
def run_single_ticker(ticker):
    print(f"\n==================== {ticker} ====================")

    data_for_memd, idx, channel_names, prep_meta = download_and_prepare_data(ticker=ticker,period=PERIOD,return_metadata=True,)
    print(f"{ticker} data shape: {data_for_memd.shape} (channels x samples)")

    imfs, residue = memd(data_for_memd,max_imfs=MEMD_MAX_IMFS,max_sift_iter=MEMD_MAX_SIFT_ITER,sd_threshold=MEMD_SD_THRESHOLD,n_directions=MEMD_N_DIRECTIONS,envelope_method=MEMD_ENVELOPE_METHOD,)
    IMFs = np.stack(imfs, axis=0)
    n_imfs, _, T = IMFs.shape
    print(f"{ticker} IMFs extracted: {n_imfs}")

    T_train = int(T * TRAIN_RATIO)
    train_slice = slice(0, T_train)
    test_slice = slice(T_train, T)
    horizon = T - T_train
    close_idx = channel_names.index("Close")

    # Split IMFs by entropy
    entropies = []
    for k in range(n_imfs):
        entropies.append(
            sample_entropy(
                IMFs[k, close_idx, :],
                m=SE_M,
                r_ratio=SE_R_RATIO,
                step=SE_DOWNSAMPLE_STEP,
            )
        )
    entropies = np.array(entropies, dtype=float)
    threshold = np.nanmedian(entropies)
    high_freq_indices = [k for k, se in enumerate(entropies) if se >= threshold]
    low_freq_indices = [k for k, se in enumerate(entropies) if se < threshold]

    # Low-frequency: ARIMA-LSTM
    low_imf_forecasts = {}
    low_imf_hmats = {}
    low_imf_outputs = {}
    for k in low_freq_indices:
        series_full = IMFs[k, close_idx, :]
        low_res_cache_key = (
            f"{ticker}_low_{str(RES_MODEL_TYPE).upper()}_imf{k}_sl{RES_SEQ_LEN}_"
            f"ep{RES_EPOCHS}_h{RES_HIDDEN_SIZE}_lr{RES_LR}"
        )
        out = forecast_arima_lstm_low_imf(series_train=series_full[train_slice],series_test=series_full[test_slice],res_seq_len=RES_SEQ_LEN,res_hidden_size=RES_HIDDEN_SIZE,res_epochs=RES_EPOCHS,
            res_lr=RES_LR,res_model_type=RES_MODEL_TYPE,skip_lstm_if_iid=SKIP_LSTM_IF_IID,forecast_days_ahead=FORECAST_DAYS_AHEAD,residual_target_step=FORECAST_TRAIN_TARGET_STEP,output_horizon=FORECAST_OUTPUT_HORIZON,imf_label=f"{ticker} low-IMF {k+1}",residual_model_cache_path=_scoped_model_cache_path(low_res_cache_key),force_retrain_model=FORCE_RETRAIN_MODELS,save_trained_model=SAVE_TRAINED_MODELS,)
        low_imf_forecasts[k] = out["combined"]
        low_imf_hmats[k] = np.asarray(out.get("combined_hmat", np.asarray(out["combined"], dtype=float).reshape(-1, 1)), dtype=float)
        low_imf_outputs[k] = out

    # Residual path: apply the same low-frequency ARIMA+residual-model stack on MEMD residue
    residue_full = np.asarray(residue[close_idx, :], dtype=float).ravel()
    residue_hybrid_cache_key = (
        f"{ticker}_low_{str(RES_MODEL_TYPE).upper()}_residue_hybrid_sl{RES_SEQ_LEN}_"
        f"ep{RES_EPOCHS}_h{RES_HIDDEN_SIZE}_lr{RES_LR}"
    )
    out_residual_hybrid = forecast_arima_lstm_low_imf(series_train=residue_full[train_slice],series_test=residue_full[test_slice],res_seq_len=RES_SEQ_LEN,res_hidden_size=RES_HIDDEN_SIZE,res_epochs=RES_EPOCHS,res_lr=RES_LR,res_model_type=RES_MODEL_TYPE,skip_lstm_if_iid=SKIP_LSTM_IF_IID,forecast_days_ahead=FORECAST_DAYS_AHEAD,residual_target_step=FORECAST_TRAIN_TARGET_STEP,output_horizon=FORECAST_OUTPUT_HORIZON,imf_label=f"{ticker} residue (hybrid)",
        residual_model_cache_path=_scoped_model_cache_path(residue_hybrid_cache_key),force_retrain_model=FORCE_RETRAIN_MODELS,save_trained_model=SAVE_TRAINED_MODELS,
    )
    residue_hybrid_forecast = np.asarray(out_residual_hybrid["combined"], dtype=float).ravel()
    residue_hybrid_hmat = np.asarray(
        out_residual_hybrid.get("combined_hmat", residue_hybrid_forecast.reshape(-1, 1)),
        dtype=float,
    )
    if residue_hybrid_forecast.size < horizon:
        _pad_v = float(residue_hybrid_forecast[-1]) if residue_hybrid_forecast.size > 0 else float(residue[close_idx, T_train - 1])
        residue_hybrid_forecast = np.concatenate(
            [residue_hybrid_forecast, np.full(horizon - residue_hybrid_forecast.size, _pad_v, dtype=float)]
        )

    # High-frequency: GARCH + LSTM
    garch_lstm_forecasts = []
    garch_lstm_hmats = []
    seq_len = HF_SEQ_LEN
    for k in high_freq_indices:
        imf_k = IMFs[k]
        close_train = imf_k[close_idx, train_slice]
        garch_feature_full = fit_garch_volatility_feature(close_train=close_train,horizon=horizon,order=GARCH_ORDER,rescale=GARCH_RESCALE,)

        X_seqs = []
        y_targets = []
        hf_step = int(max(1, FORECAST_TRAIN_TARGET_STEP))
        hf_out = int(max(1, FORECAST_OUTPUT_HORIZON))
        n_hf_pairs = T_train - seq_len - max(hf_step, hf_out) + 1
        for t in range(max(0, n_hf_pairs)):
            imf_window = imf_k[:, t : t + seq_len].T
            garch_feat = garch_feature_full[t : t + seq_len].reshape(seq_len, 1)
            window = np.concatenate([imf_window, garch_feat], axis=1)
            X_seqs.append(window)
            if hf_out > 1:
                y_targets.append(
                    np.asarray(
                        imf_k[close_idx, t + seq_len : t + seq_len + hf_out],
                        dtype=float,
                    ).reshape(hf_out, 1)
                )
            else:
                y_targets.append([[imf_k[close_idx, t + seq_len + hf_step - 1]]])

        X_seqs = np.array(X_seqs, dtype=float)
        y_targets = np.array(y_targets, dtype=float)

        hf_cls = MLP if str(HF_MODEL_TYPE).upper() == "MLP" else LSTM
        hf_cache_key = f"{ticker}_hf_{str(HF_MODEL_TYPE).upper()}_imf{k}_sl{HF_SEQ_LEN}_ep{HF_EPOCHS}_h{HF_HIDDEN_SIZE}"
        model_hf = _try_load_model(hf_cls, hf_cache_key)
        if model_hf is None:
            model_hf = _build_hf_model(input_size=6, hidden_size=HF_HIDDEN_SIZE, output_horizon=hf_out)
            for _epoch in range(HF_EPOCHS):
                for i in range(len(X_seqs)):
                    model_hf.train_step(X_seqs[i], y_targets[i])
            _try_save_model(model_hf, hf_cache_key)

        fc, fc_hmat = hf_walk_forward_forecast(
            model=model_hf,
            imf_k=imf_k,
            T_train=T_train,
            horizon=horizon,
            seq_len=seq_len,
            close_idx=close_idx,
            garch_feature_full=garch_feature_full,
            use_observed_update=HF_USE_OBSERVED_WALK_FORWARD_UPDATE,
            output_horizon=FORECAST_OUTPUT_HORIZON,
            return_horizon_matrix=True,
            rollout_horizon=int(max(1, FORECAST_DAYS_AHEAD)),
        )
        if len(fc) < horizon:
            pad_value = fc[-1] if len(fc) > 0 else 0.0
            fc = np.concatenate([fc, np.full(horizon - len(fc), pad_value, dtype=float)])
            pad_h = (
                np.full((horizon - fc_hmat.shape[0], fc_hmat.shape[1]), fc_hmat[-1], dtype=float)
                if fc_hmat.shape[0] > 0
                else np.zeros((horizon - fc_hmat.shape[0], int(max(1, MPC_HORIZON))), dtype=float)
            )
            fc_hmat = np.vstack([fc_hmat, pad_h])
        garch_lstm_forecasts.append(fc)
        garch_lstm_hmats.append(fc_hmat)

    garch_lstm_forecasts = np.array(garch_lstm_forecasts, dtype=float)
    low_imf_forecasts_array = np.array([low_imf_forecasts[k] for k in low_freq_indices], dtype=float)
    Hm_stack = int(max(1, FORECAST_DAYS_AHEAD))
    low_imf_hmats_norm = []
    for k in low_freq_indices:
        hm = np.asarray(low_imf_hmats[k], dtype=float)
        if hm.ndim == 1:
            hm = hm.reshape(-1, 1)
        if hm.shape[0] < horizon:
            pad_row = hm[-1:, :] if hm.shape[0] > 0 else np.zeros((1, hm.shape[1]), dtype=float)
            hm = np.vstack([hm, np.repeat(pad_row, horizon - hm.shape[0], axis=0)])
        elif hm.shape[0] > horizon:
            hm = hm[:horizon, :]
        if hm.shape[1] < Hm_stack:
            pad_col = hm[:, [-1]] if hm.shape[1] > 0 else np.zeros((hm.shape[0], 1), dtype=float)
            hm = np.concatenate([hm, np.repeat(pad_col, Hm_stack - hm.shape[1], axis=1)], axis=1)
        elif hm.shape[1] > Hm_stack:
            hm = hm[:, :Hm_stack]
        low_imf_hmats_norm.append(hm)
    low_imf_hmats_array = (
        np.array(low_imf_hmats_norm, dtype=float)
        if len(low_freq_indices) > 0
        else np.zeros((0, horizon, int(max(1, MPC_HORIZON))), dtype=float)
    )

    run_ablation_for_ticker = str(ticker).upper() == str(RQ1_ABLATION_TICKER).upper()
    hybrid_mlp_norm = None
    ablation_memd_lstm_norm = None
    ablation_memd_mlp_norm = None
    ablation_garch_lstm_norm = None
    ablation_arima_lstm_norm = None
    if run_ablation_for_ticker:
        # MEMD-ARIMA-GARCH-MLP: same architecture as hybrid (pipeline) but MLP stacks; uses HYBRID_MLP_* tunables
        hf_mlp_seq = int(HYBRID_MLP_HF_SEQ_LEN)
        low_imf_forecasts_mlp = {}
        for k in low_freq_indices:
            series_full = IMFs[k, close_idx, :]
            low_mlp_cache_key = (
                f"{ticker}_low_MLP_imf{k}_hybrid_sl{HYBRID_MLP_RES_SEQ_LEN}_"
                f"ep{HYBRID_MLP_RES_EPOCHS}_h{HYBRID_MLP_RES_HIDDEN}_lr{HYBRID_MLP_RES_LR}"
            )
            out_mlph = forecast_arima_lstm_low_imf(series_train=series_full[train_slice],series_test=series_full[test_slice],res_seq_len=HYBRID_MLP_RES_SEQ_LEN,res_hidden_size=HYBRID_MLP_RES_HIDDEN,res_epochs=HYBRID_MLP_RES_EPOCHS,
                res_lr=HYBRID_MLP_RES_LR,res_model_type="MLP",skip_lstm_if_iid=SKIP_LSTM_IF_IID,forecast_days_ahead=FORECAST_DAYS_AHEAD,residual_target_step=FORECAST_TRAIN_TARGET_STEP,output_horizon=FORECAST_OUTPUT_HORIZON,imf_label=f"{ticker} low-IMF {k+1} (MLP hybrid)",residual_model_cache_path=_scoped_model_cache_path(low_mlp_cache_key),force_retrain_model=FORCE_RETRAIN_MODELS,save_trained_model=SAVE_TRAINED_MODELS,)
            low_imf_forecasts_mlp[k] = out_mlph["combined"]

        # Residual path for MEMD-ARIMA-GARCH-MLP: same low-frequency stack but with MLP residual model
        residue_mlp_cache_key = (
            f"{ticker}_low_MLP_residue_hybrid_sl{HYBRID_MLP_RES_SEQ_LEN}_"
            f"ep{HYBRID_MLP_RES_EPOCHS}_h{HYBRID_MLP_RES_HIDDEN}_lr{HYBRID_MLP_RES_LR}"
        )
        out_residual_mlp = forecast_arima_lstm_low_imf(series_train=residue_full[train_slice],series_test=residue_full[test_slice],res_seq_len=HYBRID_MLP_RES_SEQ_LEN,res_hidden_size=HYBRID_MLP_RES_HIDDEN,res_epochs=HYBRID_MLP_RES_EPOCHS,
            res_lr=HYBRID_MLP_RES_LR,res_model_type="MLP",skip_lstm_if_iid=SKIP_LSTM_IF_IID,forecast_days_ahead=FORECAST_DAYS_AHEAD,residual_target_step=FORECAST_TRAIN_TARGET_STEP,output_horizon=FORECAST_OUTPUT_HORIZON,imf_label=f"{ticker} residue (MLP hybrid)",residual_model_cache_path=_scoped_model_cache_path(residue_mlp_cache_key),force_retrain_model=FORCE_RETRAIN_MODELS,save_trained_model=SAVE_TRAINED_MODELS,)
        residue_mlp_forecast = np.asarray(out_residual_mlp["combined"], dtype=float).ravel()
        if residue_mlp_forecast.size < horizon:
            _pad_v_m = float(residue_mlp_forecast[-1]) if residue_mlp_forecast.size > 0 else float(residue[close_idx, T_train - 1])
            residue_mlp_forecast = np.concatenate(
                [residue_mlp_forecast, np.full(horizon - residue_mlp_forecast.size, _pad_v_m, dtype=float)]
            )

        garch_mlp_forecasts = []
        if T_train > hf_mlp_seq:
            for k in high_freq_indices:
                imf_k = IMFs[k]
                close_train_m = imf_k[close_idx, train_slice]
                garch_feature_m = fit_garch_volatility_feature(close_train=close_train_m,horizon=horizon,order=GARCH_ORDER,rescale=GARCH_RESCALE,)
                Xsm, ytm = [], []
                hf_mlp_step = int(max(1, FORECAST_TRAIN_TARGET_STEP))
                hf_mlp_out = int(max(1, FORECAST_OUTPUT_HORIZON))
                n_hf_mlp_pairs = T_train - hf_mlp_seq - max(hf_mlp_step, hf_mlp_out) + 1
                for t in range(max(0, n_hf_mlp_pairs)):
                    imf_window = imf_k[:, t : t + hf_mlp_seq].T
                    garch_feat = garch_feature_m[t : t + hf_mlp_seq].reshape(hf_mlp_seq, 1)
                    window = np.concatenate([imf_window, garch_feat], axis=1)
                    Xsm.append(window)
                    if hf_mlp_out > 1:
                        ytm.append(
                            np.asarray(
                                imf_k[close_idx, t + hf_mlp_seq : t + hf_mlp_seq + hf_mlp_out],
                                dtype=float,
                            ).reshape(hf_mlp_out, 1)
                        )
                    else:
                        ytm.append([[imf_k[close_idx, t + hf_mlp_seq + hf_mlp_step - 1]]])
                Xsm = np.array(Xsm, dtype=float)
                ytm = np.array(ytm, dtype=float)
                hf_mlp_cache_key = (
                    f"{ticker}_hf_MLP_hybrid_imf{k}_sl{hf_mlp_seq}_ep{int(HYBRID_MLP_HF_EPOCHS)}_"
                    f"h{int(HYBRID_MLP_HF_HIDDEN)}_lr{float(HYBRID_MLP_HF_LR)}"
                )
                model_hf_m = _try_load_model(MLP, hf_mlp_cache_key)
                if model_hf_m is None:
                    model_hf_m = MLP(
                        6,
                        int(HYBRID_MLP_HF_HIDDEN),
                        int(hf_mlp_out),
                        learning_rate=float(HYBRID_MLP_HF_LR),
                    )
                    for _epoch in range(int(HYBRID_MLP_HF_EPOCHS)):
                        for i in range(len(Xsm)):
                            model_hf_m.train_step(Xsm[i], ytm[i])
                    _try_save_model(model_hf_m, hf_mlp_cache_key)
                fc_m = hf_walk_forward_forecast(model=model_hf_m,imf_k=imf_k,T_train=T_train,horizon=horizon,seq_len=hf_mlp_seq,close_idx=close_idx,garch_feature_full=garch_feature_m,use_observed_update=HF_USE_OBSERVED_WALK_FORWARD_UPDATE,)
                if len(fc_m) < horizon:
                    pad_value = fc_m[-1] if len(fc_m) > 0 else 0.0
                    fc_m = np.concatenate([fc_m, np.full(horizon - len(fc_m), pad_value, dtype=float)])
                garch_mlp_forecasts.append(fc_m)

        if len(low_freq_indices) == 0:
            low_mlp_sum = np.zeros(horizon, dtype=float)
        else:
            low_mlp_arr = np.array([low_imf_forecasts_mlp[k] for k in low_freq_indices], dtype=float)
            low_mlp_sum = np.sum(low_mlp_arr, axis=0)
        if len(garch_mlp_forecasts) == 0:
            hf_mlp_sum = np.zeros(horizon, dtype=float)
        else:
            hf_mlp_sum = np.sum(np.asarray(garch_mlp_forecasts, dtype=float), axis=0)
        hybrid_mlp_norm = combine_memd_arima_garch_mlp(low_mlp_sum, hf_mlp_sum, residue_mlp_forecast)

        # Residual path for MEMD ablations (so IMF-sum ablations also reconstruct with dynamic residue forecast)
        res_tail = float(residue[close_idx, T_train - 1])
        residue_memd_lstm_cache_key = (
            f"{ticker}_low_LSTM_residue_memd_ablation_sl{MEMD_LSTM_SEQ_LEN}_"
            f"ep{MEMD_LSTM_EPOCHS}_h{MEMD_LSTM_HIDDEN}_lr{RES_LR}"
        )
        out_residual_memd_lstm = forecast_arima_lstm_low_imf(series_train=residue_full[train_slice],series_test=residue_full[test_slice],res_seq_len=MEMD_LSTM_SEQ_LEN,res_hidden_size=MEMD_LSTM_HIDDEN,res_epochs=MEMD_LSTM_EPOCHS,res_lr=RES_LR,res_model_type="LSTM",skip_lstm_if_iid=SKIP_LSTM_IF_IID,forecast_days_ahead=FORECAST_DAYS_AHEAD,residual_target_step=FORECAST_TRAIN_TARGET_STEP,output_horizon=FORECAST_OUTPUT_HORIZON,imf_label=f"{ticker} residue (MEMD-LSTM ablation)",residual_model_cache_path=_scoped_model_cache_path(residue_memd_lstm_cache_key),force_retrain_model=FORCE_RETRAIN_MODELS,save_trained_model=SAVE_TRAINED_MODELS,)
        residue_memd_lstm_forecast = np.asarray(out_residual_memd_lstm["combined"], dtype=float).ravel()
        if residue_memd_lstm_forecast.size < horizon:
            _pad_rl = float(residue_memd_lstm_forecast[-1]) if residue_memd_lstm_forecast.size > 0 else res_tail
            residue_memd_lstm_forecast = np.concatenate([residue_memd_lstm_forecast, np.full(horizon - residue_memd_lstm_forecast.size, _pad_rl, dtype=float)])

        residue_memd_mlp_cache_key = (
            f"{ticker}_low_MLP_residue_memd_ablation_sl{MEMD_MLP_SEQ_LEN}_"
            f"ep{MEMD_MLP_EPOCHS}_h{MEMD_MLP_HIDDEN}_lr{MEMD_MLP_LR}"
        )
        out_residual_memd_mlp = forecast_arima_lstm_low_imf(series_train=residue_full[train_slice],series_test=residue_full[test_slice],res_seq_len=MEMD_MLP_SEQ_LEN,res_hidden_size=MEMD_MLP_HIDDEN,res_epochs=MEMD_MLP_EPOCHS,res_lr=MEMD_MLP_LR,
            res_model_type="MLP",skip_lstm_if_iid=SKIP_LSTM_IF_IID,forecast_days_ahead=FORECAST_DAYS_AHEAD,residual_target_step=FORECAST_TRAIN_TARGET_STEP,output_horizon=FORECAST_OUTPUT_HORIZON,imf_label=f"{ticker} residue (MEMD-MLP ablation)",residual_model_cache_path=_scoped_model_cache_path(residue_memd_mlp_cache_key),force_retrain_model=FORCE_RETRAIN_MODELS,save_trained_model=SAVE_TRAINED_MODELS,)
        residue_memd_mlp_forecast = np.asarray(out_residual_memd_mlp["combined"], dtype=float).ravel()

        if residue_memd_mlp_forecast.size < horizon:
            _pad_rm = float(residue_memd_mlp_forecast[-1]) if residue_memd_mlp_forecast.size > 0 else res_tail
            residue_memd_mlp_forecast = np.concatenate([residue_memd_mlp_forecast, np.full(horizon - residue_memd_mlp_forecast.size, _pad_rm, dtype=float)])

        ablation_memd_lstm_norm = forecast_memd_lstm_sum(imfs=IMFs,close_idx=close_idx,train_slice=train_slice,test_slice=test_slice,seq_len=MEMD_LSTM_SEQ_LEN,epochs=MEMD_LSTM_EPOCHS,hidden_size=MEMD_LSTM_HIDDEN,use_observed_update=HF_USE_OBSERVED_WALK_FORWARD_UPDATE,target_step=FORECAST_TRAIN_TARGET_STEP,output_horizon=FORECAST_OUTPUT_HORIZON,)
        if ablation_memd_lstm_norm is not None:
            ablation_memd_lstm_norm = np.asarray(ablation_memd_lstm_norm, dtype=float) + residue_memd_lstm_forecast

        ablation_memd_mlp_norm = forecast_memd_mlp_sum(imfs=IMFs,close_idx=close_idx,train_slice=train_slice,test_slice=test_slice,seq_len=MEMD_MLP_SEQ_LEN,epochs=MEMD_MLP_EPOCHS,hidden_size=MEMD_MLP_HIDDEN,learning_rate=MEMD_MLP_LR,use_observed_update=HF_USE_OBSERVED_WALK_FORWARD_UPDATE,target_step=FORECAST_TRAIN_TARGET_STEP,output_horizon=FORECAST_OUTPUT_HORIZON,)
        if ablation_memd_mlp_norm is not None:
            ablation_memd_mlp_norm = np.asarray(ablation_memd_mlp_norm, dtype=float) + residue_memd_mlp_forecast

        # GARCH-LSTM ablation: LSTM on raw (MEMD-normalized) Close + GARCH vol feature (same idea as HF IMF branch)
        close_train_vec = np.asarray(data_for_memd[close_idx, train_slice], dtype=float).ravel()
        garch_feat_close = fit_garch_volatility_feature(close_train=close_train_vec,horizon=horizon,order=GARCH_ORDER,rescale=GARCH_RESCALE,)
        if T_train > seq_len:
            Xsg, ytg = [], []
            gl_step = int(max(1, FORECAST_TRAIN_TARGET_STEP))
            gl_out = int(max(1, FORECAST_OUTPUT_HORIZON))
            n_gl_pairs = T_train - seq_len - max(gl_step, gl_out) + 1
            for t in range(max(0, n_gl_pairs)):
                cw = data_for_memd[close_idx, t : t + seq_len].reshape(seq_len, 1)
                gf = garch_feat_close[t : t + seq_len].reshape(seq_len, 1)
                Xsg.append(np.concatenate([cw, gf], axis=1))
                if gl_out > 1:
                    ytg.append(
                        np.asarray(
                            data_for_memd[close_idx, t + seq_len : t + seq_len + gl_out],
                            dtype=float,
                        ).reshape(gl_out, 1)
                    )
                else:
                    ytg.append([[float(data_for_memd[close_idx, t + seq_len + gl_step - 1])]])
            glc_cache_key = f"{ticker}_garch_lstm_ablation_sl{seq_len}_ep{HF_EPOCHS}_h{HF_HIDDEN_SIZE}"
            model_glc = _try_load_model(LSTM, glc_cache_key)
            if model_glc is None:
                model_glc = LSTM(2, HF_HIDDEN_SIZE, int(gl_out))
                for _epoch in range(HF_EPOCHS):
                    for i in range(len(Xsg)):
                        model_glc.train_step(np.asarray(Xsg[i], dtype=float), np.asarray(ytg[i], dtype=float))
                _try_save_model(model_glc, glc_cache_key)
            preds_glc = walk_forward_close_garch_lstm(model_glc,
                np.asarray(data_for_memd[close_idx, :], dtype=float).ravel(),T_train,horizon,seq_len,garch_feat_close,HF_USE_OBSERVED_WALK_FORWARD_UPDATE,output_horizon=FORECAST_OUTPUT_HORIZON,)
            preds_glc = np.asarray(preds_glc, dtype=float).ravel()
            if preds_glc.size < horizon:
                pad_v = float(preds_glc[-1]) if preds_glc.size > 0 else 0.0
                preds_glc = np.concatenate([preds_glc, np.full(horizon - preds_glc.size, pad_v, dtype=float)])
            ablation_garch_lstm_norm = preds_glc

        # ARIMA-LSTM ablation: same ARIMA+LSTM(residual) stack as low-IMF path, applied to raw Close (no IMF split)
        arima_lstm_close_cache_key = (
            f"{ticker}_low_{str(RES_MODEL_TYPE).upper()}_close_arima_ablation_sl{RES_SEQ_LEN}_"
            f"ep{RES_EPOCHS}_h{RES_HIDDEN_SIZE}_lr{RES_LR}"
        )
        out_arima_lstm_close = forecast_arima_lstm_low_imf(series_train=np.asarray(data_for_memd[close_idx, train_slice], dtype=float).ravel(),series_test=np.asarray(data_for_memd[close_idx, test_slice], dtype=float).ravel(),res_seq_len=RES_SEQ_LEN,res_hidden_size=RES_HIDDEN_SIZE,res_epochs=RES_EPOCHS,res_lr=RES_LR,res_model_type=RES_MODEL_TYPE,skip_lstm_if_iid=SKIP_LSTM_IF_IID,forecast_days_ahead=FORECAST_DAYS_AHEAD,residual_target_step=FORECAST_TRAIN_TARGET_STEP,output_horizon=FORECAST_OUTPUT_HORIZON,imf_label=f"{ticker} close (ARIMA-LSTM ablation)",residual_model_cache_path=_scoped_model_cache_path(arima_lstm_close_cache_key),force_retrain_model=FORCE_RETRAIN_MODELS,save_trained_model=SAVE_TRAINED_MODELS,)
        ablation_arima_lstm_norm = np.asarray(out_arima_lstm_close["combined"], dtype=float).ravel()
    else:
        print(f"{ticker}: skipping expensive ablation models (RQ1_ABLATION_TICKER={RQ1_ABLATION_TICKER}).")

    low_sum = np.sum(low_imf_forecasts_array, axis=0) if low_imf_forecasts_array.size > 0 else np.zeros(horizon, dtype=float)
    hf_sum = np.sum(garch_lstm_forecasts, axis=0) if garch_lstm_forecasts.size > 0 else np.zeros(horizon, dtype=float)
    close_forecast = combine_hybrid_pipeline(low_sum, hf_sum, residue_hybrid_forecast)
    Hm = int(max(1, FORECAST_DAYS_AHEAD))
    low_sum_hmat = (
        np.sum(low_imf_hmats_array, axis=0)
        if low_imf_hmats_array.size > 0
        else np.zeros((horizon, Hm), dtype=float)
    )
    hf_sum_hmat = (
        np.sum(np.asarray(garch_lstm_hmats, dtype=float), axis=0)
        if len(garch_lstm_hmats) > 0
        else np.zeros((horizon, Hm), dtype=float)
    )

    if residue_hybrid_hmat.ndim == 1:
        residue_hybrid_hmat = residue_hybrid_hmat.reshape(-1, 1)
    if residue_hybrid_hmat.shape[1] < Hm:
        pad_cols = Hm - residue_hybrid_hmat.shape[1]
        residue_hybrid_hmat = np.concatenate(
            [residue_hybrid_hmat, np.repeat(residue_hybrid_hmat[:, [-1]], pad_cols, axis=1)],
            axis=1,
        )
    close_forecast_hmat = low_sum_hmat[:, :Hm] + hf_sum_hmat[:, :Hm] + residue_hybrid_hmat[:, :Hm]
    actual_close_test = data_for_memd[close_idx, test_slice]
    close_min = float(prep_meta["min_vals"][close_idx])
    close_range = float(prep_meta["ranges"][close_idx])
    raw_close_full = prep_meta["raw_dataframe"]["Close"].to_numpy(dtype=float)
    actual_close_test_usd = raw_close_full[test_slice]
    close_forecast_usd = close_forecast * close_range + close_min
    close_forecast_usd_hmat = close_forecast_hmat * close_range + close_min

    mse = float(np.mean((actual_close_test - close_forecast) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((actual_close_test - close_forecast) ** 2)
    ss_tot = np.sum((actual_close_test - actual_close_test.mean()) ** 2) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)

    print(f"{ticker} -> R^2={r2:.4f}, RMSE={rmse:.6f}")

    plt.figure(figsize=(12, 4))
    plt.plot(idx[test_slice],actual_close_test,label=f"{ticker} actual",color="black",linewidth=1.0,linestyle="--",alpha=0.95,)
    plt.plot(idx[test_slice], close_forecast, label=f"{ticker} forecast", color="red", alpha=0.7)
    plt.title(f"{ticker} close forecast vs actual | R²={r2:.3f}")
    plt.xlabel("Date")
    plt.ylabel("Normalized value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(_results_path(f"{ticker}_close_forecast_vs_actual.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # Standardized eval plot from evaluation module
    evaluation_graphs.forecasting_error_LSTM_Hybrid(
        forecast=close_forecast,
        real_value=actual_close_test,
        save_path=_results_path(f"{ticker}_eval_forecast_error.png"),
        show=True,
    )

    rq1_metrics_by_model = None
    if RQ1_PLOTS:
        idx_t = idx[test_slice]
        test_usd = actual_close_test_usd
        H = len(test_usd)
        train_usd = raw_close_full[:T_train]
        rq1_fc = {"Hybrid (pipeline)": close_forecast_usd}

        try:
            arima_fc, _ = forecast_arima_standalone(np.asarray(train_usd, dtype=float), H)
            arima_fc = np.asarray(arima_fc, dtype=float).reshape(-1)
            if arima_fc.size == H and np.all(np.isfinite(arima_fc)):
                rq1_fc["ARIMA (standalone)"] = arima_fc
        except Exception:
            pass

        lstm_norm = _baseline_lstm_walk_forward_close(data_for_memd[close_idx, train_slice],data_for_memd[close_idx, test_slice],BASELINE_LSTM_SEQ_LEN,BASELINE_LSTM_EPOCHS,BASELINE_LSTM_HIDDEN,learning_rate=BASELINE_LSTM_LR,model_cache_key=f"{ticker}_baseline_lstm_sl{BASELINE_LSTM_SEQ_LEN}_ep{BASELINE_LSTM_EPOCHS}_h{BASELINE_LSTM_HIDDEN}_lr{BASELINE_LSTM_LR}",use_observed_update=HF_USE_OBSERVED_WALK_FORWARD_UPDATE,target_step=FORECAST_TRAIN_TARGET_STEP,output_horizon=FORECAST_OUTPUT_HORIZON,)
        if lstm_norm is not None:
            rq1_fc["LSTM (close only)"] = lstm_norm * close_range + close_min

        mlp_norm = _baseline_mlp_walk_forward_close(data_for_memd[close_idx, train_slice],data_for_memd[close_idx, test_slice],BASELINE_MLP_SEQ_LEN,BASELINE_MLP_EPOCHS,BASELINE_MLP_HIDDEN,learning_rate=BASELINE_MLP_LR,model_cache_key=f"{ticker}_baseline_mlp_sl{BASELINE_MLP_SEQ_LEN}_ep{BASELINE_MLP_EPOCHS}_h{BASELINE_MLP_HIDDEN}_lr{BASELINE_MLP_LR}",use_observed_update=HF_USE_OBSERVED_WALK_FORWARD_UPDATE,target_step=FORECAST_TRAIN_TARGET_STEP,output_horizon=FORECAST_OUTPUT_HORIZON,)
        if mlp_norm is not None:
            rq1_fc["MLP (close only)"] = mlp_norm * close_range + close_min

        if ablation_memd_lstm_norm is not None:
            ml = np.asarray(ablation_memd_lstm_norm, dtype=float).ravel()
            if ml.size == H and np.all(np.isfinite(ml)):
                rq1_fc["MEMD-LSTM"] = ml * close_range + close_min

        if ablation_memd_mlp_norm is not None:
            mm = np.asarray(ablation_memd_mlp_norm, dtype=float).ravel()
            if mm.size == H and np.all(np.isfinite(mm)):
                rq1_fc["MEMD-MLP"] = mm * close_range + close_min

        if ablation_garch_lstm_norm is not None:
            gl = np.asarray(ablation_garch_lstm_norm, dtype=float).ravel()
            if gl.size == H and np.all(np.isfinite(gl)):
                rq1_fc["GARCH-LSTM"] = gl * close_range + close_min

        agl = np.asarray(ablation_arima_lstm_norm, dtype=float).ravel()
        if agl.size == H and np.all(np.isfinite(agl)):
            rq1_fc["ARIMA-LSTM"] = agl * close_range + close_min

        hml = np.asarray(hybrid_mlp_norm, dtype=float).ravel()
        if hml.size == H and np.all(np.isfinite(hml)):
            rq1_fc["MEMD-ARIMA-GARCH-MLP"] = hml * close_range + close_min

        rq1_fc = {
            k: np.asarray(v, dtype=float).ravel()
            for k, v in rq1_fc.items()
            if v is not None and len(np.asarray(v, dtype=float).ravel()) == H
        }
        rq1_metrics_by_model = {
            name: evaluation_graphs.compute_forecast_error_metrics(test_usd, series)
            for name, series in rq1_fc.items()
        }
        ablation_param_table = {
            "ARIMA (standalone)": {
                "selection": "auto-ARIMA",
                "max_p/max_q": "3/3",
                "seasonal": False,
            },
            "LSTM (close only)": {
                "seq_len": BASELINE_LSTM_SEQ_LEN,
                "epochs": BASELINE_LSTM_EPOCHS,
                "hidden": BASELINE_LSTM_HIDDEN,
                "lr": BASELINE_LSTM_LR,
            },
            "MLP (close only)": {
                "seq_len": BASELINE_MLP_SEQ_LEN,
                "epochs": BASELINE_MLP_EPOCHS,
                "hidden": BASELINE_MLP_HIDDEN,
                "lr": BASELINE_MLP_LR,
            },
            "MEMD-LSTM": {
                "seq_len": MEMD_LSTM_SEQ_LEN,
                "epochs": MEMD_LSTM_EPOCHS,
                "hidden": MEMD_LSTM_HIDDEN,
                "residue": "ARIMA-LSTM forecasted residue",
            },
            "MEMD-MLP": {
                "seq_len": MEMD_MLP_SEQ_LEN,
                "epochs": MEMD_MLP_EPOCHS,
                "hidden": MEMD_MLP_HIDDEN,
                "lr": MEMD_MLP_LR,
                "residue": "ARIMA-MLP forecasted residue",
            },
            "GARCH-LSTM": {
                "seq_len": HF_SEQ_LEN,
                "epochs": HF_EPOCHS,
                "hidden": HF_HIDDEN_SIZE,
                "garch_order": GARCH_ORDER,
                "garch_rescale": GARCH_RESCALE,
            },
            "ARIMA-LSTM": {
                "res_seq_len": RES_SEQ_LEN,
                "res_epochs": RES_EPOCHS,
                "res_hidden": RES_HIDDEN_SIZE,
                "res_lr": RES_LR,
                "res_model_type": RES_MODEL_TYPE,
            },
            "MEMD-ARIMA-GARCH-MLP": {
                "low_res_seq_len": HYBRID_MLP_RES_SEQ_LEN,
                "low_res_epochs": HYBRID_MLP_RES_EPOCHS,
                "low_res_hidden": HYBRID_MLP_RES_HIDDEN,
                "low_res_lr": HYBRID_MLP_RES_LR,
                "hf_seq_len": HYBRID_MLP_HF_SEQ_LEN,
                "hf_epochs": HYBRID_MLP_HF_EPOCHS,
                "hf_hidden": HYBRID_MLP_HF_HIDDEN,
                "hf_lr": HYBRID_MLP_HF_LR,
                "garch_order": GARCH_ORDER,
            },
            "Hybrid (pipeline)": {
                "low_res_seq_len": RES_SEQ_LEN,
                "low_res_epochs": RES_EPOCHS,
                "low_res_hidden": RES_HIDDEN_SIZE,
                "low_res_lr": RES_LR,
                "low_res_model": RES_MODEL_TYPE,
                "hf_seq_len": HF_SEQ_LEN,
                "hf_epochs": HF_EPOCHS,
                "hf_hidden": HF_HIDDEN_SIZE,
                "hf_model": HF_MODEL_TYPE,
                "garch_order": GARCH_ORDER,
                "garch_rescale": GARCH_RESCALE,
                "residue": "ARIMA-LSTM forecasted residue",
            },
        }
        if run_ablation_for_ticker:
            ab_order_for_table = [m for m in RQ2_ABLATION_ORDER if m in ablation_param_table]
            evaluation_graphs.rq1_ablation_parameter_table({m: ablation_param_table[m] for m in ab_order_for_table},title=f"{ticker}: ablation model parameter settings",save_path=_results_path(f"{ticker}_rq1_ablation_parameters_table.png"),show=True,)

        evaluation_graphs.rq1_accuracy_timeseries(test_usd,rq1_fc,x=idx_t,title=f"{ticker}: actual vs forecasts (USD, test)",ylabel="Close (USD)",save_path=_results_path(f"{ticker}_rq1_timeseries_usd.png"),show=True,)
        
        if run_ablation_for_ticker:
            scatter_names = list(rq1_fc.keys())
            if rq1_metrics_by_model is not None and len(rq1_metrics_by_model) > 0:
                scatter_names = sorted(
                    [n for n in rq1_fc.keys() if n in rq1_metrics_by_model],
                    key=lambda n: float(rq1_metrics_by_model[n]["R2"]),
                    reverse=True,
                )
            top_k = int(max(1, RQ1_TOP_SCATTER_MODELS))
            for name in scatter_names[:top_k]:
                series = rq1_fc[name]
                safe = "".join(c if c.isalnum() else "_" for c in name)
                evaluation_graphs.rq1_prediction_scatter(
                    test_usd,
                    series,
                    model_name=name,
                    save_path=_results_path(f"{ticker}_rq1_scatter_{safe}.png"),
                    show=True,
                )
        
        if rq1_metrics_by_model is not None and len(rq1_metrics_by_model) >= 2:
            evaluation_graphs.rq1_metric_table(
                rq1_metrics_by_model,
                title=f"{ticker}: ablation forecasting metrics",
                save_path=_results_path(f"{ticker}_rq1_metrics_table.png"),
                show=True,
            )
        if RQ1_SIGNIFICANCE_TESTS and "Hybrid (pipeline)" in rq1_fc and len(rq1_fc) >= 2:
            dm_rows = []
            comp_dict = {k: v for k, v in rq1_fc.items() if k != "Hybrid (pipeline)"}
            for comp_name, comp_fc in comp_dict.items():
                for loss_name in ("mse", "mae"):
                    dm = evaluation_graphs.diebold_mariano_test(actual=test_usd,pred_ref=rq1_fc["Hybrid (pipeline)"],pred_cmp=comp_fc,loss=loss_name,max_lag=RQ1_DM_MAX_LAG,)
                    dm_rows.append(
                        {
                            "loss": loss_name.upper(),
                            "model": comp_name,
                            "mean_diff": dm["mean_diff"],
                            "dm_stat": dm["dm_stat"],
                            "p_ref_better": dm["p_ref_better"],
                        }
                    )
            spa_rows = []
            for loss_name in ("mse", "mae"):
                spa = evaluation_graphs.superior_predictive_ability_test(actual=test_usd,pred_ref=rq1_fc["Hybrid (pipeline)"],preds_comp_by_name=comp_dict,loss=loss_name,n_boot=RQ1_SPA_BOOT,block_len=RQ1_SPA_BLOCK_LEN,random_seed=RQ1_SPA_SEED,)
                spa_rows.append(
                    {
                        "loss": loss_name.upper(),
                        "spa_stat": spa["spa_stat"],
                        "p_value": spa["p_value"],
                        "n_models": spa["n_models"],
                    }
                )
            evaluation_graphs.rq1_significance_table(dm_rows,spa_rows,title=f"{ticker}: DM pairwise + SPA (reference: Hybrid)",save_path=_results_path(f"{ticker}_rq1_significance_tests.png"),show=True,)

    if RQ2_PLOTS:
        if RQ2_DIRECTION_PLOT_3D and str(ticker).upper() == str(RQ2_DIRECTION_PLOT_TICKER).upper():
            plot_direction_hypersphere_3d(n_channels=data_for_memd.shape[0],n_directions=MEMD_N_DIRECTIONS,project_dims=(0, 1, 2),show_interpolated_lines=True,title=f"{ticker}: MEMD direction vectors (3D projection)",save_path=_results_path(f"{ticker}_rq2_memd_direction_hypersphere_3d.png"),show=True,)
        evaluation_graphs.rq2_memd_decomposition(idx,data_for_memd[close_idx],IMFs,residue[close_idx],channel_label=f"{ticker} Close (normalized)",imfs_channel_idx=close_idx,save_path=_results_path(f"{ticker}_rq2_memd_decomposition.png"),show=True,)
        
        if RQ2_MEMD_ALL_CHANNELS and str(ticker).upper() == str(RQ2_MEMD_ALL_CHANNELS_TICKER).upper():
            for ch_idx, ch_name in enumerate(channel_names):
                safe_ch = "".join(c if c.isalnum() else "_" for c in str(ch_name))
                evaluation_graphs.rq2_memd_decomposition(idx,data_for_memd[ch_idx],IMFs,residue[ch_idx],channel_label=f"{ticker} {ch_name} (normalized)",imfs_channel_idx=ch_idx,save_path=_results_path(f"{ticker}_rq2_memd_decomposition_{safe_ch}.png"),show=True,)

        if run_ablation_for_ticker and rq1_metrics_by_model:
            ab_order = [m for m in RQ2_ABLATION_ORDER if m in rq1_metrics_by_model]
            if len(ab_order) >= 2:evaluation_graphs.rq2_ablation_sequential_heatmap(ab_order,rq1_metrics_by_model,title=f"{ticker}: marginal gain vs previous benchmark",save_path=_results_path(f"{ticker}_rq2_ablation_heatmap.png"),show=True,)

        if len(low_freq_indices) > 0 and len(low_imf_outputs) > 0:
            k0 = low_freq_indices[0]
            o0 = low_imf_outputs[k0]
            ser_test = IMFs[k0, close_idx, test_slice]
            stages = {
                f"Low IMF {k0 + 1}: train post-ARIMA residual": o0["train_resid"],
                f"Low IMF {k0 + 1}: test ARIMA-only error": np.asarray(ser_test, dtype=float) - np.asarray(o0["arima_only"], dtype=float),
                f"Low IMF {k0 + 1}: test after ARIMA+LSTM": np.asarray(ser_test, dtype=float) - np.asarray(o0["combined"], dtype=float),
            }
            try:
                evaluation_graphs.rq2_residual_acf_pacf(stages,nlags=RQ2_ACF_NLAGS,title=f"{ticker}: residual ACF / PACF (low-frequency IMF path)",save_path=_results_path(f"{ticker}_rq2_residual_acf_pacf.png"),show=True,)
            except Exception as _e:
                print(f"{ticker} ACF/PACF skipped: {_e}")

    return {
        "ticker": ticker,
        "r2": r2,
        "rmse": rmse,
        "idx_test": idx[test_slice],
        "actual_test": actual_close_test,
        "forecast_test": close_forecast,
        "actual_test_usd": actual_close_test_usd,
        "forecast_test_usd": close_forecast_usd,
        "forecast_test_usd_hmat": close_forecast_usd_hmat,
        "rq1_metrics_by_model": rq1_metrics_by_model,
        "hybrid_metrics_usd": evaluation_graphs.compute_forecast_error_metrics(actual_close_test_usd, close_forecast_usd),
        "train_close_usd": np.asarray(raw_close_full[:T_train], dtype=float).copy(),
    }


# Function to align the portfolio inputs
def _aligned_portfolio_inputs(results):
    if len(results) == 0:
        return None
    L = min(len(r["actual_test_usd"]) for r in results)
    actual_mat = np.column_stack([r["actual_test_usd"][:L] for r in results])
    forecast_mat = np.column_stack([r["forecast_test_usd"][:L] for r in results])
    has_hmat = all(("forecast_test_usd_hmat" in r and r["forecast_test_usd_hmat"] is not None) for r in results)
    idx_ref = results[0]["idx_test"][:L]
    real_ret = actual_mat[1:] / (actual_mat[:-1] + 1e-12) - 1.0
    exp_ret = forecast_mat[1:] / (forecast_mat[:-1] + 1e-12) - 1.0
    exp_ret_horizon = None
    if has_hmat:
        # Horizon matrices are created via recursive 1-step rollout.
        # Use the widest rollout available (covers RQ4 horizon grid).
        Hout = int(max(1, FORECAST_DAYS_AHEAD))
        Tm1 = max(0, L - 1)
        n_assets = len(results)
        exp_ret_horizon = np.zeros((Tm1, Hout, n_assets), dtype=float)
        for a, r in enumerate(results):
            hmat = np.asarray(r["forecast_test_usd_hmat"][:L], dtype=float)
            if hmat.ndim == 1:
                hmat = hmat.reshape(-1, 1)
            if hmat.shape[1] < Hout:
                pad_v = hmat[:, [-1]] if hmat.shape[1] > 0 else np.zeros((hmat.shape[0], 1), dtype=float)
                hmat = np.concatenate([hmat, np.repeat(pad_v, Hout - hmat.shape[1], axis=1)], axis=1)
            for t in range(Tm1):
                prev = float(actual_mat[t, a])
                for k in range(Hout):
                    p_k = float(hmat[t, k])
                    exp_ret_horizon[t, k, a] = p_k / (prev + 1e-12) - 1.0
                    prev = p_k
    return {
        "L": L,
        "actual_mat": actual_mat,
        "forecast_mat": forecast_mat,
        "idx": idx_ref,
        "real_ret": real_ret,
        "exp_ret": exp_ret,
        "exp_ret_horizon": exp_ret_horizon,
    }


# Function to build the multi-asset curves
def _build_multi_asset_curves(results, initial_cash=1000.0, mpc=None):
    aligned = _aligned_portfolio_inputs(results)
    if aligned is None:
        return None

    L = aligned["L"]
    actual_mat = aligned["actual_mat"]
    forecast_mat = aligned["forecast_mat"]
    idx_ref = aligned["idx"]
    n_assets = actual_mat.shape[1]
    real_ret = aligned["real_ret"]

    if mpc is None:
        mpc = TradingMPC()
    exp_ret_horizon = aligned.get("exp_ret_horizon")
    if exp_ret_horizon is not None:
        mpc_out = run_mpc_backtest_with_horizon_returns(real_ret=aligned["real_ret"],exp_ret_horizon=exp_ret_horizon,mpc=mpc,initial_cash=initial_cash,)
    else:
        mpc_out = run_mpc_backtest(
            actual_mat,
            forecast_mat,
            mpc,
            initial_cash=initial_cash,
        )
    dyn_w = mpc_out["weights"]
    dyn_step_ret = mpc_out["step_ret"]
    dyn_curve = mpc_out["equity"]

    # Buy-and-hold:
    p0 = actual_mat[0] + 1e-12
    shares_bh = (initial_cash / n_assets) / p0
    bh_curve = np.sum(shares_bh.reshape(1, -1) * actual_mat, axis=1)

    # Equal-weight
    w_eq = np.full(n_assets, 1.0 / n_assets, dtype=float)
    ew_curve = np.zeros(L, dtype=float)
    ew_curve[0] = initial_cash
    for t in range(L - 1):
        r_p = float(np.sum(w_eq * real_ret[t]))
        ew_curve[t + 1] = ew_curve[t] * (1.0 + r_p)

    bh_step_ret = bh_curve[1:] / (bh_curve[:-1] + 1e-12) - 1.0
    ew_step_ret = ew_curve[1:] / (ew_curve[:-1] + 1e-12) - 1.0

    markowitz_curve = None
    mvo_step_ret = None
    markowitz_weights = None
    if all("train_close_usd" in r for r in results):
        try:
            Ltr = min(len(r["train_close_usd"]) for r in results)
            train_mat = np.column_stack([r["train_close_usd"][:Ltr] for r in results])
            train_r = train_mat[1:] / (train_mat[:-1] + 1e-12) - 1.0
            mu_hat = np.mean(train_r, axis=0)
            sigma_hat = np.cov(train_r, rowvar=False)
            markowitz_weights = evaluation_graphs.markowitz_longonly_max_sharpe_weights(mu_hat, sigma_hat)
            mvo_curve = np.zeros(L, dtype=float)
            mvo_curve[0] = initial_cash
            mvo_step_ret = np.zeros(L - 1, dtype=float)
            for t in range(L - 1):
                mvo_step_ret[t] = float(np.sum(markowitz_weights * real_ret[t]))
                mvo_curve[t + 1] = mvo_curve[t] * (1.0 + mvo_step_ret[t])
            markowitz_curve = mvo_curve
        except Exception as ex:
            print(f"Markowitz MVO baseline skipped: {ex}")
            markowitz_curve = None
            mvo_step_ret = None
            markowitz_weights = None

    return {
        "idx": idx_ref,
        "tickers": [r["ticker"] for r in results],
        "dynamic_curve": dyn_curve,
        "dynamic_step_ret": dyn_step_ret,
        "dynamic_weights": dyn_w,
        "buy_hold_curve": bh_curve,
        "equal_weight_curve": ew_curve,
        "buy_hold_step_ret": bh_step_ret,
        "equal_weight_step_ret": ew_step_ret,
        "markowitz_curve": markowitz_curve,
        "markowitz_step_ret": mvo_step_ret,
        "markowitz_weights": markowitz_weights,
        "test_asset_returns": real_ret,
    }


def _run_mpc_with_aligned(aligned_inputs, mpc, initial_cash):
    exp_ret_horizon = aligned_inputs.get("exp_ret_horizon")
    if exp_ret_horizon is not None:
        return run_mpc_backtest_with_horizon_returns(real_ret=aligned_inputs["real_ret"],exp_ret_horizon=exp_ret_horizon,mpc=mpc,initial_cash=initial_cash,)
    return run_mpc_backtest(aligned_inputs["actual_mat"],aligned_inputs["forecast_mat"],mpc,initial_cash=initial_cash,)


# Main function
if __name__ == "__main__":
    _script_t0 = time.perf_counter()
    _script_start_epoch = time.time()
    print(
        "Run config | "
        f"backend={'TF' if USE_TF_MODELS else 'NumPy'} | "
        f"forecast_mode={str(FORECAST_HORIZON_TRAINING_MODE).lower()} | "
        f"forecast_days_ahead={int(FORECAST_DAYS_AHEAD)} | "
        f"train_target_step={int(FORECAST_TRAIN_TARGET_STEP)} | "
        f"output_horizon={int(FORECAST_OUTPUT_HORIZON)} | "
        f"mpc_horizon={int(MPC_HORIZON)}"
    )
    _orig_show = plt.show
    if not SHOW_PLOTS_DURING_RUN:
        plt.show = lambda *args, **kwargs: None
    try:
        # Run the single ticker for each ticker in the ticker list
        summary = []
        for tkr in ticker_list:
            try:
                summary.append(run_single_ticker(tkr))
            except Exception as e:
                print(f"{tkr} failed: {e}")
                print(traceback.format_exc())

        # Print the summary of the single ticker for each ticker in the ticker list
        if len(summary) > 0:
            print("\n===== Multi-ticker summary =====")
            for row in summary:
                print(f"{row['ticker']}: R^2={row['r2']:.4f}, RMSE={row['rmse']:.6f}")
            try:
                ticker_metric_map = {}
                for row in summary:
                    metrics = row.get("hybrid_metrics_usd")
                    if metrics is None:
                        metrics = evaluation_graphs.compute_forecast_error_metrics(
                            row["actual_test_usd"], row["forecast_test_usd"]
                        )
                    ticker_metric_map[str(row["ticker"])] = metrics
                evaluation_graphs.rq1_ticker_metric_table(
                    ticker_metric_map,
                    title="RQ1: Per-ticker forecasting metrics (Hybrid)",
                    save_path=_results_path("rq1_ticker_metrics_table.png"),
                    show=True,
                )
            except Exception as ex:
                print(f"Per-ticker RQ1 metrics table skipped: {ex}")


            # Build the MPC model
            mpc = TradingMPC(horizon=MPC_HORIZON,risk_aversion=MPC_RISK_AVERSION,trade_cost=MPC_TRADE_COST,linear_cost=MPC_LINEAR_COST,impact_cost=MPC_IMPACT_COST,fixed_ticket_cost=MPC_FIXED_TICKET_COST,fixed_trade_epsilon=MPC_FIXED_TRADE_EPS,max_turnover_l1=MPC_MAX_TURNOVER_L1,min_trade_threshold=MPC_MIN_TRADE_THRESHOLD,)
            
            # Build the multi-asset curves
            port = _build_multi_asset_curves(summary, initial_cash=MULTI_INITIAL_CASH, mpc=mpc,)

            if port is not None:
                # Get the index of the portfolio
                idx_port = port["idx"]
                # Get the dynamic curve of the portfolio
                dyn_curve = port["dynamic_curve"]
                # Get the tickers of the portfolio
                tickers = port["tickers"]
                dyn_w = port["dynamic_weights"]
                # Get the buy-hold curve of the portfolio
                bh_curve = port["buy_hold_curve"]
                # Get the equal-weight curve of the portfolio
                ew_curve = port["equal_weight_curve"]
                # Get the main label of the portfolio
                main_label = "MPC"

                # Plot the cumulative portfolio return
                evaluation_graphs.cumulative_portfolio_return({main_label: dyn_curve,"Buy & hold": bh_curve,"Equal weight": ew_curve,},save_path=_results_path("multi_eval_cumulative_portfolio_return.png"),show=True,)

                # Plot the allocation plot
                evaluation_graphs.allocation_plot(stocks_shares=dyn_w,ETF_shares=None,labels=tickers,x=idx_port[:-1],save_path=_results_path("multi_eval_allocation_plot.png"),show=True,)

                # Plot the allocation concentration plot
                evaluation_graphs.allocation_concentration_plot(dyn_w,labels=tickers,x=idx_port[:-1],save_path=_results_path("multi_eval_allocation_holdings_by_ticker.png"),show=True,)
                evaluation_graphs.allocation_stackplot_top_k_other(dyn_w,labels=tickers,top_k=5,x=idx_port[:-1],save_path=_results_path("multi_eval_allocation_top5_other.png"),show=True,)
                # Plot the RQ3 cumulative wealth curves
                if RQ3_PLOTS:
                    mvo_curve = port.get("markowitz_curve")
                    mpc_objective_outputs = {
                        "MPC (mean-variance utility)": {
                            "equity": dyn_curve,
                            "step_ret": port["dynamic_step_ret"],
                            "weights": dyn_w,
                        }
                    }
                    curves_rq3 = {
                        "MPC (mean-variance utility)": dyn_curve,
                        "Equal weight": ew_curve,
                        "Buy & hold": bh_curve,
                    }
                    if RQ3_COMPARE_MPC_OBJECTIVES:
                        aligned_rq3 = _aligned_portfolio_inputs(summary)
                        if aligned_rq3 is not None:
                            for obj_mode, lbl in [
                                ("terminal_wealth", "MPC (terminal wealth)"),
                                ("sharpe_like", "MPC (Sharpe-like objective)"),
                            ]:
                                mpc_obj = TradingMPC(horizon=MPC_HORIZON,risk_aversion=MPC_RISK_AVERSION,trade_cost=MPC_TRADE_COST,objective_mode=obj_mode,linear_cost=MPC_LINEAR_COST,impact_cost=MPC_IMPACT_COST,fixed_ticket_cost=MPC_FIXED_TICKET_COST,fixed_trade_epsilon=MPC_FIXED_TRADE_EPS,max_turnover_l1=MPC_MAX_TURNOVER_L1,min_trade_threshold=MPC_MIN_TRADE_THRESHOLD,)
                                out_obj = _run_mpc_with_aligned(aligned_rq3, mpc_obj, initial_cash=MULTI_INITIAL_CASH)
                                curves_rq3[lbl] = out_obj["equity"]
                                mpc_objective_outputs[lbl] = out_obj

                    # Per-objective allocation plots (weights over time)
                    for lbl, out_obj in mpc_objective_outputs.items():
                        w_obj = out_obj.get("weights")
                        if w_obj is None:
                            continue
                        safe_lbl = "".join(
                            ch.lower() if ch.isalnum() else "_"
                            for ch in lbl.replace("(", "").replace(")", "")
                        ).strip("_")
                        evaluation_graphs.allocation_plot(stocks_shares=w_obj,ETF_shares=None,labels=tickers,x=idx_port[:-1],save_path=_results_path(f"multi_eval_allocation_{safe_lbl}.png"),show=True,)
                        evaluation_graphs.allocation_concentration_plot(w_obj,labels=tickers,x=idx_port[:-1],save_path=_results_path(f"multi_eval_allocation_holdings_{safe_lbl}.png"),show=True,)
                        evaluation_graphs.allocation_stackplot_top_k_other(w_obj,labels=tickers,top_k=5,x=idx_port[:-1],save_path=_results_path(f"multi_eval_allocation_top5_other_{safe_lbl}.png"),show=True,)

                    if mvo_curve is not None:
                        curves_rq3["Static Markowitz MVO"] = mvo_curve
                    # Plot the RQ3 cumulative wealth curves
                    evaluation_graphs.rq3_cumulative_wealth_curves(curves_rq3,x=idx_port,title="RQ3: Cumulative wealth — MPC objective variants vs EW, B&H, static Markowitz",save_path=_results_path("rq3_cumulative_wealth.png"),show=True,)
                    # Plot the RQ3 bootstrap terminal profit histogram
                    step_dict = {
                        "MPC (mean-variance utility)": port["dynamic_step_ret"],
                        "Equal weight": port["equal_weight_step_ret"],
                        "Buy & hold": port["buy_hold_step_ret"],
                    }
                    for lbl, out_obj in mpc_objective_outputs.items():
                        if lbl == "MPC (mean-variance utility)":
                            continue
                        if "step_ret" in out_obj:
                            step_dict[lbl] = out_obj["step_ret"]
                    mvo_step = port.get("markowitz_step_ret")
                    if mvo_step is not None:
                        step_dict["Static Markowitz MVO"] = mvo_step
                    # Plot the RQ3 bootstrap terminal profit histogram
                    evaluation_graphs.rq3_bootstrap_terminal_profit_histogram(step_dict,initial_cash=MULTI_INITIAL_CASH,n_sims=RQ3_BOOTSTRAP_SIMS,random_seed=RQ3_BOOTSTRAP_SEED,save_path=_results_path("rq3_bootstrap_terminal_profit.png"),show=True,)

                if RQ4_PLOTS:
                    # Plot the RQ4 transaction cost sensitivity lines
                    aligned_rq4 = _aligned_portfolio_inputs(summary)
                    if aligned_rq4 is not None:
                        rr = aligned_rq4["real_ret"]
                        er = aligned_rq4["exp_ret"]
                        erh = aligned_rq4.get("exp_ret_horizon")
                        Tm1 = int(rr.shape[0])

                        def _rq4_cost_sweep(x_grid, mode):
                            total_rets_loc, sharpes_loc = [], []
                            for x in x_grid:
                                if mode == "linear":
                                    c_lin = float(MPC_LINEAR_COST) * float(x) / max(float(RQ4_BPS_FOR_BASELINE_PENALTY), 1e-9)
                                    c_imp = MPC_IMPACT_COST
                                    c_fix = MPC_FIXED_TICKET_COST
                                elif mode == "impact":
                                    c_lin = MPC_LINEAR_COST
                                    base_imp = MPC_TRADE_COST if MPC_IMPACT_COST is None else MPC_IMPACT_COST
                                    c_imp = float(base_imp) * float(x)
                                    c_fix = MPC_FIXED_TICKET_COST
                                elif mode == "fixed":
                                    c_lin = MPC_LINEAR_COST
                                    c_imp = MPC_IMPACT_COST
                                    c_fix = float(MPC_FIXED_TICKET_COST) * float(x)
                                else:
                                    raise ValueError(f"Unknown sweep mode: {mode}")
                                mpc_s = TradingMPC(horizon=MPC_HORIZON,risk_aversion=MPC_RISK_AVERSION,trade_cost=MPC_TRADE_COST,linear_cost=c_lin,impact_cost=c_imp,fixed_ticket_cost=c_fix,fixed_trade_epsilon=MPC_FIXED_TRADE_EPS,max_turnover_l1=MPC_MAX_TURNOVER_L1,min_trade_threshold=MPC_MIN_TRADE_THRESHOLD,)
                                out_s = _run_mpc_with_aligned(aligned_rq4, mpc_s, initial_cash=MULTI_INITIAL_CASH)
                                eqs = out_s["equity"]
                                total_rets_loc.append(float(eqs[-1] / eqs[0] - 1.0))
                                sharpes_loc.append(
                                    evaluation_graphs.annualized_sharpe_step_returns(
                                        out_s["step_ret"], ann_factor=RQ4_ANN_FACTOR
                                    )
                                )
                            return np.asarray(total_rets_loc, dtype=float), np.asarray(sharpes_loc, dtype=float)

                        # Linear-cost sweep (main friction curve for three-component model)
                        total_rets, sharpes = _rq4_cost_sweep(RQ4_COST_BPS_GRID, mode="linear")
                        foot_rq4 = (
                            f"Linear cost c_lin(bps) = c0×(bps/{RQ4_BPS_FOR_BASELINE_PENALTY:.0f}), c0={MPC_LINEAR_COST:g}; "
                            f"impact={MPC_TRADE_COST if MPC_IMPACT_COST is None else MPC_IMPACT_COST:g}, "
                            f"fixed_ticket={MPC_FIXED_TICKET_COST:g}, eps={MPC_FIXED_TRADE_EPS:g}."
                        )
                        evaluation_graphs.rq4_transaction_cost_sensitivity_lines(RQ4_COST_BPS_GRID,total_rets,sharpes,title="RQ4: Sensitivity to linear transaction cost",xlabel="Linear-cost index (basis points; mapped to c_lin)",footnote=foot_rq4,save_path=_results_path("rq4_transaction_cost_sensitivity.png"),show=True,)

                        # Impact-cost sweep (quadratic market-impact coefficient)
                        impact_rets, impact_sharpes = _rq4_cost_sweep(RQ4_IMPACT_SCALE_GRID, mode="impact")
                        evaluation_graphs.rq4_transaction_cost_sensitivity_lines(RQ4_IMPACT_SCALE_GRID,impact_rets,impact_sharpes,title="RQ4: Sensitivity to impact-cost coefficient",xlabel="Impact-cost multiplier (x baseline)",
                            footnote=(
                                f"impact = impact0 × multiplier, impact0={MPC_TRADE_COST if MPC_IMPACT_COST is None else MPC_IMPACT_COST:g}; "
                                f"linear={MPC_LINEAR_COST:g}, fixed_ticket={MPC_FIXED_TICKET_COST:g}, eps={MPC_FIXED_TRADE_EPS:g}."
                            ),save_path=_results_path("rq4_impact_cost_sensitivity.png"),show=True,)

                        # Fixed-ticket sweep (per-trade fixed fee, normalized by current equity)
                        fixed_rets, fixed_sharpes = _rq4_cost_sweep(RQ4_FIXED_SCALE_GRID, mode="fixed")
                        evaluation_graphs.rq4_transaction_cost_sensitivity_lines(RQ4_FIXED_SCALE_GRID,fixed_rets,fixed_sharpes,title="RQ4: Sensitivity to fixed ticket cost",xlabel="Fixed-ticket multiplier (x baseline)",
                            footnote=(
                                f"fixed_ticket = fixed0 × multiplier, fixed0={MPC_FIXED_TICKET_COST:g}; "
                                f"linear={MPC_LINEAR_COST:g}, impact={MPC_TRADE_COST if MPC_IMPACT_COST is None else MPC_IMPACT_COST:g}, "
                                f"ticket threshold eps={MPC_FIXED_TRADE_EPS:g}."
                            ),save_path=_results_path("rq4_fixed_ticket_sensitivity.png"),show=True,)

                        # Build full strategy set for drawdown statistics table
                        dd_curves_all = {
                            f"{main_label} (mean-variance utility)": dyn_curve,
                            "Buy & hold": bh_curve,
                            "Equal weight": ew_curve,
                        }
                        if RQ3_COMPARE_MPC_OBJECTIVES:
                            for obj_mode, lbl in [
                                ("terminal_wealth", "MPC (terminal wealth)"),
                                ("sharpe_like", "MPC (Sharpe-like objective)"),
                            ]:
                                mpc_dd = TradingMPC(horizon=MPC_HORIZON,risk_aversion=MPC_RISK_AVERSION,trade_cost=MPC_TRADE_COST,objective_mode=obj_mode,linear_cost=MPC_LINEAR_COST,impact_cost=MPC_IMPACT_COST,fixed_ticket_cost=MPC_FIXED_TICKET_COST,fixed_trade_epsilon=MPC_FIXED_TRADE_EPS,max_turnover_l1=MPC_MAX_TURNOVER_L1,min_trade_threshold=MPC_MIN_TRADE_THRESHOLD,)
                                out_dd = _run_mpc_with_aligned(aligned_rq4, mpc_dd, initial_cash=MULTI_INITIAL_CASH)
                                dd_curves_all[lbl] = out_dd["equity"]
                        mvo_dd = port.get("markowitz_curve")
                        if mvo_dd is not None:
                            dd_curves_all["Static Markowitz MVO"] = mvo_dd

                        # Plot-only: show hybrid line only for readability.
                        dd_curves_plot = {f"{main_label} (mean-variance utility)": dyn_curve}
                        evaluation_graphs.rq4_underwater_drawdown(
                            dd_curves_plot,
                            x=idx_port,
                            title="Drawdown from running peak (out-of-sample) — Hybrid",
                            save_path=_results_path("rq4_underwater_drawdown.png"),
                            show=True,
                        )
                        evaluation_graphs.rq4_drawdown_stats_table(
                            dd_curves_all,
                            title="RQ4 drawdown statistics across strategies",
                            save_path=_results_path("rq4_underwater_drawdown_stats.png"),
                            show=True,
                        )

                        # Plot the RQ4 horizon profit violin (for each MPC objective variant)
                        horizon_objectives = [("mean_variance", "mean_variance")]
                        if RQ3_COMPARE_MPC_OBJECTIVES:
                            horizon_objectives.extend(
                                [("terminal_wealth", "terminal_wealth"), ("sharpe_like", "sharpe_like")]
                            )
                        for obj_i, (obj_mode, obj_tag) in enumerate(horizon_objectives):
                            rng_h = np.random.default_rng(int(RQ4_HORIZON_SEED) + obj_i)
                            profits_by_h = {}
                            for H in RQ4_HORIZON_GRID:
                                h_int = int(H)
                                if h_int > max(1, Tm1 - 1):
                                    continue
                                mpc_h = TradingMPC(horizon=h_int,risk_aversion=MPC_RISK_AVERSION,trade_cost=MPC_TRADE_COST,objective_mode=obj_mode,linear_cost=MPC_LINEAR_COST,impact_cost=MPC_IMPACT_COST,fixed_ticket_cost=MPC_FIXED_TICKET_COST,fixed_trade_epsilon=MPC_FIXED_TRADE_EPS,max_turnover_l1=MPC_MAX_TURNOVER_L1,min_trade_threshold=MPC_MIN_TRADE_THRESHOLD,)
                                block_len = max(h_int * int(RQ4_HORIZON_BLOCK_MULT), 10)
                                finals = []
                                for _ in range(int(RQ4_HORIZON_SIMS)):
                                    if erh is not None:
                                        rrb, erhb = evaluation_graphs.block_bootstrap_horizon_return_paths(
                                            rr, erh, rng_h, block_len, n_steps=Tm1
                                        )
                                        oh = run_mpc_backtest_with_horizon_returns(
                                            real_ret=rrb,
                                            exp_ret_horizon=erhb,
                                            mpc=mpc_h,
                                            initial_cash=MULTI_INITIAL_CASH,
                                        )
                                    else:
                                        rrb, erb = evaluation_graphs.block_bootstrap_return_paths(
                                            rr, er, rng_h, block_len, n_steps=Tm1
                                        )
                                        oh = run_mpc_backtest_returns(rrb, erb, mpc_h, initial_cash=MULTI_INITIAL_CASH)
                                    finals.append(float(oh["equity"][-1] - MULTI_INITIAL_CASH))
                                profits_by_h[f"H = {h_int}"] = np.asarray(finals, dtype=float)
                            if len(profits_by_h) > 0:
                                save_name = ("rq4_horizon_profit_violin.png"if obj_mode == "mean_variance"else f"rq4_horizon_profit_violin_{obj_tag}.png")
                                evaluation_graphs.rq4_horizon_profit_violin(profits_by_h,title=f"Distribution of terminal profit by MPC horizon ({obj_tag})",highlight_horizon=MPC_HORIZON,save_path=_results_path(save_name),show=True,)

    finally:
        plt.show = _orig_show
        _elapsed = time.perf_counter() - _script_t0
        _m, _s = divmod(_elapsed, 60.0)
        if _m >= 1.0:
            print(f"\n===== Total wall time: {int(_m)} min {_s:.1f} s ({_elapsed:.0f} s) =====")
        else:
            print(f"\n===== Total wall time: {_elapsed:.1f} s =====")
