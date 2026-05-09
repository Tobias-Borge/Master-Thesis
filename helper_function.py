import numpy as np
import os
import matplotlib.pyplot as plt


# Function to calculate the sample entropy
def sample_entropy(signal, m=2, r_ratio=0.2, step=5):
    x = np.asarray(signal, dtype=float)[::step]
    n = len(x)
    if n <= m + 2:
        return np.nan
    r = r_ratio * np.std(x)
    if r == 0.0:
        return 0.0

    def _phi(order):
        count = 0
        for i in range(n - order):
            for j in range(i + 1, n - order):
                if np.all(np.abs(x[i : i + order] - x[j : j + order]) <= r):
                    count += 1
        return count

    b = _phi(m)
    a = _phi(m + 1)
    if b == 0 or a == 0:
        return 0.0
    return -np.log(a / b)


# Function to return the results path
def results_path(results_dir, filename):
    return os.path.join(results_dir, filename)


# Function to return the model cache path
def model_cache_path(model_cache_dir, model_key):
    safe = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(model_key))
    return os.path.join(model_cache_dir, f"{safe}.npz")


# Function to return the cache context prefix
def cache_context_prefix(use_tf_models, forecast_days_ahead, forecast_mode, forecast_train_target_step):
    backend = "tf" if use_tf_models else "np"
    horizon = int(max(1, forecast_days_ahead))
    mode = str(forecast_mode).lower()
    tstep = int(max(1, forecast_train_target_step))
    return f"{backend}_fd{horizon}_{mode}_ts{tstep}"


# Function to return the scoped model key
def scoped_model_key(prefix, model_key):
    return f"{prefix}__{model_key}"


# Function to replay the generated plots after the run
def replay_generated_plots_after_run(run_start_epoch, original_show, results_dir, enabled):
    if not enabled:
        return
    try:
        files = []
        for fn in os.listdir(results_dir):
            if not fn.lower().endswith(".png"):
                continue
            fp = results_path(results_dir, fn)
            try:
                mt = os.path.getmtime(fp)
            except OSError:
                continue
            if mt >= float(run_start_epoch) - 1.0:
                files.append((mt, fp))
        files.sort(key=lambda x: x[0])
        for _mt, fp in files:
            try:
                img = plt.imread(fp)
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(img)
                ax.set_title(os.path.basename(fp))
                ax.axis("off")
                fig.tight_layout()
                original_show()
            except Exception as ex:
                print(f"Plot replay skipped for {fp}: {ex}")
    except Exception as ex:
        print(f"Plot replay skipped: {ex}")


# Function to try to load the model
def try_load_model(model_cls,model_key,*,force_retrain_models,use_tf_models,forecast_days_ahead,forecast_mode,forecast_train_target_step,model_cache_dir):
    if force_retrain_models:
        return None
    prefix = cache_context_prefix(
        use_tf_models,
        forecast_days_ahead,
        forecast_mode,
        forecast_train_target_step,
    )
    fp = model_cache_path(model_cache_dir, scoped_model_key(prefix, model_key))
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
def try_save_model(model_obj,model_key,*,save_trained_models,use_tf_models,forecast_days_ahead,forecast_mode,forecast_train_target_step,model_cache_dir,):
    if not save_trained_models:
        return
    prefix = cache_context_prefix(
        use_tf_models,
        forecast_days_ahead,
        forecast_mode,
        forecast_train_target_step,
    )
    fp = model_cache_path(model_cache_dir, scoped_model_key(prefix, model_key))
    try:
        model_obj.save(fp)
    except Exception as ex:
        print(f"Model save skipped ({fp}): {ex}")


# Function to return the aligned portfolio inputs
def aligned_portfolio_inputs(results, *, forecast_output_horizon, mpc_horizon):
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
        Hout = min(int(max(1, forecast_output_horizon)), int(max(1, mpc_horizon)))
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


