import numpy as np

from GARCH import fit_garch_volatility_feature, walk_forward_close_garch_lstm
from LSTM import LSTM


def forecast_garch_lstm_close(close_full,t_train,horizon,seq_len,hidden_size,epochs,garch_order=(1, 1),garch_rescale=100.0,use_observed_update=True,target_step=1,output_horizon=1,):
    close_full = np.asarray(close_full, dtype=float).ravel()
    close_train = close_full[: int(t_train)]
    garch_feat = fit_garch_volatility_feature(
        close_train=close_train,
        horizon=int(horizon),
        order=tuple(garch_order),
        rescale=float(garch_rescale),
    )
    out_h = int(max(1, output_horizon))
    t_step = int(max(1, target_step))
    n_pairs = int(t_train) - int(seq_len) - max(out_h, t_step) + 1
    if n_pairs <= 0:
        return None

    Xs, ys = [], []
    for t in range(n_pairs):
        cw = close_full[t : t + int(seq_len)].reshape(int(seq_len), 1)
        gf = garch_feat[t : t + int(seq_len)].reshape(int(seq_len), 1)
        Xs.append(np.concatenate([cw, gf], axis=1))
        if out_h > 1:
            ys.append(close_full[t + int(seq_len) : t + int(seq_len) + out_h].reshape(out_h, 1))
        else:
            ys.append([[close_full[t + int(seq_len) + t_step - 1]]])

    model = LSTM(2, int(hidden_size), int(out_h))
    for _ in range(int(epochs)):
        for i in range(len(Xs)):
            model.train_step(Xs[i], ys[i])

    fc = walk_forward_close_garch_lstm(
        model=model,
        close_channel_full=close_full,
        T_train=int(t_train),
        horizon=int(horizon),
        seq_len=int(seq_len),
        garch_feature_full=garch_feat,
        use_observed_update=bool(use_observed_update),
        output_horizon=out_h,
        return_horizon_matrix=False,
    )
    return np.asarray(fc, dtype=float).ravel()

