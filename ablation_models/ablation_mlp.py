import numpy as np

from MLP import MLP


def forecast_mlp_close_only(train_norm,test_norm,seq_len,epochs,hidden_size,learning_rate=5e-5,use_observed_update=True,target_step=1,output_horizon=1,):
    train = np.asarray(train_norm, dtype=float).ravel()
    test = np.asarray(test_norm, dtype=float).ravel()
    t_train, horizon = len(train), len(test)
    seq_len = int(seq_len)
    out_h = int(max(1, output_horizon))
    t_step = int(max(1, target_step))
    n_pairs = t_train - seq_len - max(out_h, t_step) + 1
    if n_pairs <= 0:
        return None

    model = MLP(1, int(hidden_size), int(out_h), learning_rate=float(learning_rate))
    Xs, ys = [], []
    for t in range(n_pairs):
        Xs.append(train[t : t + seq_len].reshape(seq_len, 1))
        if out_h > 1:
            ys.append(train[t + seq_len : t + seq_len + out_h].reshape(out_h, 1))
        else:
            ys.append([[train[t + seq_len + t_step - 1]]])
    for _ in range(int(epochs)):
        for i in range(len(Xs)):
            model.train_step(Xs[i], ys[i])

    full = np.concatenate([train, test]).copy()
    pred = np.zeros(horizon, dtype=float)
    for h in range(horizon):
        window = full[t_train + h - seq_len : t_train + h].reshape(seq_len, 1)
        y = (model.Why @ model.forward(window)[-1][0] + model.by).flatten()
        pred[h] = float(y[0])
        if not use_observed_update:
            full[t_train + h] = pred[h]
    return pred

