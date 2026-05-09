import numpy as np

from .ablation_mlp import forecast_mlp_close_only


def forecast_memd_mlp_sum(imfs,close_idx,train_slice,test_slice,seq_len,epochs,hidden_size,learning_rate=5e-5,use_observed_update=True,target_step=1,output_horizon=1,):
    imfs = np.asarray(imfs, dtype=float)
    n_imfs = int(imfs.shape[0])
    h = int(test_slice.stop - test_slice.start)
    out = np.zeros(h, dtype=float)
    for k in range(n_imfs):
        tr = imfs[k, close_idx, train_slice]
        te = imfs[k, close_idx, test_slice]
        fc = forecast_mlp_close_only(
            tr,
            te,
            seq_len=seq_len,
            epochs=epochs,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            use_observed_update=use_observed_update,
            target_step=target_step,
            output_horizon=output_horizon,
        )
        if fc is None:
            return None
        out += np.asarray(fc, dtype=float).ravel()
    return out

