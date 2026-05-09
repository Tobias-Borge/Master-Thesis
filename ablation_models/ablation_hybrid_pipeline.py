import numpy as np


def combine_hybrid_pipeline(low_sum, hf_sum, residue_fc):
    low_sum = np.asarray(low_sum, dtype=float).ravel()
    hf_sum = np.asarray(hf_sum, dtype=float).ravel()
    residue_fc = np.asarray(residue_fc, dtype=float).ravel()
    h = min(len(low_sum), len(hf_sum), len(residue_fc))
    return low_sum[:h] + hf_sum[:h] + residue_fc[:h]

