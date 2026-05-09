import numpy as np
import pandas as pd
import yfinance as yf
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Forward-fill at most this many consecutive missing rows per OHLCV gap, longer gaps stay NaN and rows are dropped.
OHLCV_FFILL_MAX_CONSECUTIVE = 3


# Downloading OHLCV data from Yahoo Finance and returning a normalized multivariate array for MEMD.
def download_and_prepare_data(ticker="AAPL", period="10y", return_metadata=False, ffill_limit=OHLCV_FFILL_MAX_CONSECUTIVE):
    #  These are the channel names
    channel_names = ["Open", "High", "Low", "Close", "Volume"]

    # yfinance may return empty frames, this just retrys once with a simpler call.
    stock = yf.download(ticker, period=period, progress=False, auto_adjust=True, threads=False)
    if stock is None or stock.empty:
        stock = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True, threads=False)
    if stock is None or stock.empty:
        raise RuntimeError(
            f"Failed to download data for ticker='{ticker}', period='{period}'. "
            "Yahoo returned no rows."
        )
    # Sort the stock by index
    stock = stock.sort_index()

    # yfinance often returns MultiIndex columns, e.g. ('Close', 'JPM'); code expects flat 'Close', etc.
    if isinstance(stock.columns, pd.MultiIndex) and stock.columns.nlevels > 1:
        stock = stock.copy()
        stock.columns = stock.columns.droplevel(-1)

    # Check if the stock is missing any columns
    missing = [c for c in channel_names if c not in stock.columns]
    # If missing, raise an error
    if missing:
        raise RuntimeError(
            f"Downloaded data is missing required columns: {missing}. "
            f"Available columns: {list(stock.columns)}"
        )

    # Carry last valid OHLCV forward across short gaps only (set to 3 by default)
    if ffill_limit is not None and int(ffill_limit) > 0:
        stock[channel_names] = stock[channel_names].ffill(limit=int(ffill_limit))

    # Rows still NaN (gap longer than ffill_limit, or leading NaNs) are dropped
    stock = stock.dropna(subset=channel_names)
    # Check if the stock is empty
    if stock.empty:
        raise RuntimeError(
            f"Downloaded data for ticker='{ticker}' is empty after dropna()."
        )

    multivariate = stock[channel_names].values

    # Find the minimum and maximum values for each channel
    min_vals = multivariate.min(axis=0)
    max_vals = multivariate.max(axis=0)
    # Find the range for each channel
    ranges = max_vals - min_vals
    ranges[ranges == 0.0] = 1.0
    # Normalize the multivariate data
    multivariate_norm = (multivariate - min_vals) / ranges
    # Transpose the multivariate data
    data_for_memd = multivariate_norm.T

    # If return_metadata is True, return the metadata
    if return_metadata:
        # Create the metadata
        metadata = {
            "raw_dataframe": stock[channel_names].copy(),
            "min_vals": min_vals.copy(),
            "max_vals": max_vals.copy(),
            "ranges": ranges.copy(),
        }
        return data_for_memd, stock.index, channel_names, metadata

    return data_for_memd, stock.index, channel_names

# Finding the indices of local maxima and minima in the 1D signal.
def find_local_extrema_1d(signal):
    maxima = []
    minima = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            maxima.append(i)
        if signal[i] < signal[i - 1] and signal[i] < signal[i + 1]:
            minima.append(i)
    return np.array(maxima, dtype=int), np.array(minima, dtype=int)

# Interpolate one envelope from given extrema indices and values
def interpolate_envelope(indices, values, n_samples, method="linear"):
    # Check if the number of indices is less than 2
    if len(indices) < 2:
        return np.zeros(n_samples, dtype=float)

    # Convert the indices to integers
    idx = indices.astype(int)
    # Convert the values to floats
    vals = values.astype(float)

    # Ensure left boundary is 0
    if idx[0] != 0:
        idx = np.concatenate(([0], idx))
        vals = np.concatenate(([vals[0]], vals))
    
    # Ensure right boundary is the last index
    if idx[-1] != n_samples - 1:
        idx = np.concatenate((idx, [n_samples - 1]))
        vals = np.concatenate((vals, [vals[-1]]))

    # Choose interpolation kind (cubic being more smooth as it uses cubic splines)
    if method == "cubic" and len(idx) >= 4:
        kind = "cubic"
    else:
        kind = "linear"

    f = interp1d(idx, vals, kind=kind, bounds_error=False, fill_value="nearest")
    return f(np.arange(n_samples))

# 1D low-discrepancy sequence helper used for Hammersley points.
def van_der_corput(n, base):
    vdc = 0.0
    denom = 1.0
    while n:
        n, remainder = divmod(n, base)
        denom *= base
        vdc += remainder / denom
    return vdc

# Generate direction vectors using a Hammersley point set mapped onto the unit hypersphere.
def generate_direction_vectors(n_channels, n_directions):
    # Simple list of primes for van der Corput dimensions
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    if n_channels - 1 > len(primes):
        raise ValueError("Not enough primes defined for the requested number of channels.")

    # Build Hammersley points in [0, 1]^n_channels
    h_points = np.zeros((n_directions, n_channels))
    for i in range(n_directions):
        m = i + 1
        h_points[i, 0] = m / float(n_directions)
        for dim in range(1, n_channels):
            h_points[i, dim] = van_der_corput(m, primes[dim - 1])

    # Center around 0 and normalize to unit length (hypersphere)
    directions = h_points - 0.5
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    directions = directions / norms
    return directions


def plot_direction_hypersphere_3d(n_channels,n_directions=512,project_dims=(0, 1, 2),show_interpolated_lines=True,interp_bands=7,interp_points_per_band=220,title=None,save_path=None,show=True):
    if len(project_dims) != 3:
        raise ValueError("project_dims must contain exactly 3 indices.")
    dirs = generate_direction_vectors(int(n_channels), int(n_directions))
    pdims = tuple(int(d) for d in project_dims)
    if max(pdims) >= int(n_channels) or min(pdims) < 0:
        raise ValueError("project_dims indices must be within [0, n_channels).")

    pts = dirs[:, pdims]
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    pts = pts / norms

    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0.0, 2.0 * np.pi, 40)
    v = np.linspace(0.0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="#d9d9d9", alpha=0.14, linewidth=0)

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=9, alpha=0.8, c="tab:blue", depthshade=True)
    if bool(show_interpolated_lines):
        z = pts[:, 2]
        n_bands = int(max(1, interp_bands))
        edges = np.quantile(z, np.linspace(0.0, 1.0, n_bands + 1))
        for bi in range(n_bands):
            lo, hi = float(edges[bi]), float(edges[bi + 1])
            if bi == n_bands - 1:
                mask = (z >= lo) & (z <= hi)
            else:
                mask = (z >= lo) & (z < hi)
            band = pts[mask]
            if band.shape[0] < 10:
                continue
            phi = np.arctan2(band[:, 1], band[:, 0])
            order = np.argsort(phi)
            phi_s = phi[order]
            xyz_s = band[order]

            # Build periodic interpolation on angle and resample a smooth closed curve.
            phi_ext = np.concatenate([phi_s - 2.0 * np.pi, phi_s, phi_s + 2.0 * np.pi])
            x_ext = np.concatenate([xyz_s[:, 0], xyz_s[:, 0], xyz_s[:, 0]])
            y_ext = np.concatenate([xyz_s[:, 1], xyz_s[:, 1], xyz_s[:, 1]])
            z_ext = np.concatenate([xyz_s[:, 2], xyz_s[:, 2], xyz_s[:, 2]])
            phi_u = np.linspace(-np.pi, np.pi, int(max(40, interp_points_per_band)))
            x_u = np.interp(phi_u, phi_ext, x_ext)
            y_u = np.interp(phi_u, phi_ext, y_ext)
            z_u = np.interp(phi_u, phi_ext, z_ext)
            nr = np.sqrt(x_u * x_u + y_u * y_u + z_u * z_u) + 1e-12
            x_u, y_u, z_u = x_u / nr, y_u / nr, z_u / nr
            ax.plot(x_u, y_u, z_u, color="tab:orange", linewidth=0.9, alpha=0.55)

    ax.set_xlabel(f"dim {pdims[0] + 1}")
    ax.set_ylabel(f"dim {pdims[1] + 1}")
    ax.set_zlabel(f"dim {pdims[2] + 1}")
    t = title or f"MEMD direction set projected to 3D (n={n_channels}, dirs={n_directions})"
    ax.set_title(t)
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=170, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

# Multivariate Empirical Mode Decomposition (simplified).
def memd(data, max_imfs=None, max_sift_iter=50, sd_threshold=0.2, n_directions=512, envelope_method="cubic"):
    # Get the number of channels and samples
    n_channels, n_samples = data.shape
    directions = generate_direction_vectors(n_channels, n_directions)
    # Initialize the residue
    residue = data.copy()
    imfs = []
    # Loop until the maximum number of IMFs is reached
    while True:
        if max_imfs is not None and len(imfs) >= max_imfs:
            break

        imf = residue.copy()

        for _ in range(max_sift_iter):
            # Initialize the maximum envelope
            max_env = np.zeros_like(imf)
            # Initialize the minimum envelope
            min_env = np.zeros_like(imf)
            # Initialize the valid direction count
            valid_dir_count = 0

            # For each direction, project, find extrema, and accumulate envelopes
            for d in directions:
                # Project the data onto the direction
                projection = np.dot(d, imf)
                # Find the local maxima and minima
                max_idx, min_idx = find_local_extrema_1d(projection)
                if len(max_idx) < 2 or len(min_idx) < 2:
                    continue

                valid_dir_count += 1
                # For each channel, interpolate the envelope
                for ch in range(n_channels):
                    ch_values = imf[ch]
                    max_vals = ch_values[max_idx]
                    min_vals = ch_values[min_idx]
                    # Interpolate the maximum envelope
                    max_env[ch] += interpolate_envelope(
                        max_idx, max_vals, n_samples, method=envelope_method
                    )
                    # Interpolate the minimum envelope
                    min_env[ch] += interpolate_envelope(
                        min_idx, min_vals, n_samples, method=envelope_method
                    )

            # If no valid directions, break
            if valid_dir_count == 0:
                break

            # Normalize the maximum and minimum envelopes
            max_env /= valid_dir_count
            min_env /= valid_dir_count
            # Calculate the mean envelope
            mean_env = 0.5 * (max_env + min_env)

            # Calculate the new IMF
            new_imf = imf - mean_env

            # Calculate the SD index: relative change between iterations
            num = np.sum((imf - new_imf) ** 2)
            den = np.sum(imf ** 2) + 1e-12
            sd = num / den
            # If the SD index is less than the threshold, break
            if sd < sd_threshold:
                # Update the IMF
                imf = new_imf
                break
            # Update the IMF
            imf = new_imf

        # Append the IMF to the list
        imfs.append(imf.copy())
        # Update the residue
        residue = residue - imf

        # Stopping condition on residue: monotonic in all channels
        # Initialize the monotonic all flag
        monotonic_all = True
        # Check if the residue is monotonic in all channels
        for ch in range(n_channels):
            max_idx, min_idx = find_local_extrema_1d(residue[ch])
            if len(max_idx) >= 2 and len(min_idx) >= 2:
                monotonic_all = False
                break
        # If the residue is monotonic in all channels, break
        if monotonic_all:
            break

    return imfs, residue


def plot_imfs(time_index, original_series, imfs_for_channel, residue_for_channel,
              channel_name="Close", save_path="memd_decomposition.png"):
    # Plot original series, IMFs, and final residue for one channel.
    n_imfs = len(imfs_for_channel)
    fig, axes = plt.subplots(n_imfs + 2, 1, figsize=(12, 2 * (n_imfs + 2)))
    fig.suptitle(f"MEMD decomposition - {channel_name}", fontsize=14)

    # Original signal
    axes[0].plot(time_index, original_series, "k-", linewidth=1.5)
    axes[0].set_title(f"Original {channel_name}")
    axes[0].grid(True, alpha=0.3)

    # IMFs
    for i, imf_ch in enumerate(imfs_for_channel):
        axes[i + 1].plot(time_index, imf_ch, linewidth=1)
        axes[i + 1].set_title(f"IMF {i + 1}")
        axes[i + 1].grid(True, alpha=0.3)

    # Residue at the bottom
    axes[-1].plot(time_index, residue_for_channel, "r-", linewidth=1)
    axes[-1].set_title("Residue")
    axes[-1].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
