# Multi-asset MPC: long-only, fully invested (weights on the probability simplex).
# Tunable parameters are intended to be set in Main_multi.py (or your driver script).

import numpy as np
from scipy.optimize import minimize

# Function to project the vector onto the simplex
def euclidean_proj_simplex(v, s=1.0):
    v = np.asarray(v, dtype=float).reshape(-1)
    n = v.size
    if n == 0:
        return v
    if n == 1:
        return np.array([float(s)], dtype=float)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - float(s)
    ind = np.arange(1, n + 1, dtype=float)
    mask = u - cssv / ind > 0.0
    if not np.any(mask):
        return np.full(n, float(s) / n, dtype=float)
    rho = int(ind[mask][-1])
    theta = float(cssv[mask][-1] / rho)
    w = np.maximum(v - theta, 0.0)
    return w

# MPC model
class TradingMPC:
    # Initialize the MPC model
    def __init__(self,horizon=5,risk_aversion=0.5,trade_cost=0.05,objective_mode="mean_variance",linear_cost=0.0,impact_cost=None,fixed_ticket_cost=0.0,fixed_trade_epsilon=1e-3,max_turnover_l1=None,min_trade_threshold=1e-4):
        self.horizon = int(max(1, horizon))
        self.risk_aversion = float(risk_aversion)
        self.objective_mode = str(objective_mode).lower()
        # Backward compatibility: trade_cost maps to impact cost unless impact_cost is explicitly set.
        self.trade_cost = float(trade_cost)
        self.linear_cost = float(linear_cost)
        self.impact_cost = float(trade_cost) if impact_cost is None else float(impact_cost)
        self.fixed_ticket_cost = float(fixed_ticket_cost)
        self.fixed_trade_epsilon = float(fixed_trade_epsilon)
        self.max_turnover_l1 = None if max_turnover_l1 is None else float(max_turnover_l1)
        self.min_trade_threshold = float(min_trade_threshold)
        if (
            self.risk_aversion < 0.0
            or self.linear_cost < 0.0
            or self.impact_cost < 0.0
            or self.fixed_ticket_cost < 0.0
        ):
            raise ValueError("risk_aversion and all cost coefficients must be non-negative.")
        if self.fixed_trade_epsilon < 0.0:
            raise ValueError("fixed_trade_epsilon must be non-negative.")
        if self.max_turnover_l1 is not None and self.max_turnover_l1 <= 0.0:
            raise ValueError("max_turnover_l1 must be positive or None.")
        if self.objective_mode not in {"terminal_wealth", "mean_variance", "sharpe_like"}:
            raise ValueError("objective_mode must be one of: terminal_wealth, mean_variance, sharpe_like.")

    def solve_weights(self, exp_ret_horizon, w_prev):
        # exp_ret_horizon: (H, n_assets) forecast simple returns for H steps ahead.
        # w_prev: (n_assets,) previous weights on simplex.
        R = np.asarray(exp_ret_horizon, dtype=float)
        if R.ndim == 1:
            R = R.reshape(1, -1)
        H, n = R.shape
        w_prev = np.asarray(w_prev, dtype=float).reshape(-1)
        if w_prev.shape[0] != n:
            raise ValueError("w_prev length must match number of assets.")

        if n == 1:
            return np.ones(1, dtype=float)

        w_prev = np.clip(w_prev, 0.0, 1.0)
        s0 = w_prev.sum()
        if s0 < 1e-15:
            w_prev = np.full(n, 1.0 / n, dtype=float)
        else:
            w_prev = w_prev / s0

        r_agg = np.sum(R, axis=0)
        mu_h = np.mean(R, axis=0)
        if H >= 2:
            sigma_h = np.cov(R, rowvar=False)
            if np.ndim(sigma_h) == 0:
                sigma_h = np.array([[float(sigma_h)]], dtype=float)
        else:
            sigma_h = np.eye(n, dtype=float) * 1e-6
        sigma_h = np.asarray(sigma_h, dtype=float)
        if sigma_h.shape != (n, n):
            sigma_h = np.eye(n, dtype=float) * 1e-6
        sigma_h = sigma_h + 1e-8 * np.eye(n)

        def neg_objective(wv):
            wv = np.asarray(wv, dtype=float)
            ret_term = float(np.dot(wv, r_agg))
            var_term = float(wv @ sigma_h @ wv)
            if self.objective_mode == "terminal_wealth":
                utility_term = ret_term
            elif self.objective_mode == "sharpe_like":
                utility_term = float(np.dot(wv, mu_h)) / np.sqrt(max(var_term, 1e-12))
            else:
                # Original behavior as default utility proxy.
                utility_term = ret_term - self.risk_aversion * float(np.dot(wv, wv))
            d = wv - w_prev
            linear_term = self.linear_cost * float(np.sum(np.abs(d)))
            impact_term = self.impact_cost * float(np.dot(d, d))
            # Smooth approximation of fixed ticket count for optimizer stability.
            smooth_scale = max(self.fixed_trade_epsilon * 0.25, 1e-6)
            z = (np.abs(d) - self.fixed_trade_epsilon) / smooth_scale
            fixed_count_smooth = np.sum(1.0 / (1.0 + np.exp(-z)))
            fixed_term = self.fixed_ticket_cost * float(fixed_count_smooth)
            return -(utility_term - linear_term - impact_term - fixed_term)

        cons = ({"type": "eq", "fun": lambda wv: float(np.sum(wv) - 1.0)},)
        bounds = [(0.0, 1.0)] * n
        x0 = w_prev.copy()

        res = minimize(
            neg_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 800, "ftol": 1e-9},
        )

        if res.success:
            w_new = np.clip(np.asarray(res.x, dtype=float), 0.0, 1.0)
            s = w_new.sum()
            if s < 1e-15:
                w_new = np.full(n, 1.0 / n, dtype=float)
            else:
                w_new = w_new / s
        else:
            w_new = np.full(n, 1.0 / n, dtype=float)

        if np.max(np.abs(w_new - w_prev)) < self.min_trade_threshold:
            w_new = w_prev.copy()

        if self.max_turnover_l1 is not None:
            delta = w_new - w_prev
            l1 = float(np.sum(np.abs(delta)))
            if l1 > self.max_turnover_l1 + 1e-12 and l1 > 1e-15:
                alpha = self.max_turnover_l1 / l1
                w_cand = w_prev + alpha * delta
                w_new = euclidean_proj_simplex(w_cand, s=1.0)

        s = w_new.sum()
        if s < 1e-15:
            w_new = np.full(n, 1.0 / n, dtype=float)
        else:
            w_new = w_new / s
        return w_new


def run_mpc_backtest(actual_prices, forecast_prices, mpc, initial_cash=1000.0):
    # Backtest MPC on aligned price matrices (one column per asset).
    # actual_prices: (T, n_assets)
    # forecast_prices: (T, n_assets)
    P = np.asarray(actual_prices, dtype=float)
    F = np.asarray(forecast_prices, dtype=float)
    if P.shape != F.shape:
        raise ValueError("actual_prices and forecast_prices must have the same shape.")
    if P.ndim != 2:
        raise ValueError("Prices must be 2D (time, n_assets).")
    T, n = P.shape
    if T < 3:
        raise ValueError("Need at least 3 price rows.")

    real_ret = P[1:] / (P[:-1] + 1e-12) - 1.0
    exp_ret = F[1:] / (F[:-1] + 1e-12) - 1.0
    out = run_mpc_backtest_returns(real_ret, exp_ret, mpc, initial_cash=initial_cash)
    out["real_ret"] = real_ret
    out["exp_ret"] = exp_ret
    return out


def run_mpc_backtest_returns(real_ret, exp_ret, mpc, initial_cash=1000.0):
    # Same portfolio dynamics as run_mpc_backtest, but inputs are aligned simple-return
    # matrices (Tm1, n_assets) for realized and forecast paths — useful for frictions /
    # horizon experiments without reconstructing price levels.
    real_ret = np.asarray(real_ret, dtype=float)
    exp_ret = np.asarray(exp_ret, dtype=float)
    if real_ret.shape != exp_ret.shape:
        raise ValueError("real_ret and exp_ret must have the same shape.")
    if real_ret.ndim != 2:
        raise ValueError("Returns must be 2D (time, n_assets).")
    Tm1, n = real_ret.shape
    if Tm1 < 1:
        raise ValueError("Need at least one return row.")

    w_prev = np.full(n, 1.0 / n, dtype=float)
    weights = np.zeros((Tm1, n), dtype=float)
    step_ret = np.zeros(Tm1, dtype=float)
    step_ret_gross = np.zeros(Tm1, dtype=float)
    cost_fraction = np.zeros(Tm1, dtype=float)
    turnover = np.zeros(Tm1, dtype=float)
    equity = np.zeros(Tm1 + 1, dtype=float)
    equity[0] = float(initial_cash)

    for t in range(Tm1):
        H_eff = min(mpc.horizon, Tm1 - t)
        R_h = exp_ret[t : t + H_eff, :]
        w_new = mpc.solve_weights(R_h, w_prev)
        delta = w_new - w_prev
        turnover[t] = float(np.sum(np.abs(delta)))

        c_lin = mpc.linear_cost * float(np.sum(np.abs(delta)))
        c_imp = mpc.impact_cost * float(np.dot(delta, delta))
        n_tickets = int(np.sum(np.abs(delta) > mpc.fixed_trade_epsilon))
        c_fix = (mpc.fixed_ticket_cost * float(n_tickets)) / max(float(equity[t]), 1e-12)
        total_cost = c_lin + c_imp + c_fix
        cost_fraction[t] = total_cost

        weights[t] = w_new
        step_ret_gross[t] = float(np.sum(w_new * real_ret[t]))
        step_ret[t] = step_ret_gross[t] - total_cost
        equity[t + 1] = equity[t] * (1.0 + step_ret[t])
        w_prev = w_new

    return {
        "equity": equity,
        "weights": weights,
        "step_ret": step_ret,
        "step_ret_gross": step_ret_gross,
        "cost_fraction": cost_fraction,
        "turnover": turnover,
    }


def run_mpc_backtest_with_horizon_returns(real_ret, exp_ret_horizon, mpc, initial_cash=1000.0):
    # MPC backtest where each time t has an explicit forecast horizon matrix.
    # exp_ret_horizon: shape (Tm1, H, n_assets), with expected returns for t+1..t+H.
    real_ret = np.asarray(real_ret, dtype=float)
    exp_ret_horizon = np.asarray(exp_ret_horizon, dtype=float)
    if real_ret.ndim != 2:
        raise ValueError("real_ret must be 2D (time, n_assets).")
    if exp_ret_horizon.ndim != 3:
        raise ValueError("exp_ret_horizon must be 3D (time, horizon, n_assets).")
    Tm1, n = real_ret.shape
    if exp_ret_horizon.shape[0] != Tm1 or exp_ret_horizon.shape[2] != n:
        raise ValueError("exp_ret_horizon must align with real_ret on time and assets.")

    w_prev = np.full(n, 1.0 / n, dtype=float)
    weights = np.zeros((Tm1, n), dtype=float)
    step_ret = np.zeros(Tm1, dtype=float)
    step_ret_gross = np.zeros(Tm1, dtype=float)
    cost_fraction = np.zeros(Tm1, dtype=float)
    turnover = np.zeros(Tm1, dtype=float)
    equity = np.zeros(Tm1 + 1, dtype=float)
    equity[0] = float(initial_cash)

    for t in range(Tm1):
        H_eff = min(mpc.horizon, exp_ret_horizon.shape[1], Tm1 - t)
        R_h = exp_ret_horizon[t, :H_eff, :]
        w_new = mpc.solve_weights(R_h, w_prev)
        delta = w_new - w_prev
        turnover[t] = float(np.sum(np.abs(delta)))

        c_lin = mpc.linear_cost * float(np.sum(np.abs(delta)))
        c_imp = mpc.impact_cost * float(np.dot(delta, delta))
        n_tickets = int(np.sum(np.abs(delta) > mpc.fixed_trade_epsilon))
        c_fix = (mpc.fixed_ticket_cost * float(n_tickets)) / max(float(equity[t]), 1e-12)
        total_cost = c_lin + c_imp + c_fix
        cost_fraction[t] = total_cost

        weights[t] = w_new
        step_ret_gross[t] = float(np.sum(w_new * real_ret[t]))
        step_ret[t] = step_ret_gross[t] - total_cost
        equity[t + 1] = equity[t] * (1.0 + step_ret[t])
        w_prev = w_new

    return {
        "equity": equity,
        "weights": weights,
        "step_ret": step_ret,
        "step_ret_gross": step_ret_gross,
        "cost_fraction": cost_fraction,
        "turnover": turnover,
    }
