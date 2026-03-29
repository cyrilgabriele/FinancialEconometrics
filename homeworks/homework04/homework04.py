import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.stats.diagnostic import acorr_ljungbox


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:
    - parse date columns
    - sort by date
    - reset index
    """
    cleaned_df = df.copy()

    date_cols = [col for col in cleaned_df.columns if "date" in col.lower()]
    for col in date_cols:
        cleaned_df[col] = pd.to_datetime(
            cleaned_df[col],
            yearfirst=True,
            errors="coerce",
        )

    if "date" in cleaned_df.columns:
        cleaned_df = cleaned_df.sort_values("date").reset_index(drop=True)

    for col in ["hitec", "mkt", "rf"]:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

    return cleaned_df


def exercise_0_exess_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Homework 4 preprocessing.

    IMPORTANT
    ---------
    s4_data.txt already contains DAILY SIMPLE RETURNS IN PERCENT.
    Therefore we do NOT use pct_change().

    We compute LOG EXCESS RETURNS IN PERCENT:
        hitec_log_excess = 100 * [log(1 + hitec/100) - log(1 + rf/100)]
        mkt_log_excess   = 100 * [log(1 + mkt/100)   - log(1 + rf/100)]

    The returned dataframe includes the columns:
        - hitec_log_excess
        - mkt_log_excess
        - hitec_log_excess_sq
    """
    required_cols = ["date", "hitec", "mkt", "rf"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Simple returns in decimal form
    hitec_simple = df["hitec"] / 100.0
    mkt_simple = df["mkt"] / 100.0
    rf_simple = df["rf"] / 100.0

    # Log excess returns, expressed in percent
    df["hitec_log_excess"] = 100.0 * (np.log1p(hitec_simple) - np.log1p(rf_simple))
    df["mkt_log_excess"] = 100.0 * (np.log1p(mkt_simple) - np.log1p(rf_simple))

    # Useful for ARCH-type exercises
    df["hitec_log_excess_sq"] = df["hitec_log_excess"] ** 2

    return df


def _numerical_hessian(func, theta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Numerical Hessian of a scalar-valued function using central differences.
    """
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    hess = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = eps
            ej[j] = eps

            if i == j:
                f_plus = func(theta + ei)
                f_0 = func(theta)
                f_minus = func(theta - ei)
                hess[i, i] = (f_plus - 2.0 * f_0 + f_minus) / (eps ** 2)
            else:
                f_pp = func(theta + ei + ej)
                f_pm = func(theta + ei - ej)
                f_mp = func(theta - ei + ej)
                f_mm = func(theta - ei - ej)
                hess[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps ** 2)

    return hess


def exercise_1_MLE(df: pd.DataFrame) -> dict:
    """
    Question 1:
    Estimate by MLE the model

        y_t = beta0 + beta1 * x_t + u_t
        u_t ~ Laplace(0, s)

    where:
        y_t = hitec log excess return (in percent)
        x_t = market log excess return (in percent)

    Returns
    -------
    dict with:
        - beta0
        - beta1
        - s
        - se_beta0
        - se_beta1
        - se_s
        - loglik
        - nobs
        - convergence
        - data    : estimation sample with fitted values and residuals
        - summary : compact result table
    """
    work = df[["date", "hitec_log_excess", "mkt_log_excess"]].dropna().copy()
    work = work.rename(
        columns={
            "hitec_log_excess": "y",
            "mkt_log_excess": "x",
        }
    ).reset_index(drop=True)

    y = work["y"].to_numpy(dtype=float)
    x = work["x"].to_numpy(dtype=float)
    n = len(work)

    if n == 0:
        raise ValueError("No valid observations available after preprocessing.")

    # OLS starting values
    X = np.column_stack([np.ones(n), x])
    beta_init = np.linalg.lstsq(X, y, rcond=None)[0]

    def neg_loglik(theta: np.ndarray) -> float:
        beta0, beta1, log_s = theta
        s = np.exp(log_s)
        u = y - beta0 - beta1 * x
        # Negative log-likelihood for Laplace(0, s)
        return n * np.log(2.0 * s) + np.sum(np.abs(u) / s)

    def profiled_neg_loglik(beta: np.ndarray) -> float:
        beta0, beta1 = beta
        u = y - beta0 - beta1 * x
        s = np.mean(np.abs(u))

        if not np.isfinite(s) or s <= 0:
            return np.inf

        # For Laplace errors, the MLE of s conditional on beta is mean(|u|).
        return n * (1.0 + np.log(2.0 * s))

    opt = minimize(
        profiled_neg_loglik,
        beta_init,
        method="Powell",
        options={"disp": False, "maxiter": 10000},
    )

    if not opt.success:
        raise RuntimeError(f"Optimization failed: {opt.message}")

    beta0_hat, beta1_hat = opt.x
    residuals = y - beta0_hat - beta1_hat * x
    s_hat = np.mean(np.abs(residuals))

    if not np.isfinite(s_hat) or s_hat <= 0:
        raise RuntimeError("Estimated Laplace scale parameter is not positive.")

    log_s_hat = np.log(s_hat)
    theta_hat = np.array([beta0_hat, beta1_hat, log_s_hat], dtype=float)

    # Standard errors from inverse Hessian of the negative log-likelihood
    hess = _numerical_hessian(neg_loglik, theta_hat, eps=1e-5)
    cov_theta = np.linalg.pinv(hess)

    se_beta0 = np.sqrt(max(cov_theta[0, 0], 0.0))
    se_beta1 = np.sqrt(max(cov_theta[1, 1], 0.0))
    se_log_s = np.sqrt(max(cov_theta[2, 2], 0.0))
    se_s = s_hat * se_log_s  # delta method

    fitted = beta0_hat + beta1_hat * x
    residuals = y - fitted
    loglik = -neg_loglik(theta_hat)

    work["fitted"] = fitted
    work["residual"] = residuals
    work["abs_residual"] = np.abs(residuals)

    summary = pd.DataFrame(
        {
            "parameter": ["beta0", "beta1", "s"],
            "estimate": [beta0_hat, beta1_hat, s_hat],
            "std_error": [se_beta0, se_beta1, se_s],
        }
    )

    return {
        "beta0": beta0_hat,
        "beta1": beta1_hat,
        "s": s_hat,
        "se_beta0": se_beta0,
        "se_beta1": se_beta1,
        "se_s": se_s,
        "loglik": loglik,
        "nobs": n,
        "convergence": opt.success,
        "data": work,
        "summary": summary,
    }


def exercise_2_ljung_box_squared_hitec(df: pd.DataFrame, lags: int = 12) -> dict:
    """
    Question 2:
    Perform a Ljung-Box test with 12 lags on the squared returns of hitec
    to test for ARCH effects.

    IMPORTANT
    ---------
    For this question, use the raw hitec returns directly from s4_data.txt.
    The file already contains daily simple returns in percent, so we do NOT
    use exercise_0_exess_returns() here.

    H0: no autocorrelation in squared hitec returns up to the chosen lag.
    """
    work = df.copy()

    required_cols = ["date", "hitec"]
    missing = [col for col in required_cols if col not in work.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work["hitec"] = pd.to_numeric(work["hitec"], errors="coerce")
    work = work[["date", "hitec"]].dropna().sort_values("date").reset_index(drop=True)

    # Squared raw returns of hitec
    work["hitec_sq"] = work["hitec"] ** 2
    test_series = work["hitec_sq"]

    if len(test_series) <= lags:
        raise ValueError(
            f"Not enough observations for Ljung-Box test with {lags} lags."
        )

    lb = acorr_ljungbox(test_series, lags=[lags], return_df=True)

    lb_stat = float(lb["lb_stat"].iloc[0])
    p_value = float(lb["lb_pvalue"].iloc[0])

    alpha = 0.05
    reject_h0 = p_value < alpha

    if reject_h0:
        conclusion = (
            "Reject H0 at the 5% significance level: squared hitec returns "
            "display autocorrelation, which indicates ARCH effects."
        )
    else:
        conclusion = (
            "Do not reject H0 at the 5% significance level: no evidence of "
            "ARCH effects in squared hitec returns."
        )

    summary = pd.DataFrame(
        {
            "test": ["Ljung-Box"],
            "series": ["hitec^2"],
            "lags": [lags],
            "lb_stat": [lb_stat],
            "p_value": [p_value],
            "alpha": [alpha],
            "reject_h0": [reject_h0],
        }
    )

    return {
        "lb_stat": lb_stat,
        "p_value": p_value,
        "lags": lags,
        "alpha": alpha,
        "reject_h0": reject_h0,
        "conclusion": conclusion,
        "tested_series": test_series,
        "summary": summary,
    }


def exercise_3_ARCH4_model1(df: pd.DataFrame, make_qq_plot: bool = True) -> dict:
    """
    Question 3:
    Estimate an ARCH(4) model on raw hitec returns, assuming normal innovations.

    Model 1:
        r_t = mu + u_t
        u_t = sigma_t * z_t,   z_t ~ N(0,1)
        sigma_t^2 = omega + alpha1*u_{t-1}^2 + alpha2*u_{t-2}^2
                          + alpha3*u_{t-3}^2 + alpha4*u_{t-4}^2

    Notes
    -----
    - This uses RAW hitec returns from s4_data.txt (not excess returns).
    - ARCH(4) is estimated as a constant-mean volatility model.
    - Positivity and finite unconditional variance are imposed via parameter
      transformation:
          omega > 0, alpha_i >= 0, sum(alpha_i) < 1

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe containing at least ['date', 'hitec'].
    make_qq_plot : bool
        If True, display a QQ plot of standardized residuals.

    Returns
    -------
    dict with:
        - mu
        - omega
        - alpha1, alpha2, alpha3, alpha4
        - persistence
        - loglik
        - nobs
        - convergence
        - data       : dataframe with residuals, sigma2, sigma, std_resid
        - summary    : compact parameter table
        - opt_result : scipy optimization result
    """
    work = df.copy()

    required_cols = ["date", "hitec"]
    missing = [col for col in required_cols if col not in work.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work = work[["date", "hitec"]].dropna().sort_values("date").reset_index(drop=True)

    y = work["hitec"].to_numpy(dtype=float)
    n = len(y)

    if n <= 10:
        raise ValueError("Not enough observations to estimate an ARCH(4) model.")

    # We condition on the first 4 observations
    start_idx = 4


    def unpack_params(theta: np.ndarray):
        """
        Transform unconstrained parameters into:
            mu (real),
            omega > 0,
            alpha_i >= 0,
            sum(alpha_i) < 1
        """
        mu = theta[0]
        omega = np.exp(theta[1])

        exp_a = np.exp(theta[2:6])
        denom = 1.0 + np.sum(exp_a)
        alphas = exp_a / denom  # each >= 0 and total < 1

        return mu, omega, alphas


    def compute_sigma2(theta: np.ndarray):
        mu, omega, alphas = unpack_params(theta)
        resid = y - mu

        sigma2 = np.full(n, np.var(y, ddof=1), dtype=float)

        # unconditional variance implied by the model
        persistence = np.sum(alphas)
        unc_var = omega / (1.0 - persistence)

        if np.isfinite(unc_var) and unc_var > 0:
            sigma2[:start_idx] = unc_var
        else:
            sigma2[:start_idx] = np.var(y, ddof=1)

        for t in range(start_idx, n):
            sigma2[t] = (
                omega
                + alphas[0] * resid[t - 1] ** 2
                + alphas[1] * resid[t - 2] ** 2
                + alphas[2] * resid[t - 3] ** 2
                + alphas[3] * resid[t - 4] ** 2
            )

            if not np.isfinite(sigma2[t]) or sigma2[t] <= 0:
                return None, None, None, None

        return resid, sigma2, omega, alphas


    def neg_loglik(theta: np.ndarray) -> float:
        mu, omega, alphas = unpack_params(theta)
        resid, sigma2, _, _ = compute_sigma2(theta)

        if resid is None:
            return 1e12

        ll = -0.5 * np.sum(
            np.log(2.0 * np.pi)
            + np.log(sigma2[start_idx:])
            + (resid[start_idx:] ** 2) / sigma2[start_idx:]
        )

        if not np.isfinite(ll):
            return 1e12

        return -ll

    # Starting values
    mu0 = float(np.mean(y))
    var0 = float(np.var(y, ddof=1))

    omega0 = max(0.10 * var0, 1e-6)
    alpha0 = np.array([0.20, 0.10, 0.05, 0.05], dtype=float)  # sum < 1
    base = 1.0 - np.sum(alpha0)

    theta0 = np.array(
        [
            mu0,
            np.log(omega0),
            np.log(alpha0[0] / base),
            np.log(alpha0[1] / base),
            np.log(alpha0[2] / base),
            np.log(alpha0[3] / base),
        ],
        dtype=float,
    )

    opt = minimize(
        neg_loglik,
        theta0,
        method="Powell",
        options={"disp": False, "maxiter": 20000},
    )

    if not opt.success:
        raise RuntimeError(f"Optimization failed: {opt.message}")

    mu_hat, omega_hat, alpha_hat = unpack_params(opt.x)
    resid_hat, sigma2_hat, _, _ = compute_sigma2(opt.x)

    sigma_hat = np.sqrt(sigma2_hat)
    std_resid = resid_hat / sigma_hat
    loglik = -neg_loglik(opt.x)

    out = work.copy()
    out["residual"] = resid_hat
    out["sigma2"] = sigma2_hat
    out["sigma"] = sigma_hat
    out["std_resid"] = std_resid

    summary = pd.DataFrame(
        {
            "parameter": ["mu", "omega", "alpha1", "alpha2", "alpha3", "alpha4"],
            "estimate": [
                mu_hat,
                omega_hat,
                alpha_hat[0],
                alpha_hat[1],
                alpha_hat[2],
                alpha_hat[3],
            ],
        }
    )

    if make_qq_plot:
        import matplotlib.pyplot as plt
        from scipy.stats import probplot

        fig, ax = plt.subplots(figsize=(6, 6))
        probplot(std_resid[start_idx:], dist="norm", plot=ax)
        ax.set_title("QQ plot of standardized residuals: ARCH(4), normal")
        ax.grid(True, alpha=0.3)
        plt.show()

    return {
        "mu": mu_hat,
        "omega": omega_hat,
        "alpha1": alpha_hat[0],
        "alpha2": alpha_hat[1],
        "alpha3": alpha_hat[2],
        "alpha4": alpha_hat[3],
        "persistence": float(np.sum(alpha_hat)),
        "loglik": loglik,
        "nobs": n,
        "convergence": opt.success,
        "data": out,
        "summary": summary,
        "opt_result": opt,
    }