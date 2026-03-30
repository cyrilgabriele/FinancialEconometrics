import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import jarque_bera, norm, probplot
from scipy.stats import t as student_t
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], yearfirst=True, errors="coerce")
    for column in ["hitec", "mkt", "rf"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def exercise_0_exess_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional preprocessing helper.

    The sheet itself works with raw returns, but this helper keeps the earlier
    excess-return transformation available for notebook experiments.
    """
    out = clean_data(df)
    out["hitec_log_excess"] = 100.0 * (
        np.log1p(out["hitec"] / 100.0) - np.log1p(out["rf"] / 100.0)
    )
    out["mkt_log_excess"] = 100.0 * (
        np.log1p(out["mkt"] / 100.0) - np.log1p(out["rf"] / 100.0)
    )
    return out


def jarque_bera_normality_test(
    series: pd.Series | np.ndarray,
    alpha: float,
    series_name: str,
) -> dict:
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    stat, p_value = jarque_bera(values)
    reject_h0 = p_value < alpha

    if reject_h0:
        conclusion = (
            f"Reject normality for {series_name} at the {100 * alpha:.0f}% level."
        )
    else:
        conclusion = (
            f"Do not reject normality for {series_name} at the {100 * alpha:.0f}% level."
        )

    summary = pd.DataFrame(
        {
            "series": [series_name],
            "jb_stat": [float(stat)],
            "p_value": [float(p_value)],
            "alpha": [alpha],
            "reject_h0": [reject_h0],
        }
    )

    return {
        "jb_stat": float(stat),
        "p_value": float(p_value),
        "alpha": alpha,
        "reject_h0": reject_h0,
        "conclusion": conclusion,
        "summary": summary,
    }


def _numerical_hessian(func, theta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    hess = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = eps
            ej[j] = eps
            hess[i, j] = (
                func(theta + ei + ej)
                - func(theta + ei - ej)
                - func(theta - ei + ej)
                + func(theta - ei - ej)
            ) / (4.0 * eps**2)

    return hess


def _reported_covariance(func, theta_hat: np.ndarray, transform) -> np.ndarray:
    hess = _numerical_hessian(func, theta_hat)
    cov_theta = np.linalg.pinv(hess)

    reported = np.asarray(transform(theta_hat), dtype=float)
    jac = np.zeros((len(reported), len(theta_hat)), dtype=float)
    step = 1e-6

    for j in range(len(theta_hat)):
        direction = np.zeros(len(theta_hat))
        direction[j] = step
        jac[:, j] = (
            np.asarray(transform(theta_hat + direction), dtype=float)
            - np.asarray(transform(theta_hat - direction), dtype=float)
        ) / (2.0 * step)

    return jac @ cov_theta @ jac.T


def _simplex_weights(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(np.clip(logits, -50.0, 50.0))
    return exp_logits / (1.0 + np.sum(exp_logits))


def plot_qq_standardized_residuals(
    std_resid: np.ndarray,
    start_idx: int,
    distribution: str,
    title: str,
    df_t: float | None = None,
) -> None:
    values = np.asarray(std_resid, dtype=float)[start_idx:]
    values = values[np.isfinite(values)]

    fig, ax = plt.subplots(figsize=(6, 6))

    if distribution == "norm":
        probplot(values, dist="norm", plot=ax)

    elif distribution == "t":
        scale = np.sqrt((df_t - 2.0) / df_t)
        probs = (np.arange(1, len(values) + 1) - 0.5) / len(values)
        theoretical = student_t.ppf(probs, df=df_t) * scale
        sample = np.sort(values)
        slope, intercept = np.polyfit(theoretical, sample, 1)
        x_grid = np.linspace(theoretical.min(), theoretical.max(), 200)
        ax.scatter(theoretical, sample, s=12)
        ax.plot(x_grid, intercept + slope * x_grid, color="tab:red")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Ordered Values")

    else:
        raise ValueError("distribution must be 'norm' or 't'.")

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.show()


def exercise_1_MLE(df: pd.DataFrame) -> dict:
    """
    Question 1:
    Estimate

        hitec_t = beta0 + beta1 * mkt_t + u_t
        u_t ~ Laplace(0, s)

    by maximum likelihood.
    """
    work = clean_data(df)[["date", "hitec", "mkt"]].dropna().reset_index(drop=True)
    y = work["hitec"].to_numpy(dtype=float)
    x = work["mkt"].to_numpy(dtype=float)

    X = np.column_stack([np.ones(len(work)), x])
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_ols = y - X @ beta_ols
    s0 = np.mean(np.abs(resid_ols))
    theta0 = np.array([beta_ols[0], beta_ols[1], np.log(s0)], dtype=float)

    def neg_loglik(theta: np.ndarray) -> float:
        beta0, beta1, log_s = theta
        s = np.exp(log_s)
        resid = y - beta0 - beta1 * x
        value = len(y) * np.log(2.0 * s) + np.sum(np.abs(resid) / s)
        if np.isfinite(value):
            return float(value)
        return 1e12

    opt = minimize(
        neg_loglik,
        theta0,
        method="Powell",
        options={"maxiter": 10000},
    )

    beta0_hat, beta1_hat, log_s_hat = opt.x
    s_hat = float(np.exp(log_s_hat))
    fitted = beta0_hat + beta1_hat * x
    residual = y - fitted
    loglik = float(-neg_loglik(opt.x))

    def reported_params(theta: np.ndarray) -> np.ndarray:
        return np.array([theta[0], theta[1], np.exp(theta[2])], dtype=float)

    cov = _reported_covariance(neg_loglik, opt.x, reported_params)
    se_beta0, se_beta1, se_s = np.sqrt(np.maximum(np.diag(cov), 0.0))

    out = work.copy()
    out["fitted"] = fitted
    out["residual"] = residual

    summary = pd.DataFrame(
        {
            "parameter": ["beta0", "beta1", "s"],
            "estimate": [beta0_hat, beta1_hat, s_hat],
            "std_error": [se_beta0, se_beta1, se_s],
        }
    )

    return {
        "beta0": float(beta0_hat),
        "beta1": float(beta1_hat),
        "s": s_hat,
        "se_beta0": float(se_beta0),
        "se_beta1": float(se_beta1),
        "se_s": float(se_s),
        "loglik": loglik,
        "nobs": int(len(work)),
        "convergence": bool(opt.success),
        "data": out,
        "summary": summary,
    }


def exercise_2_ljung_box_squared_hitec(df: pd.DataFrame, lags: int) -> dict:
    """
    Question 2:
    Ljung-Box test with 12 lags on squared hitec returns.
    """
    work = clean_data(df)[["date", "hitec"]].dropna().reset_index(drop=True)
    work["hitec_sq"] = work["hitec"] ** 2

    lb = acorr_ljungbox(work["hitec_sq"], lags=[lags], return_df=True)
    lb_stat = float(lb["lb_stat"].iloc[0])
    p_value = float(lb["lb_pvalue"].iloc[0])
    reject_h0 = p_value < 0.05

    if reject_h0:
        conclusion = (
            "Reject H0 at the 5% level: squared hitec returns are autocorrelated, "
            "so there is evidence of ARCH effects."
        )
    else:
        conclusion = (
            "Do not reject H0 at the 5% level: there is no evidence of ARCH effects."
        )

    summary = pd.DataFrame(
        {
            "test": ["Ljung-Box"],
            "series": ["hitec^2"],
            "lags": [lags],
            "lb_stat": [lb_stat],
            "p_value": [p_value],
            "reject_h0": [reject_h0],
        }
    )

    return {
        "lb_stat": lb_stat,
        "p_value": p_value,
        "lags": lags,
        "reject_h0": reject_h0,
        "conclusion": conclusion,
        "summary": summary,
    }


def _arch4_sigma2(
    y: np.ndarray,
    mu: float,
    omega: float,
    alphas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    start_idx = 4
    resid = y - mu
    sigma2 = np.full(len(y), np.var(y, ddof=1), dtype=float)

    persistence = float(np.sum(alphas))
    unc_var = omega / (1.0 - persistence)
    if np.isfinite(unc_var) and unc_var > 0:
        sigma2[:start_idx] = unc_var

    for t in range(start_idx, len(y)):
        sigma2[t] = (
            omega
            + alphas[0] * resid[t - 1] ** 2
            + alphas[1] * resid[t - 2] ** 2
            + alphas[2] * resid[t - 3] ** 2
            + alphas[3] * resid[t - 4] ** 2
        )
        if sigma2[t] <= 0 or not np.isfinite(sigma2[t]):
            return resid, None

    return resid, sigma2


def exercise_3_ARCH4_model1(df: pd.DataFrame, make_qq_plot: bool) -> dict:
    """
    Question 3:
    ARCH(4) with normal innovations on raw hitec returns.
    """
    work = clean_data(df)[["date", "hitec"]].dropna().reset_index(drop=True)
    y = work["hitec"].to_numpy(dtype=float)
    start_idx = 4

    var0 = np.var(y, ddof=1)
    alpha0 = np.array([0.20, 0.10, 0.05, 0.05], dtype=float)
    base0 = 1.0 - np.sum(alpha0)
    theta0 = np.array(
        [
            np.mean(y),
            np.log(0.10 * var0),
            np.log(alpha0[0] / base0),
            np.log(alpha0[1] / base0),
            np.log(alpha0[2] / base0),
            np.log(alpha0[3] / base0),
        ],
        dtype=float,
    )

    def unpack(theta: np.ndarray) -> tuple[float, float, np.ndarray]:
        mu = float(theta[0])
        omega = float(np.exp(theta[1]))
        alphas = _simplex_weights(theta[2:6])
        return mu, omega, alphas

    def neg_loglik(theta: np.ndarray) -> float:
        mu, omega, alphas = unpack(theta)
        resid, sigma2 = _arch4_sigma2(y, mu, omega, alphas)
        if sigma2 is None:
            return 1e12

        ll = -0.5 * np.sum(
            np.log(2.0 * np.pi)
            + np.log(sigma2[start_idx:])
            + resid[start_idx:] ** 2 / sigma2[start_idx:]
        )
        if np.isfinite(ll):
            return float(-ll)
        return 1e12

    opt = minimize(
        neg_loglik,
        theta0,
        method="Powell",
        options={"maxiter": 20000},
    )

    mu_hat, omega_hat, alpha_hat = unpack(opt.x)
    resid_hat, sigma2_hat = _arch4_sigma2(y, mu_hat, omega_hat, alpha_hat)
    sigma_hat = np.sqrt(sigma2_hat)
    std_resid = resid_hat / sigma_hat
    loglik = float(-neg_loglik(opt.x))

    jb = jarque_bera_normality_test(
        std_resid[start_idx:],
        alpha=0.05,
        series_name="ARCH(4) standardized residuals",
    )
    if jb["reject_h0"]:
        interpretation = (
            "The QQ plot should deviate from the straight normal line, especially in the "
            "tails, so the normal innovation assumption is not fully convincing."
        )
    else:
        interpretation = (
            "The QQ plot should stay fairly close to the straight normal line, so the "
            "normal innovation assumption looks acceptable."
        )

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
        plot_qq_standardized_residuals(
            std_resid=std_resid,
            start_idx=start_idx,
            distribution="norm",
            title="QQ plot of ARCH(4) standardized residuals vs Normal",
        )

    return {
        "mu": float(mu_hat),
        "omega": float(omega_hat),
        "alpha1": float(alpha_hat[0]),
        "alpha2": float(alpha_hat[1]),
        "alpha3": float(alpha_hat[2]),
        "alpha4": float(alpha_hat[3]),
        "persistence": float(np.sum(alpha_hat)),
        "loglik": loglik,
        "nobs": int(len(work)),
        "convergence": bool(opt.success),
        "start_idx": start_idx,
        "interpretation": interpretation,
        "jb_p_value": float(jb["p_value"]),
        "data": out,
        "summary": summary,
    }


def exercise_4_ARCH4_model1_oos_volatility(df: pd.DataFrame) -> dict:
    """
    Question 4:
    Re-estimate ARCH(4)-normal without the last 9 observations and compute 10
    one-step-ahead volatility forecasts for T-8, ..., T, T+1.
    """
    work = clean_data(df)[["date", "hitec"]].dropna().reset_index(drop=True)
    n_total = len(work)
    n_train = n_total - 9

    train_model = exercise_3_ARCH4_model1(work.iloc[:n_train].copy(), make_qq_plot=False)

    mu = train_model["mu"]
    omega = train_model["omega"]
    alphas = np.array(
        [
            train_model["alpha1"],
            train_model["alpha2"],
            train_model["alpha3"],
            train_model["alpha4"],
        ],
        dtype=float,
    )

    y = work["hitec"].to_numpy(dtype=float)
    resid = y - mu
    rows = []

    for horizon, t in enumerate(range(n_train, n_total + 1), start=1):
        sigma2_t = (
            omega
            + alphas[0] * resid[t - 1] ** 2
            + alphas[1] * resid[t - 2] ** 2
            + alphas[2] * resid[t - 3] ** 2
            + alphas[3] * resid[t - 4] ** 2
        )
        sigma_t = np.sqrt(sigma2_t)

        if t < n_total:
            target_date = work.loc[t, "date"]
            target_label = str(pd.Timestamp(target_date).date())
            realized_hitec = y[t]
        else:
            target_date = pd.NaT
            target_label = "T+1"
            realized_hitec = np.nan

        rows.append(
            {
                "horizon": horizon,
                "target_label": target_label,
                "date": target_date,
                "sigma2_forecast": sigma2_t,
                "sigma_forecast": sigma_t,
                "realized_hitec": realized_hitec,
            }
        )

    forecasts = pd.DataFrame(rows)

    return {
        "train_model": train_model,
        "forecasts": forecasts,
        "n_train": n_train,
        "n_total": n_total,
    }


def exercise_5_var_10day_95(q4_result: dict) -> dict:
    """
    Question 5:
    Build a 10-day 95% VaR from Question 4.
    """
    mu = float(q4_result["train_model"]["mu"])
    sigma_forecasts = q4_result["forecasts"]["sigma_forecast"].to_numpy(dtype=float)
    z_05 = float(norm.ppf(0.05))
    mean_10d = 10.0 * mu
    sd_10d = float(np.sqrt(np.sum(sigma_forecasts**2)))
    var_return_5pct = float(mean_10d + z_05 * sd_10d)
    var_loss_95 = float(-var_return_5pct)

    summary = pd.DataFrame(
        {
            "measure": [
                "daily_mu",
                "mean_10d",
                "sd_10d",
                "z_0.05",
                "VaR_5pct_return_quantile",
                "VaR_95_loss",
            ],
            "value": [
                mu,
                mean_10d,
                sd_10d,
                z_05,
                var_return_5pct,
                var_loss_95,
            ],
        }
    )

    return {
        "mu_daily": mu,
        "mean_10d": mean_10d,
        "sd_10d": sd_10d,
        "z_05": z_05,
        "var_return_5pct": var_return_5pct,
        "var_loss_95": var_loss_95,
        "sigma_forecasts": sigma_forecasts,
        "summary": summary,
    }


def exercise_6_ARCH4_model2_t(df: pd.DataFrame, make_qq_plot: bool) -> dict:
    """
    Question 6:
    ARCH(4) with Student-t innovations on raw hitec returns.
    """
    work = clean_data(df)[["date", "hitec"]].dropna().reset_index(drop=True)
    y = work["hitec"].to_numpy(dtype=float)
    start_idx = 4

    var0 = np.var(y, ddof=1)
    alpha0 = np.array([0.20, 0.10, 0.05, 0.05], dtype=float)
    base0 = 1.0 - np.sum(alpha0)
    theta0 = np.array(
        [
            np.mean(y),
            np.log(0.10 * var0),
            np.log(alpha0[0] / base0),
            np.log(alpha0[1] / base0),
            np.log(alpha0[2] / base0),
            np.log(alpha0[3] / base0),
            np.log(8.0 - 2.01),
        ],
        dtype=float,
    )

    def unpack(theta: np.ndarray) -> tuple[float, float, np.ndarray, float]:
        mu = float(theta[0])
        omega = float(np.exp(theta[1]))
        alphas = _simplex_weights(theta[2:6])
        nu = float(2.01 + np.exp(theta[6]))
        return mu, omega, alphas, nu

    def neg_loglik(theta: np.ndarray) -> float:
        mu, omega, alphas, nu = unpack(theta)
        resid, sigma2 = _arch4_sigma2(y, mu, omega, alphas)
        if sigma2 is None:
            return 1e12

        sigma = np.sqrt(sigma2[start_idx:])
        scale = np.sqrt((nu - 2.0) / nu)
        x = resid[start_idx:] / (sigma * scale)
        ll = np.sum(student_t.logpdf(x, df=nu) - np.log(sigma * scale))
        if np.isfinite(ll):
            return float(-ll)
        return 1e12

    opt = minimize(
        neg_loglik,
        theta0,
        method="Powell",
        options={"maxiter": 20000},
    )

    mu_hat, omega_hat, alpha_hat, nu_hat = unpack(opt.x)
    resid_hat, sigma2_hat = _arch4_sigma2(y, mu_hat, omega_hat, alpha_hat)
    sigma_hat = np.sqrt(sigma2_hat)
    std_resid = resid_hat / sigma_hat
    loglik = float(-neg_loglik(opt.x))

    if nu_hat < 10:
        interpretation = (
            f"The estimated shape coefficient is {nu_hat:.3f}. This is far from the "
            "normal limit, so the t model captures heavy tails and supports the tail "
            "deviations seen in Question 3."
        )
    elif nu_hat < 30:
        interpretation = (
            f"The estimated shape coefficient is {nu_hat:.3f}. The tails are still "
            "heavier than normal, but the deviation from normality is moderate."
        )
    else:
        interpretation = (
            f"The estimated shape coefficient is {nu_hat:.3f}. This is relatively large, "
            "so the t distribution is close to normal and tail departures are mild."
        )

    out = work.copy()
    out["residual"] = resid_hat
    out["sigma2"] = sigma2_hat
    out["sigma"] = sigma_hat
    out["std_resid"] = std_resid

    summary = pd.DataFrame(
        {
            "parameter": ["mu", "omega", "alpha1", "alpha2", "alpha3", "alpha4", "shape"],
            "estimate": [
                mu_hat,
                omega_hat,
                alpha_hat[0],
                alpha_hat[1],
                alpha_hat[2],
                alpha_hat[3],
                nu_hat,
            ],
        }
    )

    if make_qq_plot:
        plot_qq_standardized_residuals(
            std_resid=std_resid,
            start_idx=start_idx,
            distribution="t",
            title=f"QQ plot of ARCH(4)-t standardized residuals vs t(df={nu_hat:.2f})",
            df_t=nu_hat,
        )

    return {
        "mu": float(mu_hat),
        "omega": float(omega_hat),
        "alpha1": float(alpha_hat[0]),
        "alpha2": float(alpha_hat[1]),
        "alpha3": float(alpha_hat[2]),
        "alpha4": float(alpha_hat[3]),
        "shape": float(nu_hat),
        "persistence": float(np.sum(alpha_hat)),
        "loglik": loglik,
        "nobs": int(len(work)),
        "convergence": bool(opt.success),
        "start_idx": start_idx,
        "interpretation": interpretation,
        "data": out,
        "summary": summary,
    }


def _gjr11_sigma2(
    y: np.ndarray,
    mu: float,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    resid = y - mu
    sigma2 = np.full(len(y), np.var(y, ddof=1), dtype=float)

    persistence = alpha + 0.5 * gamma + beta
    unc_var = omega / (1.0 - persistence)
    if np.isfinite(unc_var) and unc_var > 0:
        sigma2[0] = unc_var

    for t in range(1, len(y)):
        indicator = 1.0 if resid[t - 1] < 0 else 0.0
        sigma2[t] = (
            omega
            + alpha * resid[t - 1] ** 2
            + gamma * indicator * resid[t - 1] ** 2
            + beta * sigma2[t - 1]
        )
        if sigma2[t] <= 0 or not np.isfinite(sigma2[t]):
            return resid, None

    return resid, sigma2


def exercise_7_GJR11_model3_t(
    df: pd.DataFrame,
    model1: dict,
    model2: dict,
    make_qq_plot: bool,
) -> dict:
    """
    Question 7:
    GJR(1,1) with Student-t innovations on raw hitec returns.
    """
    work = clean_data(df)[["date", "hitec"]].dropna().reset_index(drop=True)
    y = work["hitec"].to_numpy(dtype=float)
    start_idx = 1

    mu0 = float(model1["mu"])
    nu0 = max(float(model2["shape"]), 4.0)
    var0 = np.var(y, ddof=1)
    omega0 = max(0.05 * var0, 1e-8)
    alpha0 = 0.08
    gamma0 = 0.05
    beta0 = 0.85
    total0 = alpha0 + gamma0 + beta0
    if total0 >= 0.98:
        scale = 0.98 / total0
        alpha0 *= scale
        gamma0 *= scale
        beta0 *= scale

    theta0 = np.array(
        [
            mu0,
            np.log(omega0),
            np.log(alpha0 / (1.0 - alpha0 - gamma0 - beta0)),
            np.log(gamma0 / (1.0 - alpha0 - gamma0 - beta0)),
            np.log(beta0 / (1.0 - alpha0 - gamma0 - beta0)),
            np.log(nu0 - 2.01),
        ],
        dtype=float,
    )

    def unpack(theta: np.ndarray) -> tuple[float, float, float, float, float, float]:
        mu = float(theta[0])
        omega = float(np.exp(theta[1]))
        weights = _simplex_weights(theta[2:5])
        alpha = float(weights[0])
        gamma = float(weights[1])
        beta = float(weights[2])
        nu = float(2.01 + np.exp(theta[5]))
        return mu, omega, alpha, gamma, beta, nu

    def neg_loglik(theta: np.ndarray) -> float:
        mu, omega, alpha, gamma, beta, nu = unpack(theta)
        resid, sigma2 = _gjr11_sigma2(y, mu, omega, alpha, gamma, beta)
        if sigma2 is None:
            return 1e12

        sigma = np.sqrt(sigma2[start_idx:])
        scale = np.sqrt((nu - 2.0) / nu)
        x = resid[start_idx:] / (sigma * scale)
        ll = np.sum(student_t.logpdf(x, df=nu) - np.log(sigma * scale))
        if np.isfinite(ll):
            return float(-ll)
        return 1e12

    opt = minimize(
        neg_loglik,
        theta0,
        method="Powell",
        options={"maxiter": 25000},
    )

    mu_hat, omega_hat, alpha_hat, gamma_hat, beta_hat, nu_hat = unpack(opt.x)
    resid_hat, sigma2_hat = _gjr11_sigma2(
        y,
        mu_hat,
        omega_hat,
        alpha_hat,
        gamma_hat,
        beta_hat,
    )
    sigma_hat = np.sqrt(sigma2_hat)
    std_resid = resid_hat / sigma_hat
    loglik = float(-neg_loglik(opt.x))

    def reported_params(theta: np.ndarray) -> np.ndarray:
        mu, omega, alpha, gamma, beta, nu = unpack(theta)
        return np.array([mu, omega, alpha, gamma, beta, nu], dtype=float)

    cov = _reported_covariance(neg_loglik, opt.x, reported_params)
    se_mu, se_omega, se_alpha, se_gamma, se_beta, se_shape = np.sqrt(
        np.maximum(np.diag(cov), 0.0)
    )

    z_gamma = float(gamma_hat / se_gamma) if se_gamma > 0 else np.nan
    p_value_gamma = float(2.0 * (1.0 - norm.cdf(abs(z_gamma)))) if np.isfinite(z_gamma) else np.nan
    significant_at_5pct = bool(p_value_gamma < 0.05) if np.isfinite(p_value_gamma) else False

    if gamma_hat > 0:
        direction_text = "Variance goes up after a negative shock."
        negative_shock_effect = "up"
    elif gamma_hat < 0:
        direction_text = "Variance goes down after a negative shock."
        negative_shock_effect = "down"
    else:
        direction_text = "Negative shocks do not change variance relative to positive shocks."
        negative_shock_effect = "no change"

    if significant_at_5pct:
        significance_text = "The effect is statistically significant at the 5% level."
    else:
        significance_text = "The effect is not statistically significant at the 5% level."

    interpretation = (
        f"{direction_text} {significance_text} "
        f"(gamma = {gamma_hat:.4f}, p-value = {p_value_gamma:.4f})"
    )

    out = work.copy()
    out["residual"] = resid_hat
    out["sigma2"] = sigma2_hat
    out["sigma"] = sigma_hat
    out["std_resid"] = std_resid

    summary = pd.DataFrame(
        {
            "parameter": ["mu", "omega", "alpha", "gamma", "beta", "shape"],
            "estimate": [mu_hat, omega_hat, alpha_hat, gamma_hat, beta_hat, nu_hat],
            "std_error": [se_mu, se_omega, se_alpha, se_gamma, se_beta, se_shape],
        }
    )

    if make_qq_plot:
        plot_qq_standardized_residuals(
            std_resid=std_resid,
            start_idx=start_idx,
            distribution="t",
            title=f"QQ plot of GJR(1,1)-t standardized residuals vs t(df={nu_hat:.2f})",
            df_t=nu_hat,
        )

    return {
        "mu": float(mu_hat),
        "omega": float(omega_hat),
        "alpha": float(alpha_hat),
        "gamma": float(gamma_hat),
        "beta": float(beta_hat),
        "shape": float(nu_hat),
        "se_mu": float(se_mu),
        "se_omega": float(se_omega),
        "se_alpha": float(se_alpha),
        "se_gamma": float(se_gamma),
        "se_beta": float(se_beta),
        "se_shape": float(se_shape),
        "z_gamma": z_gamma,
        "p_value_gamma": p_value_gamma,
        "negative_shock_effect": negative_shock_effect,
        "significant_at_5pct": significant_at_5pct,
        "loglik": loglik,
        "nobs": int(len(work)),
        "convergence": bool(opt.success),
        "start_idx": start_idx,
        "interpretation": interpretation,
        "data": out,
        "summary": summary,
    }


def exercise_8_compare_model2_model3(
    model2_result: dict,
    model3_result: dict,
    lags: int,
) -> dict:
    """
    Question 8:
    Compare model 2 and model 3 with the ACF of standardized residuals and
    squared standardized residuals.
    """
    start2 = model2_result["start_idx"]
    start3 = model3_result["start_idx"]

    std2 = model2_result["data"]["std_resid"].to_numpy(dtype=float)[start2:]
    std3 = model3_result["data"]["std_resid"].to_numpy(dtype=float)[start3:]
    std2 = std2[np.isfinite(std2)]
    std3 = std3[np.isfinite(std3)]
    std2_sq = std2**2
    std3_sq = std3**2

    fig1, axes1 = plt.subplots(2, 1, figsize=(8, 8))
    plot_acf(std2, lags=lags, ax=axes1[0], title="Model 2: ACF of standardized residuals")
    plot_acf(std3, lags=lags, ax=axes1[1], title="Model 3: ACF of standardized residuals")
    plt.tight_layout()
    plt.show()

    fig2, axes2 = plt.subplots(2, 1, figsize=(8, 8))
    plot_acf(std2_sq, lags=lags, ax=axes2[0], title="Model 2: ACF of squared standardized residuals")
    plot_acf(std3_sq, lags=lags, ax=axes2[1], title="Model 3: ACF of squared standardized residuals")
    plt.tight_layout()
    plt.show()

    lb2_std = acorr_ljungbox(std2, lags=[lags], return_df=True).iloc[0]
    lb2_sq = acorr_ljungbox(std2_sq, lags=[lags], return_df=True).iloc[0]
    lb3_std = acorr_ljungbox(std3, lags=[lags], return_df=True).iloc[0]
    lb3_sq = acorr_ljungbox(std3_sq, lags=[lags], return_df=True).iloc[0]

    acf_score_2 = float(
        np.sum(np.abs(acf(std2, nlags=lags, fft=False)[1:]))
        + np.sum(np.abs(acf(std2_sq, nlags=lags, fft=False)[1:]))
    )
    acf_score_3 = float(
        np.sum(np.abs(acf(std3, nlags=lags, fft=False)[1:]))
        + np.sum(np.abs(acf(std3_sq, nlags=lags, fft=False)[1:]))
    )

    diagnostics = pd.DataFrame(
        {
            "model": ["Model 2: ARCH(4)-t", "Model 3: GJR(1,1)-t"],
            "lb_std_stat": [lb2_std["lb_stat"], lb3_std["lb_stat"]],
            "lb_std_pvalue": [lb2_std["lb_pvalue"], lb3_std["lb_pvalue"]],
            "lb_std_sq_stat": [lb2_sq["lb_stat"], lb3_sq["lb_stat"]],
            "lb_std_sq_pvalue": [lb2_sq["lb_pvalue"], lb3_sq["lb_pvalue"]],
            "acf_score": [acf_score_2, acf_score_3],
        }
    )

    if acf_score_2 < acf_score_3:
        better_model = "Model 2: ARCH(4)-t"
    else:
        better_model = "Model 3: GJR(1,1)-t"

    interpretation = (
        f"{better_model} leaves less residual dependence overall. In Question 8 the most "
        "important diagnostic is the ACF of squared standardized residuals, because it shows "
        "whether volatility clustering is still left in the residuals."
    )

    return {
        "diagnostics": diagnostics,
        "better_model": better_model,
        "interpretation": interpretation,
    }


def exercise_9_compare_aic(
    model1_result: dict,
    model2_result: dict,
    model3_result: dict,
) -> dict:
    """
    Question 9:
    Compare the three models using AIC = 2k - 2 logLik.
    """
    rows = []

    for name, result in [
        ("Model 1: ARCH(4)-normal", model1_result),
        ("Model 2: ARCH(4)-t", model2_result),
        ("Model 3: GJR(1,1)-t", model3_result),
    ]:
        k = len(result["summary"])
        loglik = float(result["loglik"])
        rows.append(
            {
                "model": name,
                "n_params": k,
                "loglik": loglik,
                "AIC": 2.0 * k - 2.0 * loglik,
            }
        )

    comparison = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    comparison["rank"] = np.arange(1, len(comparison) + 1)
    best_model = comparison.loc[0, "model"]
    ranking = comparison["model"].tolist()

    if best_model == "Model 1: ARCH(4)-normal":
        interpretation = (
            "Model 1 has the lowest AIC, so the normal ARCH(4) model gives the best "
            "fit-parsimony trade-off in this sample."
        )
    elif best_model == "Model 2: ARCH(4)-t":
        interpretation = (
            "Model 2 has the lowest AIC, so allowing for heavy tails improves fit enough "
            "to dominate the normal model and the GJR model."
        )
    else:
        interpretation = (
            "Model 3 has the lowest AIC, so both heavy tails and asymmetric volatility "
            "matter for the hitec return series."
        )

    return {
        "comparison": comparison,
        "best_model": best_model,
        "ranking": ranking,
        "interpretation": interpretation,
    }
