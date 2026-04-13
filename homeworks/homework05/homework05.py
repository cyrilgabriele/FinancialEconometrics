import string

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from scipy.stats import norm


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], yearfirst=True, errors="coerce")
    for column in ["fund2_qoq", "rf_qoq", "mktrf_qoq", "smb_qoq", "hml_qoq"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _log_excess_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_data(df)
    out["fund2_qoq_log_excess"] = 100.0 * (
        np.log1p(out["fund2_qoq"] / 100.0) - np.log1p(out["rf_qoq"] / 100.0)
    )
    return out


def _get_samples(df: pd.DataFrame, date: string = "1991-03-01") -> pd.DataFrame:
    out = _log_excess_returns(df)
    out = out.loc[out["date"] >= date].copy()
    out = out[
        ["date", "fund2_qoq_log_excess", "mktrf_qoq", "smb_qoq", "hml_qoq"]
    ].dropna()
    return out


def _fit_fama_french_from_samples(df_sample: pd.DataFrame) -> RegressionResultsWrapper:
    X = df_sample[["mktrf_qoq", "smb_qoq", "hml_qoq"]]
    X = sm.add_constant(X, has_constant="add")
    y = df_sample["fund2_qoq_log_excess"]
    model = sm.OLS(y, X).fit()
    return model


def exercise_1_fama_french(df: pd.DataFrame):
    df_sample = _get_samples(df)
    return _fit_fama_french_from_samples(df_sample)


def exercise_2_bootstrap_once(df: pd.DataFrame, seed: int = 42):# -> dict[str, Any]:
    """
    Construct one bootstrap sample by resampling residuals with replacement
    and re-estimate the Fama-French model on the artificial response.

    Returns
    -------
    dict with:
        - original_model
        - bootstrap_model
        - bootstrap_sample
        - bootstrap_coefficients
    """
    rng = np.random.default_rng(seed)

    df_sample = _get_samples(df)
    original_model = _fit_fama_french_from_samples(df_sample)

    fitted_values = original_model.fittedvalues.to_numpy()
    residuals = original_model.resid.to_numpy()

    resampled_residuals = rng.choice(residuals, size=len(residuals), replace=True)
    y_star = fitted_values + resampled_residuals

    df_boot = df_sample.copy()
    df_boot["fund2_qoq_log_excess"] = y_star

    bootstrap_model = _fit_fama_french_from_samples(df_boot)

    return {
        "original_model": original_model,
        "bootstrap_model": bootstrap_model,
        "bootstrap_sample": df_boot,
        "bootstrap_coefficients": bootstrap_model.params.copy(),
    }


def exercise_3_bootstrap_pvalues(
    df: pd.DataFrame,
    n_boot: int = 10000,
    seed: int = 42,
    ci_level: float = 0.95,
):
    """
    Residual bootstrap for the Fama-French regression.

    - bootstrap percentile confidence intervals
    - two-sided bootstrap p-values computed from the empirical share of
      bootstrapped coefficients on either side of 0

    Returns
    -------
    dict with:
        - original_model
        - bootstrap_coefficients
        - bootstrap_pvalues_two_sided
        - bootstrap_confidence_intervals
        - summary_table
    """
    rng = np.random.default_rng(seed)

    df_sample = _get_samples(df)
    original_model = _fit_fama_french_from_samples(df_sample)

    X = sm.add_constant(
        df_sample[["mktrf_qoq", "smb_qoq", "hml_qoq"]],
        has_constant="add"
    )
    y_hat = original_model.fittedvalues.to_numpy()
    residuals = original_model.resid.to_numpy()

    param_names = list(original_model.params.index)
    k = len(param_names)

    boot_betas = np.empty((n_boot, k))

    for b in range(n_boot):
        resampled_residuals = rng.choice(residuals, size=len(residuals), replace=True)
        y_star = y_hat + resampled_residuals
        boot_model = sm.OLS(y_star, X).fit()
        boot_betas[b, :] = boot_model.params.to_numpy()

    bootstrap_coefficients = pd.DataFrame(boot_betas, columns=param_names)

    # 2 * min(share below 0, share above 0)
    bootstrap_pvalues_two_sided = pd.Series(index=param_names, dtype=float)

    for name in param_names:
        draws = bootstrap_coefficients[name].to_numpy()
        p_left = np.mean(draws <= 0.0)
        p_right = np.mean(draws >= 0.0)
        bootstrap_pvalues_two_sided[name] = min(1.0, 2.0 * min(p_left, p_right))

    alpha = 1.0 - ci_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    bootstrap_confidence_intervals = pd.DataFrame({
        "ci_lower": bootstrap_coefficients.quantile(lower_q / 100.0),
        "ci_upper": bootstrap_coefficients.quantile(upper_q / 100.0),
    })

    summary_table = pd.DataFrame({
        "coef_hat": original_model.params,
        "bootstrap_pvalue_two_sided": bootstrap_pvalues_two_sided,
        "ci_lower": bootstrap_confidence_intervals["ci_lower"],
        "ci_upper": bootstrap_confidence_intervals["ci_upper"],
    })

    return {
        "original_model": original_model,
        "bootstrap_coefficients": bootstrap_coefficients,
        "bootstrap_pvalues_two_sided": bootstrap_pvalues_two_sided,
        "bootstrap_confidence_intervals": bootstrap_confidence_intervals,
        "summary_table": summary_table,
    }


def _moving_block_bootstrap_indices(
    n_obs: int,
    block_length: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Moving block bootstrap indices:
    draw overlapping blocks with replacement until length n_obs is reached.
    """
    if block_length == None:
        block_length = n_obs ** (1/3)   # from lecture notes 

    n_blocks = int(np.ceil(n_obs / block_length))
    max_start = n_obs - block_length
    starts = rng.integers(0, max_start + 1, size=n_blocks)

    indices = np.concatenate(
        [np.arange(start, start + block_length) for start in starts]
    )[:n_obs]

    return indices


def exercise_4_block_bootstrap_pvalues(
    df: pd.DataFrame,
    n_boot: int = 10000,
    seed: int = 42,
):
    """
    Exercise 4:
    Block bootstrap version of Exercise 3.

    Returns
    -------
    dict with:
        - original_model
        - bootstrap_coefficients
        - bootstrap_pvalues_two_sided
        - summary_table
    """
    block_length = int(np.ceil(len(_get_samples(df)) ** (1/3)))

    rng = np.random.default_rng(seed)

    df_sample = _get_samples(df)
    original_model = _fit_fama_french_from_samples(df_sample)

    X = sm.add_constant(
        df_sample[["mktrf_qoq", "smb_qoq", "hml_qoq"]],
        has_constant="add",
    )
    y_hat = original_model.fittedvalues.to_numpy()
    residuals = original_model.resid.to_numpy()

    n_obs = len(residuals)
    param_names = list(original_model.params.index)
    k = len(param_names)

    boot_betas = np.empty((n_boot, k))

    for b in range(n_boot):
        block_idx = _moving_block_bootstrap_indices(
            n_obs=n_obs,
            block_length=block_length,
            rng=rng,
        )

        u_star = residuals[block_idx]
        y_star = y_hat + u_star

        boot_model = sm.OLS(y_star, X).fit()
        boot_betas[b, :] = boot_model.params.to_numpy()

    bootstrap_coefficients = pd.DataFrame(boot_betas, columns=param_names)

    bootstrap_pvalues_two_sided = pd.Series(index=param_names, dtype=float)

    for name in param_names:
        draws = bootstrap_coefficients[name].to_numpy()
        p_left = np.mean(draws <= 0.0)
        p_right = np.mean(draws >= 0.0)
        bootstrap_pvalues_two_sided[name] = min(1.0, 2.0 * min(p_left, p_right))

    summary_table = pd.DataFrame({
        "coef_hat": original_model.params,
        "block_bootstrap_pvalue_two_sided": bootstrap_pvalues_two_sided,
    })

    return {
        "original_model": original_model,
        "bootstrap_coefficients": bootstrap_coefficients,
        "bootstrap_pvalues_two_sided": bootstrap_pvalues_two_sided,
        "summary_table": summary_table,
    }


def exercise_5_block_bootstrap_ci(
    df: pd.DataFrame,
    n_boot: int = 10000,
    seed: int = 42,
    ci_level: float = 0.90,
):
    """
    Exercise 5:
    Compute confidence intervals for the 4 Fama-French coefficients
    based on the block bootstrap from Exercise 4.

    Returns
    -------
    dict with:
        - original_model
        - bootstrap_coefficients
        - confidence_intervals
        - summary_table
    """
    ex4_result = exercise_4_block_bootstrap_pvalues(
        df=df,
        n_boot=n_boot,
        seed=seed,
    )

    original_model = ex4_result["original_model"]
    bootstrap_coefficients = ex4_result["bootstrap_coefficients"]

    alpha = 1.0 - ci_level      # alpha: significance level 
                                # => since confidence band divide it by 2 after in the CI computation

    confidence_intervals = pd.DataFrame({
        "ci_lower": bootstrap_coefficients.quantile(alpha / 2),
        "ci_upper": bootstrap_coefficients.quantile(1 - alpha / 2),
    })

    summary_table = pd.DataFrame({
        "coef_hat": original_model.params,
        "ci_lower": confidence_intervals["ci_lower"],
        "ci_upper": confidence_intervals["ci_upper"],
    })

    return {
        "original_model": original_model,
        "bootstrap_coefficients": bootstrap_coefficients,
        "confidence_intervals": confidence_intervals,
        "summary_table": summary_table,
    }

