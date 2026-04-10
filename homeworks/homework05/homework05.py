import string

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], yearfirst=True, errors="coerce")
    for column in ["fund2_qoq", "rf_qoq", "mktrf_qoq", "smb_qoq", "hml_qoq"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _exercise_1_log_excess_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_data(df)
    out["fund2_qoq_log_excess"] = 100.0 * (
        np.log1p(out["fund2_qoq"] / 100.0) - np.log1p(out["rf_qoq"] / 100.0)
    )
    return out


def _exercise_1_sample(df: pd.DataFrame, date: string = "1991-03-01") -> pd.DataFrame:
    out = _exercise_1_log_excess_returns(df)
    out = out.loc[out["date"] >= date].copy()
    out = out[
        ["date", "fund2_qoq_log_excess", "mktrf_qoq", "smb_qoq", "hml_qoq"]
    ].dropna()
    return out


def _fit_fama_french_from_sample(df_sample: pd.DataFrame) -> RegressionResultsWrapper:
    X = df_sample[["mktrf_qoq", "smb_qoq", "hml_qoq"]]
    X = sm.add_constant(X, has_constant="add")
    y = df_sample["fund2_qoq_log_excess"]
    model = sm.OLS(y, X).fit()
    return model


def exercise_1_fama_french(df: pd.DataFrame):
    df_sample = _exercise_1_sample(df)
    return _fit_fama_french_from_sample(df_sample)


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

    df_sample = _exercise_1_sample(df)
    original_model = _fit_fama_french_from_sample(df_sample)

    fitted_values = original_model.fittedvalues.to_numpy()
    residuals = original_model.resid.to_numpy()

    resampled_residuals = rng.choice(residuals, size=len(residuals), replace=True)
    y_star = fitted_values + resampled_residuals

    df_boot = df_sample.copy()
    df_boot["fund2_qoq_log_excess"] = y_star

    bootstrap_model = _fit_fama_french_from_sample(df_boot)

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

    df_sample = _exercise_1_sample(df)
    original_model = _fit_fama_french_from_sample(df_sample)

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
