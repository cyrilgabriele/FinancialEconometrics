from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt 

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
from linearmodels.system import SUR



def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with obvious type fixes (dates->datetime, numerics)."""
    cleaned = df.copy()

    date_cols = [col for col in cleaned.columns if "date" in col.lower()]
    for col in date_cols:
        cleaned[col] = pd.to_datetime(cleaned[col], dayfirst=True, errors="coerce")

    return cleaned


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)

    for col in ["DIST", "FI", "MAC", "ELS", "MULT"]:
        out[f"{col}_ret"] = out[col].pct_change() * 100
        out[f"{col}_excess"] = out[f"{col}_ret"] - out["rf"]

    return out.iloc[1:].copy()   # drop 2000-12-31 i.e. the first row


def estimate_market_model(df: pd.DataFrame):
    y = df["MULT_excess"]
    X = sm.add_constant(df["mktrf"])
    return sm.OLS(y, X, missing="drop").fit()


def plot_residuals(df, model):
    residuals = model.resid
    fitted = model.fittedvalues

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].scatter(fitted, residuals, alpha=0.6)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs. Fitted")

    axes[1].plot(df["date"], residuals, marker="o", linestyle="-", alpha=0.7)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals over Time")

    plt.show()


def test_heterodkedasticity(df, model):
    white_stat, white_pvalue, f_stat, f_pvalue = het_white(model.resid, model.model.exog)
    return {
        "white_stat": white_stat,
        "white_pvalue": white_pvalue,
        "f_stat": f_stat,
        "f_pvalue": f_pvalue
    }


def exercise_3_autocorr_check(model):
    residuals = model.resid

    # correlogram with 10 lags
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(residuals, lags=10, ax=ax)
    ax.set_title("Correlogram of residuals (10 lags)")
    plt.show()

    # Durbin-Watson test
    dw_stat = durbin_watson(residuals)

    if dw_stat < 1.5:
        conclusion = "evidence of positive first-order autocorrelation"
    elif dw_stat > 1.5:
        conclusion = "evidence of negative first-order autocorrelation"
    else:
        conclusion = "no first-order autocorrelation"

    return {
        "durbin_watson": dw_stat,
        "conclusion": conclusion
    }


def exercise_5_hac_results(model, maxlags=1):
    """
    Calculate HAC (Newey-West) standard errors for the OLS coefficients
    and compare them to the usual OLS results from exercise 1.

    Parameters
    ----------
    model : statsmodels RegressionResults
        The OLS model estimated in exercise 1.
    maxlags : int, default 1
        Number of HAC/Newey-West lags.

    Returns
    -------
    comparison : pd.DataFrame
        Table comparing OLS and HAC results for const and mktrf.
    hac_model : statsmodels RegressionResults
        Same OLS coefficients, but with HAC covariance matrix.
    """
    hac_model = model.get_robustcov_results(
        cov_type="HAC",
        maxlags=maxlags,
        use_correction=True
    )

    idx = model.params.index

    comparison = pd.DataFrame({
        "coef_OLS": model.params,
        "se_OLS": model.bse,
        "t_OLS": model.tvalues,
        "p_OLS": model.pvalues,
        "coef_HAC": pd.Series(hac_model.params, index=idx),
        "se_HAC": pd.Series(hac_model.bse, index=idx),
        "t_HAC": pd.Series(hac_model.tvalues, index=idx),
        "p_HAC": pd.Series(hac_model.pvalues, index=idx),
    })

    return comparison, hac_model


def exercise_6_sure_capm(df):
    """
    Estimate the CAPM on MAC, MULT and ELS in a SURE fashion.
    Uses a non-HAC (unadjusted) covariance estimator.
    Returns the fitted SUR model and the residual covariance matrix.
    """
    formulas = {
        "MAC":  "MAC_excess  ~ 1 + mktrf",
        "MULT": "MULT_excess ~ 1 + mktrf",
        "ELS":  "ELS_excess  ~ 1 + mktrf",
    }
    sur_model = SUR.from_formula(formulas, data=df).fit(cov_type="unadjusted")
    resid_cov = sur_model.sigma
    return sur_model, resid_cov


def exercise_7_joint_alpha_test(sur_model, significance=0.05):
    """
    Wald chi-square test for joint significance of the three CAPM alphas.
    H_0: alpha_MAC = alpha_MULT = alpha_ELS = 0
    """
    import numpy as np
    from scipy import stats

    alpha_names = [n for n in sur_model.params.index if "Intercept" in n]
    alphas = sur_model.params[alpha_names].values
    cov_alphas = sur_model.cov.loc[alpha_names, alpha_names].values

    wald_stat = alphas @ np.linalg.inv(cov_alphas) @ alphas
    df = len(alphas)
    crit_value = stats.chi2.ppf(1 - significance, df)
    p_value = 1 - stats.chi2.cdf(wald_stat, df)

    return {
        "wald_stat": wald_stat,
        "crit_value": crit_value,
        "p_value": p_value,
        "df": df,
        "reject": wald_stat > crit_value,
    }


def exercise_8_post_crisis_beta_test(df, beta_null=0.21, significance=0.05):
    """
    Subsample from 2009-07, re-estimate CAPM on MULT with HAC standard errors,
    and test H_0: beta = beta_null (two-sided).
    """
    import numpy as np
    from scipy import stats

    sub = df[df["date"] >= "2009-07-01"].copy()

    y = sub["MULT_excess"]
    X = sm.add_constant(sub["mktrf"])
    model = sm.OLS(y, X).fit()
    hac = model.get_robustcov_results(cov_type="HAC", maxlags=1, use_correction=True)

    beta_hat = float(hac.params[1])
    se_beta = float(hac.bse[1])
    t_stat = (beta_hat - beta_null) / se_beta
    dof = int(hac.df_resid)
    crit = stats.t.ppf(1 - significance / 2, dof)

    return {
        "beta_hat": beta_hat,
        "se_beta_hac": se_beta,
        "t_stat": t_stat,
        "crit_value": crit,
        "dof": dof,
        "reject": abs(t_stat) > crit,
        "n_obs": len(sub),
        "hac_model": hac,
    }


def exercise_9_capm_vs_ff5(df):
    """
    Compare significance of alpha for FI under CAPM vs Fama-French 5-factor model.
    Both use HAC standard errors.
    """
    y = df["FI_excess"]

    # CAPM
    X_capm = sm.add_constant(df["mktrf"])
    capm_hac = sm.OLS(y, X_capm).fit().get_robustcov_results(
        cov_type="HAC", maxlags=1, use_correction=True
    )

    # Fama-French 5-factor
    X_ff5 = sm.add_constant(df[["mktrf", "smb", "hml", "rmw", "cma"]])
    ff5_hac = sm.OLS(y, X_ff5).fit().get_robustcov_results(
        cov_type="HAC", maxlags=1, use_correction=True
    )

    return {
        "alpha_capm": float(capm_hac.params[0]),
        "t_capm": float(capm_hac.tvalues[0]),
        "p_capm": float(capm_hac.pvalues[0]),
        "alpha_ff5": float(ff5_hac.params[0]),
        "t_ff5": float(ff5_hac.tvalues[0]),
        "p_ff5": float(ff5_hac.pvalues[0]),
    }


def exercise_10_ff5_sure(df):
    """
    Estimate Fama-French 5-factor model on FI, MULT and ELS in a SURE fashion.
    Uses a non-HAC (unadjusted) covariance estimator.
    Returns the fitted SUR model and the residual covariance matrix.
    """
    formulas = {
        "FI":   "FI_excess   ~ 1 + mktrf + smb + hml + rmw + cma",
        "MULT": "MULT_excess ~ 1 + mktrf + smb + hml + rmw + cma",
        "ELS":  "ELS_excess  ~ 1 + mktrf + smb + hml + rmw + cma",
    }
    sur_model = SUR.from_formula(formulas, data=df).fit(cov_type="unadjusted")
    resid_cov = sur_model.sigma
    return sur_model, resid_cov


def exercise_11_joint_alpha_test(sur_model, significance=0.05):
    """
    Wald chi-square test for joint significance of three FF5 alphas.
    H_0: alpha_FI = alpha_MULT = alpha_ELS = 0
    """
    import numpy as np
    from scipy import stats

    alpha_names = [n for n in sur_model.params.index if "Intercept" in n]
    alphas = sur_model.params[alpha_names].values
    cov_alphas = sur_model.cov.loc[alpha_names, alpha_names].values

    wald_stat = alphas @ np.linalg.inv(cov_alphas) @ alphas
    df = len(alphas)
    crit_value = stats.chi2.ppf(1 - significance, df)
    p_value = 1 - stats.chi2.cdf(wald_stat, df)

    return {
        "wald_stat": wald_stat,
        "crit_value": crit_value,
        "p_value": p_value,
        "df": df,
        "reject": wald_stat > crit_value,
    }


if __name__ == "__main__":
    # ex. 0
    data_path = Path(__file__).resolve().parent / "s2_data.csv"
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df = clean_data(df)
    # ---
    # ex. 1
    df = prepare_data(df)
    model = estimate_market_model(df)
    alpha = model.params["const"]
    beta = model.params["mktrf"]
    print("alpha:", alpha)
    print("beta:", beta)
    print("R^2:", model.rsquared)
    print("alpha p-value:", model.pvalues["const"])
    # H_0: alpha = 0
    # significance level: 5% (given)
    if model.pvalues["const"] <= 0.05: 
        print("REJECT H_0, where H_0: alpha = 0")
    else: 
        print("DO NOT (!) REJECT H_0, where H_0: alpha = 0")
    # ---
    # ex. 2
    plot_residuals(df, model)
    stats = test_heterodkedasticity(df, model)
    for key in stats.keys():
        print(f"{key}: {stats.get(key)}: ")
    # ---
    # ex. 3
    results = exercise_3_autocorr_check(model)
    print(results)
    # ---
    # ex. 5
    comparison, hac_model = exercise_5_hac_results(model, maxlags=1)
    print(comparison.round(6))