import string

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
import matplotlib.pyplot as plt


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
    
    # verified therefore commented out: date column has right dtype
    # print(f"df_sample.dtypes:\n {df_sample.dtypes}")
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

# switch to other dataset: s5_data_panel_hw.txt

def _parse_date_column(series: pd.Series) -> pd.Series:
    """
    Parse dates robustly for both YYYYMMDD-style numeric dates
    and already formatted date strings.
    """
    s = series.astype(str).str.strip()

    # try YYYYMMDD first
    parsed = pd.to_datetime(s, format="%Y%m%d", errors="coerce")

    # fallback for already formatted dates
    fallback = pd.to_datetime(s, errors="coerce")

    return parsed.fillna(fallback)


def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Convert selected columns to numeric if they exist.
    """
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _clean_panel_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the panel dataset used in HW05.
    """
    out = df.copy()

    out["date"] = _parse_date_column(out["date"])
    # verified therefore commented out: date column has right dtype
    # print(f"in: _clean_panel_data: out.dtypes:\n {out.dtypes}")


    numeric_cols = [
        "main_strategy",
        "id",
        "management_fee",
        "incentive_fee",
        "high_watermark",
        "firm_id",
        "strategy_code",
        "performance",
        "nav",
        "assets",
        "mktrf",
        "smb",
        "hml",
        "rf",
    ]
    out = _coerce_numeric_columns(out, numeric_cols)

    out = out.sort_values(["main_strategy", "date", "id"]).reset_index(drop=True)
    return out


def _panel_log_excess_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create log excess returns for the panel dataset.

    Uses:
        performance = fund return
        rf          = risk-free rate
    """
    out = _clean_panel_data(df)

    out["log_excess_return"] = 100.0 * (
        np.log1p(out["performance"] / 100.0) - np.log1p(out["rf"] / 100.0)
    )

    return out


def _get_panel_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the cleaned panel sample needed for the HW05 panel exercises.
    """
    out = _panel_log_excess_returns(df)

    keep_cols = [
        "date",
        "main_strategy",
        "log_excess_return",
        "mktrf",
        "smb",
        "hml",
        "id",
        "firm_id",
        "strategy_code",
    ]

    keep_cols = [col for col in keep_cols if col in out.columns]
    out = out[keep_cols].dropna(subset=["date", "main_strategy", "log_excess_return", "mktrf", "smb", "hml"])

    return out


def _get_panel_boxplot_data(df: pd.DataFrame) -> dict:
    """
    Prepare panel data for Exercise 6 boxplots.
    Groups log excess returns by main_strategy.
    """
    df_panel = _get_panel_sample(df)

    strategies = sorted(df_panel["main_strategy"].dropna().unique())
    grouped_returns = {
        strategy: df_panel.loc[
            df_panel["main_strategy"] == strategy, "log_excess_return"
        ].dropna().to_numpy()
        for strategy in strategies
    }

    return {
        "df_panel": df_panel,
        "strategies": strategies,
        "grouped_returns": grouped_returns,
    }


def exercise_6_boxplots(df: pd.DataFrame):
    """
    Exercise 6:
    Create 6 boxplots of log excess returns, one for each strategy.
    Uses main_strategy as the panel identifier.
    """
    panel_data = _get_panel_boxplot_data(df)

    strategies = panel_data["strategies"]
    grouped_returns = panel_data["grouped_returns"]

    if len(strategies) != 6:
        raise ValueError(
            f"Expected 6 strategies for Exercise 6, but found {len(strategies)}."
        )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes = axes.flatten()

    for ax, strategy in zip(axes, strategies):
        ax.boxplot(grouped_returns[strategy])
        ax.set_title(f"Strategy {strategy}")
        ax.set_ylabel("Log excess return")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Exercise 6: Boxplots of log excess returns by main_strategy")
    fig.tight_layout()
    plt.show()

    return {
        "figure": fig,
        "axes": axes,
        "df_panel": panel_data["df_panel"],
        "grouped_returns": grouped_returns,
    }


def exercise_7_pooled_ols(df: pd.DataFrame):
    """
    Exercise 7:
    Estimate the Fama-French model with pooled OLS on the panel dataset
    and report coefficients with conventional standard errors.
    """
    df_panel_sample = _get_panel_sample(df)

    X = df_panel_sample[["mktrf", "smb", "hml"]]
    X = sm.add_constant(X, has_constant="add")
    y = df_panel_sample["log_excess_return"]

    pooled_model = sm.OLS(y, X).fit()

    summary_table = pd.DataFrame({
        "coef_hat": pooled_model.params,
        "std_error": pooled_model.bse,
        "t_stat": pooled_model.tvalues,
        "p_value": pooled_model.pvalues,
    })

    return {
        "panel_sample": df_panel_sample,
        "pooled_model": pooled_model,
        "summary_table": summary_table,
    }


def exercise_8_durbin_watson_test(df: pd.DataFrame):
    """
    Exercise 8:
    Perform a panel-aware Durbin-Watson test on the residuals of the
    pooled OLS model from Exercise 7.

    Since Python does not provide a direct equivalent of plm::pdwtest(),
    we compute the DW statistic using residual differences only within
    each main_strategy over time.

    Returns
    -------
    dict with:
        - pooled_model
        - panel_sample
        - residuals_with_lags
        - summary_table
    """
    ex7_result = exercise_7_pooled_ols(df)

    pooled_model = ex7_result["pooled_model"]
    panel_sample = ex7_result["panel_sample"].copy()

    panel_sample = panel_sample.sort_values(["main_strategy", "date"]).reset_index(drop=True)

    panel_sample["residual"] = pooled_model.resid.to_numpy()

    panel_sample["residual_lag"] = (
        panel_sample.groupby("main_strategy")["residual"].shift(1)
    )

    valid = panel_sample.dropna(subset=["residual_lag"]).copy()

    # Durbin-Watson statistic:
    # DW = sum_t (e_t - e_{t-1})^2 / sum_t e_t^2
    dw_stat = (
        np.sum((valid["residual"] - valid["residual_lag"]) ** 2)
        / np.sum(panel_sample["residual"] ** 2)
    )

    # implied first-order autocorrelation from DW ≈ 2 - 2*rho_1
    implied_rho1 = 1.0 - dw_stat / 2.0

    # thershold from lecture notes
    if dw_stat < 1.5:
        conclusion = "evidence of positive autocorrelation"
    else:
        conclusion = "no strong evidence of positive autocorrelation"

    summary_table = pd.DataFrame({
        "durbin_watson_stat": [dw_stat],
        "implied_rho1": [implied_rho1],
        "n_within_panel_lags": [len(valid)],
        "conclusion_rule_of_thumb": [conclusion],
    })

    return {
        "pooled_model": pooled_model,
        "panel_sample": panel_sample,
        "residuals_with_lags": valid[[
            "date", "main_strategy", "residual", "residual_lag"
        ]].copy(),
        "summary_table": summary_table,
    }


def exercise_9_pooled_ols_robust(df: pd.DataFrame):
    df_panel_sample = _get_panel_sample(df).copy()
    df_panel_sample = df_panel_sample.sort_values(
        ["main_strategy", "date"]
    ).reset_index(drop=True)

    X = df_panel_sample[["mktrf", "smb", "hml"]]
    X = sm.add_constant(X, has_constant="add")
    y = df_panel_sample["log_excess_return"]

    pooled_model = sm.OLS(y, X).fit()

    T = df_panel_sample["date"].nunique()
    maxlags = int(np.floor(0.75 * (T ** (1 / 3))))

    # hac-groupsum is the Driscoll-Kraay in statsmodel! 
    # use the Driscoll-Kraay if autocorrelation in residuals for Panel Data!
    robust_model = pooled_model.get_robustcov_results(
        cov_type="hac-groupsum",
        time=df_panel_sample["date"].factorize()[0],
        maxlags=maxlags,
        kernel="bartlett",
        use_correction="cluster",
        df_correction=True,
    )

    param_names = pooled_model.model.exog_names

    summary_table = pd.DataFrame({
        "coef_hat": pd.Series(robust_model.params, index=param_names),
        "std_error_ols": pooled_model.bse.reindex(param_names),
        "p_value_ols": pooled_model.pvalues.reindex(param_names),
        "std_error_robust": pd.Series(robust_model.bse, index=param_names),
        "t_stat_robust": pd.Series(robust_model.tvalues, index=param_names),
        "p_value_robust": pd.Series(robust_model.pvalues, index=param_names),
    })

    return {
        "panel_sample": df_panel_sample,
        "pooled_model": pooled_model,
        "robust_model": robust_model,
        "summary_table": summary_table,
    }


def exercise_10_fixed_effects_robust(df: pd.DataFrame):
    """
    Exercise 10:
    Estimate the Fama-French model with fixed effects and robust standard errors.

    Fixed effects are implemented via the within transformation:
        y_it* = x_it*' beta + u_it*
    where all variables are demeaned within main_strategy.

    Robust standard errors are clustered by main_strategy.

    Returns
    -------
    dict with:
        - panel_sample
        - within_sample
        - fe_model
        - fe_model_robust
        - summary_table
    """
    df_panel_sample = _get_panel_sample(df).copy()
    df_panel_sample = df_panel_sample.sort_values(
        ["main_strategy", "date"]
    ).reset_index(drop=True)

    # Within transformation: demean within each strategy
    group_col = "main_strategy"
    y_col = "log_excess_return"
    x_cols = ["mktrf", "smb", "hml"]

    within_sample = df_panel_sample.copy()

    within_sample[f"{y_col}_within"] = (
        within_sample[y_col]
        - within_sample.groupby(group_col)[y_col].transform("mean")
    )

    for col in x_cols:
        within_sample[f"{col}_within"] = (
            within_sample[col]
            - within_sample.groupby(group_col)[col].transform("mean")
        )

    X_within = within_sample[[f"{col}_within" for col in x_cols]]
    y_within = within_sample[f"{y_col}_within"]

    # No intercept in FE-within regression 
    # => NOT NEEDED: X = sm.add_constant(X, has_constant="add")
    fe_model = sm.OLS(y_within, X_within).fit()

    fe_model_robust = fe_model.get_robustcov_results(
        cov_type="cluster",
        groups=within_sample[group_col],
        use_correction=True,
    )

    param_names = list(X_within.columns)

    coef_hat = pd.Series(fe_model_robust.params, index=param_names)
    std_error_robust = pd.Series(fe_model_robust.bse, index=param_names)
    t_stat_robust = pd.Series(fe_model_robust.tvalues, index=param_names)
    p_value_robust = pd.Series(fe_model_robust.pvalues, index=param_names)

    summary_table = pd.DataFrame({
        "coef_hat": coef_hat,
        "std_error_fe": pd.Series(fe_model.bse, index=param_names),
        "p_value_fe": pd.Series(fe_model.pvalues, index=param_names),
        "std_error_robust": std_error_robust,
        "t_stat_robust": t_stat_robust,
        "p_value_robust": p_value_robust,
        "significant_5pct_fe": pd.Series(fe_model.pvalues, index=param_names) < 0.05,
        "significant_5pct_robust": p_value_robust < 0.05,
    })

    return {
        "panel_sample": df_panel_sample,
        "within_sample": within_sample,
        "fe_model": fe_model,
        "fe_model_robust": fe_model_robust,
        "summary_table": summary_table,
    }


def exercise_11_recover_strategy_alphas(df: pd.DataFrame):
    """
    Exercise 11:
    Recover the strategy-group alphas from the fixed-effects model of Exercise 10
    and identify the strategy with the largest outperformance.

    Uses:
        alpha_i_hat = mean(y_i) - mean(x_i)' beta_hat

    Returns
    -------
    dict with:
        - panel_sample
        - within_sample
        - fe_model
        - fe_model_robust
        - strategy_alphas
        - best_strategy
        - summary_table
    """
    ex10_result = exercise_10_fixed_effects_robust(df)

    panel_sample = ex10_result["panel_sample"].copy()
    within_sample = ex10_result["within_sample"].copy()
    fe_model = ex10_result["fe_model"]
    fe_model_robust = ex10_result["fe_model_robust"]

    group_col = "main_strategy"
    y_col = "log_excess_return"
    x_cols = ["mktrf", "smb", "hml"]

    beta_hat = pd.Series(
        fe_model.params.values,
        index=x_cols,
    )

    group_means = (
        panel_sample.groupby(group_col)[[y_col] + x_cols]
        .mean()
        .rename(columns={y_col: "y_bar"})
    )

    strategy_alphas = group_means.copy()
    strategy_alphas["alpha_hat"] = (
        strategy_alphas["y_bar"]
        - strategy_alphas[x_cols].mul(beta_hat, axis=1).sum(axis=1)
    )

    strategy_alphas = strategy_alphas.sort_values(
        "alpha_hat", ascending=False
    ).reset_index()

    strategy_alphas["rank"] = np.arange(1, len(strategy_alphas) + 1)

    best_strategy = strategy_alphas.loc[0, "main_strategy"]
    best_alpha = strategy_alphas.loc[0, "alpha_hat"]

    summary_table = strategy_alphas[[
        "rank", "main_strategy", "alpha_hat", "y_bar", "mktrf", "smb", "hml"
    ]].copy()

    return {
        "panel_sample": panel_sample,
        "within_sample": within_sample,
        "fe_model": fe_model,
        "fe_model_robust": fe_model_robust,
        "strategy_alphas": strategy_alphas,
        "best_strategy": {
            "main_strategy": best_strategy,
            "alpha_hat": best_alpha,
        },
        "summary_table": summary_table,
    }