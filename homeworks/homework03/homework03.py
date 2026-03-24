import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import norm
from statsmodels.tsa.ar_model import ar_select_order, AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox



def clean_data(df: pd.DataFrame) -> pd.DataFrame: 
    cleaned_df = df.copy()
    date_cols = [col for col in cleaned_df.columns if "date" in col.lower()]

    for col in date_cols: 
        cleaned_df[col] = pd.to_datetime(cleaned_df[col], yearfirst=True, errors="coerce")

    return cleaned_df


def exercise_0_exess_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date")
    rf_monthly = df["rf"] / 100 

    df["r_FINANC"] = df["FINANC"].pct_change()
    df["r_SP500"] = df["SP500"].pct_change()

    df["excess_r_FINANC"] = df["r_FINANC"] - rf_monthly
    df["excess_r_SP500"] = df["r_SP500"] - rf_monthly

    df = df[df["date"].dt.year > 1999].copy()

    return df


def exercise_1_AR(df: pd.DataFrame) -> pd.DataFrame: 
    sample = df[
        (df["date"] >= "2000-01-01") & (df["date"] <= "2008-12-31")
    ].copy()

    y = sample["excess_r_FINANC"].dropna()

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))

    plot_acf(y, lags=15, alpha=0.05, ax=ax[0])
    ax[0].set_title("ACF of excess_r_FINANC (2000:01-2008:12)")

    plot_pacf(y, lags=15, alpha=0.05, method="ywm", ax=ax[1])
    ax[1].set_title("PACF of excess_r_FINANC (2000:01-2008:12)")

    plt.tight_layout()
    plt.show()


# -----------------------------------
# --- manual computation of the ACF and the PACF ---
def sample_acf(y, nlags=15):
    y = pd.Series(y).dropna().to_numpy(dtype=float)
    T = len(y)
    ybar = y.mean()

    gamma0 = np.sum((y - ybar) ** 2) / T
    acf_vals = [1.0]

    for k in range(1, nlags + 1):
        gammak = np.sum((y[k:] - ybar) * (y[:-k] - ybar)) / T
        acf_vals.append(gammak / gamma0)

    return np.array(acf_vals)


def ols_beta(X, y):
    return np.linalg.inv(X.T @ X) @ (X.T @ y)


def sample_pacf_via_ols(y, nlags=15):
    y = pd.Series(y).dropna().to_numpy(dtype=float)
    T = len(y)

    pacf_vals = [1.0]

    for p in range(1, nlags + 1):
        # dependent variable: y_t for t = p, ..., T-1
        Y = y[p:]

        # regressors: constant + y_{t-1}, ..., y_{t-p}
        X_lags = []
        for j in range(1, p + 1):
            X_lags.append(y[p - j:T - j])

        X = np.column_stack([np.ones(len(Y))] + X_lags)

        beta = ols_beta(X, Y)

        # last slope coefficient = PACF at lag p
        pacf_vals.append(beta[-1])

    return np.array(pacf_vals)


def _plot_corr(vals, title, T):
    lags = np.arange(len(vals))
    band = 1.96 / np.sqrt(T)   # approximate 95% individual band

    plt.figure(figsize=(8, 4))
    plt.axhline(0.0)
    plt.axhline(band, linestyle="--")
    plt.axhline(-band, linestyle="--")
    plt.vlines(lags, 0, vals)
    plt.scatter(lags, vals)
    plt.title(title)
    plt.xlabel("lag")
    plt.show()


def question_1_manual(df):
    sample = df[(df["date"] >= "2000-01-01") & (df["date"] <= "2008-12-31")].copy()
    y = sample["excess_r_FINANC"].dropna()

    acf_vals = sample_acf(y, nlags=15)
    pacf_vals = sample_pacf_via_ols(y, nlags=15)

    T = len(y)
    _plot_corr(acf_vals, "Manual ACF: excess_r_FINANC", T)
    _plot_corr(pacf_vals, "Manual PACF: excess_r_FINANC", T)

    return acf_vals, pacf_vals
# -----------------------------------


def exercise_2_p_selection(df):
    sample = df[(df["date"] >= "2000-01-01") & (df["date"] <= "2008-12-31")].copy()

    y = sample["excess_r_FINANC"].dropna()
    # y = y - y.mean()

    sel = ar_select_order(y, maxlag=15, ic="aic", trend="c", old_names=False)

    lags = sel.ar_lags
    p = 0 if lags is None else max(lags)

    return p


def exercise_3_demeaned_ar(df: pd.DataFrame, p) -> pd.DataFrame: 
    sample = df[(df["date"] >= "2000-01-01") & (df["date"] <= "2008-12-31")].copy()

    y = sample["excess_r_FINANC"].dropna().reset_index(drop=True)
    y_dm = y - y.mean()

    # ATTENTION: since we use y do be demeaned 
    # => trend = no trend i.e. the constant is not needed
    ar_model = AutoReg(y_dm, lags=p, trend="n", hold_back=p, old_names=False).fit()

    residuals = ar_model.resid

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # 1. Time plot
    axes[0].plot(residuals)
    axes[0].set_title("Residual Time Plot")
    axes[0].axhline(0, color='red', linestyle='--')

    # 2. ACF
    plot_acf(residuals, lags=15, ax=axes[1])
    axes[1].set_title("ACF of Residuals")

    # 3. Histogram
    axes[2].hist(residuals, bins=20, edgecolor='black')
    axes[2].set_title("Residual Histogram")

    plt.tight_layout()
    plt.show()

    return ar_model    


def exercise_4_forecast_2009_01(df: pd.DataFrame, ar_model, p):
    sample = df[(df["date"] >= "2000-01-01") & (df["date"] <= "2008-12-31")].copy()

    y = sample["excess_r_FINANC"].dropna().reset_index(drop=True)
    y_mean = y.mean()

    forecast_dm = float(ar_model.forecast(steps=1).iloc[0])
    forecast = y_mean + forecast_dm

    sigma_hat = np.sqrt(ar_model.sigma2)
    z90 = norm.ppf(0.95)

    return {
        "forecast_date": "2009-01",
        "forecast_excess_r_FINANC": forecast,
        "ci90_lower": forecast - z90 * sigma_hat,
        "ci90_upper": forecast + z90 * sigma_hat,
    }

def exercise_5_out_of_sample_forecast(df: pd.DataFrame, p):
    data = df.sort_values("date").copy()

    # keep only rows where the target exists
    data = data.loc[data["excess_r_FINANC"].notna(), ["date", "excess_r_FINANC"]].copy()
    data = data.reset_index(drop=True)

    forecasts = []

    # first out-of-sample forecast is for 2009:01
    oos_mask = data["date"] >= pd.Timestamp("2009-01-01")
    oos_idx = data.index[oos_mask]

    for idx in oos_idx:
        forecast_date = data.loc[idx, "date"]

        # estimation sample: all data strictly before the forecast month
        train = data.loc[data["date"] < forecast_date, "excess_r_FINANC"].reset_index(drop=True)

        # demean inside the current estimation window
        y_mean = train.mean()
        y_dm = train - y_mean

        # re-estimate AR(p) recursively on expanding window
        ar_model = AutoReg(y_dm, lags=p, trend="n", old_names=False).fit()

        # one-step-ahead forecast of demeaned series
        forecast_dm = float(ar_model.forecast(steps=1).iloc[0])

        # transform back to original excess-return scale
        forecast = y_mean + forecast_dm

        actual = float(data.loc[idx, "excess_r_FINANC"])
        error = actual - forecast

        forecasts.append(
            {
                "date": forecast_date,
                "forecast_excess_r_FINANC": forecast,
                "actual_excess_r_FINANC": actual,
                "forecast_error": error,
                "squared_error": error ** 2,
            }
        )

    forecasts_df = pd.DataFrame(forecasts)
    rmse = float(np.sqrt(forecasts_df["squared_error"].mean()))

    return forecasts_df, rmse


def exercise_6_stationarity_tests(
    df: pd.DataFrame,
    ar_model,
    start_date: str = "2000-01-01",
    end_date: str = "2019-09-30",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.sort_values("date").copy()
    data = data[data["date"] >= pd.Timestamp(start_date)].copy()
    data = data[data["date"] <= pd.Timestamp(end_date)].copy()


    data["pd"] = np.log(data["SP500"] / data["SPDIV"])
    y = data["pd"].dropna().reset_index(drop=True)

    # ADF (see lecture notes): H0 = unit root / non-stationary
    # constant, no trend  -> regression="c"
    adf_stat, adf_pvalue, adf_usedlag, adf_nobs, adf_crit, adf_icbest = adfuller(
        y,
        regression="c",
        autolag="AIC",
    )

    # KPSS (see lecture notes): H0 = level-stationary (stationary around a constant)
    # constant, no trend -> regression="c"
    kpss_stat, kpss_pvalue, kpss_usedlag, kpss_crit = kpss(
        y,
        regression="c",
        nlags="auto",
    )

    residuals = ar_model.resid
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(residuals); axes[0].axhline(0, color="red", ls="--"); axes[0].set_title("Residuals")
    plot_acf(residuals, lags=15, alpha=0.05, ax=axes[1]); axes[1].set_title("ACF of Residuals")
    axes[2].hist(residuals, bins=20, edgecolor="black", density=True); axes[2].set_title("Histogram")
    plt.tight_layout(); plt.show()

    print(acorr_ljungbox(residuals, lags=[10, 15, 20], return_df=True))

    results = pd.DataFrame(
        [
            {
                "test": "ADF",
                "null_hypothesis": "unit root / non-stationary",
                "statistic": adf_stat,
                "critical_value_1pct": adf_crit["1%"],
                "reject_at_1pct": adf_stat < adf_crit["1%"],
                "decision_rule": "reject if statistic < critical value",
            },
            {
                "test": "KPSS",
                "null_hypothesis": "level-stationary around a constant",
                "statistic": kpss_stat,
                "critical_value_1pct": kpss_crit["1%"],
                "reject_at_1pct": kpss_stat > kpss_crit["1%"],
                "decision_rule": "reject if statistic > critical value",
            },
        ]
    )

    return data, results


def exercise_7_special_factor_model(
    df: pd.DataFrame,
    start_date: str = "2000-01-01",
    end_date: str = "2008-12-31",
):
    data = df.sort_values("date").copy()

    # Q7 uses lagged predictors
    data["lag_excess_r_SP500"] = data["excess_r_SP500"].shift(1)
    data["lag_eurusd"] = data["eurusd"].shift(1)
    data["lag_VIX"] = data["VIX"].shift(1)
    data["lag_pd"] = data["pd"].shift(1)

    # estimation sample: 2000:01 to 2008:12
    sample = data[
        (data["date"] >= pd.Timestamp(start_date)) &
        (data["date"] <= pd.Timestamp(end_date))
    ].copy()

    sample = sample[
        [
            "date",
            "excess_r_FINANC",
            "lag_excess_r_SP500",
            "lag_eurusd",
            "lag_VIX",
            "lag_pd",
        ]
    ].dropna().reset_index(drop=True)

    y = sample["excess_r_FINANC"]
    X = sample[
        [
            "lag_excess_r_SP500",
            "lag_eurusd",
            "lag_VIX",
            "lag_pd",
        ]
    ]
    X = sm.add_constant(X, has_constant="add")

    model = sm.OLS(y, X).fit()

    # forecast 2009:01 using predictor values from 2008:12
    row_2008_12 = data.loc[data["date"] == pd.Timestamp("2008-12-31")].copy()

    Xf = pd.DataFrame(
        {
            "const": [1.0],
            "lag_excess_r_SP500": [float(row_2008_12["excess_r_SP500"].iloc[0])],
            "lag_eurusd": [float(row_2008_12["eurusd"].iloc[0])],
            "lag_VIX": [float(row_2008_12["VIX"].iloc[0])],
            "lag_pd": [float(row_2008_12["pd"].iloc[0])],
        }
    )

    pred = model.get_prediction(Xf)
    pred_summary = pred.summary_frame(alpha=0.05)

    return {
        "model": model,
        "forecast_date": "2009-01",
        "forecast_excess_r_FINANC": float(pred_summary["mean"].iloc[0]),
        "ci95_lower": float(pred_summary["obs_ci_lower"].iloc[0]),
        "ci95_upper": float(pred_summary["obs_ci_upper"].iloc[0]),
    }


def exercise_8_oos_factor_model(df: pd.DataFrame):
    data = df.sort_values("date").copy()

    # same regressors as in exercise 7
    data["lag_excess_r_SP500"] = data["excess_r_SP500"].shift(1)
    data["lag_eurusd"] = data["eurusd"].shift(1)   # keep consistent with your ex. 7
    data["lag_VIX"] = data["VIX"].shift(1)
    data["lag_pd"] = data["pd"].shift(1)

    cols = [
        "date",
        "excess_r_FINANC",
        "lag_excess_r_SP500",
        "lag_eurusd",
        "lag_VIX",
        "lag_pd",
    ]
    data = data[cols].dropna().reset_index(drop=True)

    forecasts = []

    # first OOS forecast is 2009:01
    oos_idx = data.index[data["date"] >= pd.Timestamp("2009-01-01")]

    for idx in oos_idx:
        forecast_date = data.loc[idx, "date"]

        # estimation sample: all data strictly before the forecast month
        train = data.loc[data["date"] < forecast_date].copy()

        y_train = train["excess_r_FINANC"]
        X_train = train[
            ["lag_excess_r_SP500", "lag_eurusd", "lag_VIX", "lag_pd"]
        ]
        X_train = sm.add_constant(X_train, has_constant="add")

        model = sm.OLS(y_train, X_train).fit()

        # predictors known at t-1 for forecasting month t
        Xf = data.loc[[idx], ["lag_excess_r_SP500", "lag_eurusd", "lag_VIX", "lag_pd"]]
        Xf = sm.add_constant(Xf, has_constant="add")

        forecast = float(model.predict(Xf).iloc[0])
        actual = float(data.loc[idx, "excess_r_FINANC"])
        error = actual - forecast

        forecasts.append(
            {
                "date": forecast_date,
                "forecast_excess_r_FINANC": forecast,
                "actual_excess_r_FINANC": actual,
                "forecast_error": error,
                "squared_error": error ** 2,
            }
        )

    forecasts_df = pd.DataFrame(forecasts)
    rmse = float(np.sqrt(forecasts_df["squared_error"].mean()))

    # required plot: predicted vs realized
    plt.figure(figsize=(10, 5))
    plt.plot(
        forecasts_df["date"],
        forecasts_df["actual_excess_r_FINANC"],
        label="Realized",
    )
    plt.plot(
        forecasts_df["date"],
        forecasts_df["forecast_excess_r_FINANC"],
        label="Predicted",
    )
    plt.title("Exercise 8: OOS forecasts from factor model")
    plt.xlabel("Date")
    plt.ylabel("Excess return")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return forecasts_df, rmse


import numpy as np
import pandas as pd
from scipy.stats import norm


def exercise_9_diebold_mariano(
    forecasts_ar: pd.DataFrame,
    forecasts_factor: pd.DataFrame,
    h: int = 1,
):
    """
    Two-sided Diebold-Mariano test with quadratic loss.

    Required columns in both dataframes:
        - "date"
        - "forecast_error"

    h = forecast horizon. For this homework, h=1.
    """

    # Align same OOS months from both models
    df = forecasts_ar[["date", "forecast_error"]].rename(
        columns={"forecast_error": "e_ar"}
    ).merge(
        forecasts_factor[["date", "forecast_error"]].rename(
            columns={"forecast_error": "e_factor"}
        ),
        on="date",
        how="inner",
    ).sort_values("date").reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No overlapping forecast dates between the two models.")

    # Quadratic loss differential: AR loss - Factor loss
    df["d"] = df["e_ar"] ** 2 - df["e_factor"] ** 2

    d = df["d"].to_numpy()
    T = len(d)
    d_bar = d.mean()

    # Long-run variance estimate with Bartlett weights
    # For h=1, this is just gamma_0
    gamma0 = np.mean((d - d_bar) ** 2)
    lrv = gamma0

    for lag in range(1, h):
        cov = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        weight = 1.0 - lag / h
        lrv += 2.0 * weight * cov

    dm_stat = d_bar / np.sqrt(lrv / T)
    p_value = 2.0 * (1.0 - norm.cdf(abs(dm_stat)))

    # Interpretation of sign
    if d_bar > 0:
        better_model = "Factor model"
    elif d_bar < 0:
        better_model = "AR model"
    else:
        better_model = "Tie"

    conclusion = (
        "Reject equal forecast accuracy at 5%"
        if p_value < 0.05
        else "Do not reject equal forecast accuracy at 5%"
    )

    return {
        "n_obs": T,
        "mean_loss_diff": float(d_bar),
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "better_model_by_average_quadratic_loss": better_model,
        "conclusion_5pct": conclusion,
        "details": df,
    }


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent / "s3_data.txt"
    df = pd.read_csv(data_path, sep="\t")
    df = clean_data(df)
    # print(df.head())
