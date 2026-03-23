import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import norm
from statsmodels.tsa.ar_model import ar_select_order, AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss



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


def exercise_3_demeaned_ar(df: pd.DatFrame, p) -> pd.Dataframe: 
    sample = df[(df["date"] >= "2000-01-01") & (df["date"] <= "2008-12-31")].copy()

    y = sample["excess_r_FINANC"].dropna().reset_index(drop=True)
    y_dm = y - y.mean()

    # ATTENTION: since we use y do be demeaned 
    # => trend = no trend i.e. the constant is not needed
    ar_model = AutoReg(y_dm, lags=p, trend="n", hold_back=p, old_names=False).fit()
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
    start_date: str = "2000-01-01",
    end_date: str = "2008-12-31",
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



if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent / "s3_data.txt"
    df = pd.read_csv(data_path, sep="\t")
    df = clean_data(df)
    print(df.head())
