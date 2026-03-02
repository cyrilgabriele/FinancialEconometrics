import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def monthly_log_return(df):
    """Return monthly log returns from January 2021 onward."""
    data = df.copy()
    data = data.rename(columns=lambda col: col.strip().replace("\ufeff", ""))
    data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%y")
    data = data.sort_values("Date")
    start_date = pd.Timestamp("2021-01-01")
    data = data[data["Date"] >= start_date].copy()
    data["RF_monthly"] = (data["RF"] / 100) / 12

    assets = {
        "SP500": "SPY US INDEX",
        "STOXX600": "SXXP Index",
        "NIKKEI": "NKY INDEX",
        "GOLD": "XAU CURNCY",
        "WTI": "CL1 COMDTY",
    }

    for label, column in assets.items():
        log_returns = np.log(data[column]).diff().round(4)
        data[f"{label}_monthly_log_return"] = log_returns

    result_cols = ["Date", "RF_monthly"] + [f"{label}_monthly_log_return" for label in assets]
    result = data[result_cols].dropna().reset_index(drop=True)

    sto_mean = result["STOXX600_monthly_log_return"].mean()
    nky_mean = result["NIKKEI_monthly_log_return"].mean()
    print(f"Mean monthly log return STOXX600: {sto_mean:.4f}")
    print(f"Mean monthly log return NIKKEI: {nky_mean:.4f}")

    return result, data


def wti_plots(df):
    price_series = df[["Date", "CL1 COMDTY"]].dropna()
    return_series = df.dropna(subset=["WTI_monthly_log_return"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(price_series["Date"], price_series["CL1 COMDTY"], color="tab:blue")
    axes[0].set_title("WTI Price Index")
    axes[0].set_ylabel("Index Level")

    axes[1].plot(return_series["Date"], return_series["WTI_monthly_log_return"], color="tab:orange")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("WTI Monthly Log Returns")
    axes[1].set_ylabel("Log Return")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    plt.show()

    max_idx = return_series["WTI_monthly_log_return"].abs().idxmax()
    max_date = return_series.loc[max_idx, "Date"].strftime("%b %Y")
    max_value = return_series.loc[max_idx, "WTI_monthly_log_return"]
    print(
        "WTI returns show a pronounced spike of "
        f"{max_value:.2%} around {max_date}, highlighting their volatility."
    )


def testing_avg_ret_stoxx(df): 
    # HYPOTHESIS:
    # H_0:= mu = 0 
    # H_1:= mu != 0

    significance_level = 0.05
    t, p = stats.ttest_1samp(df["STOXX600_monthly_log_return"], popmean=0.0)
    if p < significance_level:
        print("REJECT H_0")
    else: 
        print("FAILED TO REJECT H_0")
    
    # *spoiler*: we fail to reject the H_0 => H_0 is likely


if __name__ == "__main__":
    # 1.)
    df = pd.read_csv("./homework01/s1_data_hw.csv", encoding="utf-8-sig")
    monthly_returns, df_adjusted = monthly_log_return(df)
    print(monthly_returns.head())

    # 2.) 
    wti_plots(df_adjusted)

    # 3.) 
    testing_avg_ret_stoxx(df_adjusted)

