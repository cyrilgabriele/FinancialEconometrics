from calendar import month

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
    axes[0].axhline(price_series["CL1 COMDTY"].mean(), color="black", linewidth=0.8, linestyle="--")
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
        f"{max_value:.2%} around {max_date}, highlighting their volatility.\n"
    )


def testing_avg_ret_stoxx(df): 
    # HYPOTHESIS:
    # H_0:= mu = 0 
    # H_1:= mu != 0

    significance_level = 0.05
    t, p = stats.ttest_1samp(df["STOXX600_monthly_log_return"], popmean=0.0)

    print(f"t-statistic: ${t}\n")
    print(f"p-value: ${p}\n")
    if p < significance_level:
        print("REJECT H_0\n")
    else: 
        print("FAILED TO REJECT H_0\n")
    
    # *spoiler*: we fail to reject the H_0 => H_0 is likely


def testing_CI_avg_ret_gold(df): 
    level = 0.95
    gold_returns = df["GOLD_monthly_log_return"].dropna()
    n = gold_returns.count()
    mean_value = gold_returns.mean()
    se = gold_returns.std(ddof=1) / np.sqrt(n)
    a, b = stats.t.interval(level, n - 1, loc=mean_value, scale=se)
    
    print(f"CI-Testing: 2.5th percentile: ${a}\n")
    print(f"CI-Testing: 97.5th percentile: ${b}\n")
    rf_mean = df["RF_monthly"].mean()
    print(f"Average monthly risk-free rate: {rf_mean:.4f}")
    if a > rf_mean:
        print("Gold's mean return exceeds the risk-free rate at the 95% confidence level.\n")
    else:
        print("Cannot claim Gold outperformed the risk-free rate at the 95% confidence level.\n")


def correlation_matrix_risky_assets(monthly_returns): 
    risky_cols = [
        "SP500_monthly_log_return",
        "STOXX600_monthly_log_return",
        "NIKKEI_monthly_log_return",
        "GOLD_monthly_log_return",
        "WTI_monthly_log_return",
    ]
    
    correlation_matrix = monthly_returns[risky_cols].corr()
    return correlation_matrix


def highest_kurtoisis(monthly_returns): 
    risky_cols = [
        "SP500_monthly_log_return",
        "STOXX600_monthly_log_return",
        "NIKKEI_monthly_log_return",
        "GOLD_monthly_log_return",
        "WTI_monthly_log_return",
    ]

    kurt = monthly_returns[risky_cols].kurtosis()
    asset_name = kurt.idxmax()
    max_value = kurt.max()
    return asset_name, max_value

def plot_kutosis_vs_normal(monthly_returns, asset_name): 
    returns = monthly_returns[f"{asset_name}"].dropna()
    # this is a MLE (maximum likelihood estimation)
    mu, sigma = stats.norm.fit(returns)

    # histogram
    mu_observed = returns.mean()
    sigma_observed = returns.std(ddof=1)
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=20, density=True, alpha=0.6, edgecolor="black", label=f"Histogram of returns ($\\mu$={mu_observed:.4f}, $\\sigma$={sigma_observed:.4f})")

    # fitted normal curve
    x = np.linspace(returns.min(), returns.max(), 500)
    y = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, y, linewidth=2, label=f"Fitted Normal($\\mu$={mu:.4f}, $\\sigma$={sigma:.4f})")

    plt.title(f"{asset_name}: Histogram with Fitted Normal Distribution")
    plt.xlabel("Monthly log return")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def highest_jarque_bera(monthly_returns): 
    risky_cols = [
        "SP500_monthly_log_return",
        "STOXX600_monthly_log_return",
        "NIKKEI_monthly_log_return",
        "GOLD_monthly_log_return",
        "WTI_monthly_log_return",
    ]

    jarque_bera_scores = {}


    for index in risky_cols: 
        jarque_bera_score = stats.jarque_bera(monthly_returns[f"{index}"]).statistic
        jarque_bera_scores[f"{index}"] = jarque_bera_score

    asset_name_max_JB = max(jarque_bera_scores, key=jarque_bera_scores.get)
    value_max_JB = jarque_bera_scores[f"{asset_name_max_JB}"]
    return asset_name_max_JB, value_max_JB


def hike_cut_identification(df): 
    df = df.sort_values("Date")
    df["delta_DFEDTARU"] = df["DFEDTARU"].diff()
    df["is_hike"] = (df["delta_DFEDTARU"] > 0).astype(int)
    df["plateau"] = (df["delta_DFEDTARU"] == 0).astype(int)
    df["is_cut"] = (df["delta_DFEDTARU"] < 0).astype(int)

    return df    


def label_hiking_months(df_hike_cut):
    df_hike_cut = df_hike_cut.sort_values("Date").reset_index(drop=True).copy()

    df_hike_cut["D_t"] = 0
    df_hike_cut["cycle_id"] = pd.NA

    in_cycle = False
    current_cycle_id = 0
    cycle_start_idx = None
    last_hike_idx = None
    pause_length = 0

    for i in range(len(df_hike_cut)):
        current_row = df_hike_cut.iloc[i]

        if current_row["is_hike"]:
            if not in_cycle:
                in_cycle = True
                current_cycle_id += 1
                cycle_start_idx = i

            last_hike_idx = i
            pause_length = 0
            continue

        if in_cycle and current_row["is_cut"]:
            df_hike_cut.loc[cycle_start_idx:last_hike_idx, "D_t"] = 1
            df_hike_cut.loc[cycle_start_idx:last_hike_idx, "cycle_id"] = current_cycle_id

            in_cycle = False
            cycle_start_idx = None
            last_hike_idx = None
            pause_length = 0
            continue

        if in_cycle:
            pause_length += 1

            if pause_length >= 6:
                df_hike_cut.loc[cycle_start_idx:last_hike_idx, "D_t"] = 1
                df_hike_cut.loc[cycle_start_idx:last_hike_idx, "cycle_id"] = current_cycle_id

                in_cycle = False
                cycle_start_idx = None
                last_hike_idx = None
                pause_length = 0

    if in_cycle and cycle_start_idx is not None and last_hike_idx is not None:
        df_hike_cut.loc[cycle_start_idx:last_hike_idx, "D_t"] = 1
        df_hike_cut.loc[cycle_start_idx:last_hike_idx, "cycle_id"] = current_cycle_id

    return df_hike_cut
        



if __name__ == "__main__":
    # 1.) + 2.)
    df = pd.read_csv("./homework01/s1_data_hw.csv", encoding="utf-8-sig")
    monthly_returns, df_adjusted = monthly_log_return(df)
    print(monthly_returns.head())

    # 3.) 
    # wti_plots(df_adjusted)

    # 4.) 
    testing_avg_ret_stoxx(monthly_returns)

    # 5.) 
    testing_CI_avg_ret_gold(monthly_returns)

    # 6.) 
    correlation_matrix = correlation_matrix_risky_assets(monthly_returns)
    wti_max_corr = (
        correlation_matrix["WTI_monthly_log_return"]
        .drop("WTI_monthly_log_return")
        .idxmax()
    )
    print(f"WTI max positiv correlation with: ${wti_max_corr}")

    sp500_max_corr = (
        correlation_matrix["SP500_monthly_log_return"]
        .drop("SP500_monthly_log_return")
        .idxmax()
    )
    print(f"SP500 max positiv correlation with: ${sp500_max_corr}")

    # 7.) 
    asset_name_max_kurt, value_max_kurt = highest_kurtoisis(monthly_returns)
    print(f"{asset_name_max_kurt} has the highest value with: {value_max_kurt.round(4)}")
    # plot_kutosis_vs_normal(monthly_returns, asset_name_max_kurt)
    # Interpretation: 
    # I mean the observed distribution is not normal but centered around the mean.
    
    
    # 8.) 
    asset_name_max_JB, value_max_JB = highest_jarque_bera(monthly_returns)
    print(f"highest Jarque Bera value has {asset_name_max_JB} with: {value_max_JB}")


    # 9.) 
    # a.) 
    df_hike_cut = hike_cut_identification(df)
    df_hikes = label_hiking_months(df_hike_cut)
    print(df_hike_cut.head())
    df_cycles = label_hiking_months(df_hikes)
    num_hiking_cycles = (df_cycles["D_t"] == 1).sum()
    print(f"number of hiking cycles = {num_hiking_cycles}")
