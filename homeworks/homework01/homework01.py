import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


def wti_plot(df): 
    plt.plot(df["Date"], df["WTI_monthly_log_return"])
    plt.show()


if __name__ == "__main__":
    # 1.)
    df = pd.read_csv("./homework01/s1_data_hw.csv", encoding="utf-8-sig")
    monthly_returns, df_adjusted = monthly_log_return(df)
    print(monthly_returns.head())

    # 2.) 
    wti_plot(df_adjusted)


