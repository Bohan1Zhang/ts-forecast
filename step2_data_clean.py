import os
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf


def ensure_dirs():
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)


def download_prices(tickers, start="2012-01-01", end=None):

    import time
    last_err = None
    for attempt in range(3):
        try:
            df = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,     
                progress=False,
                group_by="column",    
                threads=True,
            )
            
            return df
        except Exception as e:
            last_err = e
            time.sleep(2)            
    raise last_err


def flatten_ohlcv(df_multi, tickers):
    
    
    if isinstance(df_multi.columns, pd.MultiIndex):
        level0 = list(df_multi.columns.levels[0])
        if level0[0] in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            df_multi = df_multi.swaplevel(0, 1, axis=1)
            df_multi = df_multi.sort_index(axis=1)

    wanted_fields = ["Adj Close", "Close", "Open", "High", "Low", "Volume"]
    cols = [(t, f) for t in tickers for f in wanted_fields if (t, f) in df_multi.columns]

    flat = df_multi[cols].copy()
    
    flat.columns = [f"{t}_{f.replace(' ', '')}" for (t, f) in cols]

    print("Flattened columns sample:", flat.columns[:6].tolist())
    return flat



def align_calendar(df):
    
    return df.ffill()


def add_basic_returns(df):
    
    def pick_price(prefix):
        if f"{prefix}_AdjClose" in df.columns:
            return f"{prefix}_AdjClose"
        elif f"{prefix}_Close" in df.columns:
            return f"{prefix}_Close"
        else:
            raise ValueError(f"Missing price columns for {prefix}: need {prefix}_AdjClose or {prefix}_Close")

    aapl_price = pick_price("AAPL")
    spy_price  = pick_price("SPY")

    df["AAPL_log_ret"] = np.log(df[aapl_price] / df[aapl_price].shift(1))
    df["SPY_log_ret"]  = np.log(df[spy_price]  / df[spy_price].shift(1))

    
    vix_price = None
    if "VIX_AdjClose" in df.columns:
        vix_price = "VIX_AdjClose"
    elif "VIX_Close" in df.columns:
        vix_price = "VIX_Close"

    if vix_price is not None:
        df["VIX_level"]  = df[vix_price]
        df["VIX_change"] = np.log(df[vix_price] / df[vix_price].shift(1))
    else:
        df["VIX_level"]  = np.nan
        df["VIX_change"] = np.nan

    return df


def make_targets(df):
    
    df["next_log_ret"] = df["AAPL_log_ret"].shift(-1)
    df["next_ret_up"]  = (df["next_log_ret"] > 0).astype("Int64")
    return df


def select_training_view(df):
    
    cols = [
        # core features at time t
        "AAPL_log_ret", "SPY_log_ret", "VIX_level", "VIX_change",
        # optional raw columns (helpful for plotting later)
        "AAPL_AdjClose", "SPY_AdjClose", "VIX_AdjClose"
    ]
    cols = [c for c in cols if c in df.columns]  # filter existing
    out = df[cols + ["next_log_ret", "next_ret_up"]].copy()
    out = out.dropna().astype(float)  # drop start/end NA due to shift
    # Cast label back to int 0/1 for classifiers
    out["next_ret_up"] = (out["next_ret_up"] > 0).astype(int)
    return out


def main():
    ensure_dirs()
    tickers = ["AAPL", "SPY", "^VIX"]

    print(">>> Downloading prices ...")
    raw_multi = download_prices(tickers)
    print(f"Raw shape: {raw_multi.shape}")

    print(">>> Flattening OHLCV ...")
    wide = flatten_ohlcv(raw_multi, tickers)
    print(f"Wide shape: {wide.shape}")

    print(">>> Aligning calendars (forward-fill) ...")
    wide = align_calendar(wide)

    print(">>> Adding log returns and VIX features ...")
    wide = add_basic_returns(wide)

    print(">>> Building next-day targets ...")
    wide = make_targets(wide)

    print(">>> Selecting minimal training view ...")
    train_view = select_training_view(wide)

    # Save outputs
    out_path = Path("data/processed/clean_timeseries.csv")
    train_view.to_csv(out_path, index=True)
    print(f"Saved processed dataset to: {out_path.resolve()}")
    print("Preview:")
    print(train_view.head(10))


if __name__ == "__main__":
    main()
