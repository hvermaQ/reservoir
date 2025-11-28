import pandas as pd
import numpy as np
from scipy.stats import norm
import glob
import os
data_folder = './2013-01'  # Path to optiondata.org files

def generate_data(ticker):
    # --- 1. LOAD ALL OPTIONS FILES ---
    option_files = sorted(glob.glob(os.path.join(data_folder, '*options.csv')))
    option_dfs = []
    for f in option_files:
        daily_df = pd.read_csv(f)
        daily_df['date'] = pd.to_datetime(os.path.basename(f).split('options')[0])
        option_dfs.append(daily_df)
    opts = pd.concat(option_dfs, ignore_index=True)
    opts['date'] = pd.to_datetime(opts['quote_date'])  # Use quote_date if available

    # --- 2. LOAD ALL STOCK FILES ---
    stock_files = sorted(glob.glob(os.path.join(data_folder, '*stocks.csv')))
    stock_dfs = []
    for f in stock_files:
        sdf = pd.read_csv(f)
        sdf['date'] = pd.to_datetime(os.path.basename(f).split('stocks')[0])
        stock_dfs.append(sdf)
    stocks = pd.concat(stock_dfs, ignore_index=True)
    stocks['date'] = pd.to_datetime(stocks['date'])

    # --- 3. AUTO-DETECT TICKER COLUMN IN STOCKS ---
    stock_ticker_col = None
    for candidate in ['underlying', 'symbol', 'ticker', 'name']:
        if candidate in stocks.columns:
            stock_ticker_col = candidate
            break
    if not stock_ticker_col:
        raise KeyError("No ticker column found in stocks file! Columns are: {}".format(list(stocks.columns)))

    # --- 4. FILTER FOR SELECTED TICKER AND MERGE UNDERLYING PRICE ---
    opts_ticker = opts[opts['underlying'] == ticker].copy()
    merged = pd.merge(
        opts_ticker,
        stocks[[stock_ticker_col, 'date', 'close']],
        left_on=['underlying', 'date'],
        right_on=[stock_ticker_col, 'date'],
        how='left'
    )
    merged = merged.rename(columns={'close': 'spot'})
    if stock_ticker_col != 'underlying':
        merged = merged.drop(columns=[stock_ticker_col])

    # --- 5. DETERMINE STRIKE PRICE WINDOW USING FIRST DAY SPOT PRICE ---
    first_day = merged['date'].min()
    first_day_spot = merged.loc[merged['date'] == first_day, 'spot'].iloc[0]
    low_strike, high_strike = 0.8 * first_day_spot, 1.2 * first_day_spot

    # --- 6. FILTER STRIKES WITHIN ±20% OF FIRST-DAY SPOT ---
    filtered = merged[(merged['strike'] >= low_strike) & (merged['strike'] <= high_strike)].copy()

    # --- 7. CREATE MIDPRICE COLUMN ---
    filtered['mid'] = (filtered['bid'] + filtered['ask']) / 2

    # --- 8. RUN BLACK-SCHOLES AND DEVIATION FOR CALL OPTIONS ---
    def black_scholes_call(S, K, T, r, sigma):
        T = max(T, 1e-6)
        if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
            return np.nan
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def days_to_expiry(row):
        expiry = pd.to_datetime(row['expiration'])
        quote = pd.to_datetime(row['date'])
        return max((expiry - quote).days / 365.0, 1e-6)

    r = 0.05  # example risk-free rate

    def calc_bs_and_deviation(row):
        S = row['spot']
        K = row['strike']
        T = days_to_expiry(row)
        sigma = row['implied_volatility']
        market_price = row['mid']
        if np.isnan(market_price) or np.isnan(S) or np.isnan(K) or np.isnan(sigma):
            return pd.Series([np.nan, np.nan])
        bs = black_scholes_call(S, K, T, r, sigma)
        deviation = market_price - bs
        return pd.Series([bs, deviation])

    filtered_calls = filtered[filtered['type'] == 'call'].copy()
    filtered_calls[['BS_call', 'Deviation']] = filtered_calls.apply(calc_bs_and_deviation, axis=1)

    return filtered_calls