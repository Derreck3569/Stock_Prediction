import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import pandas_datareader.data as web
import requests
#from datetime import datetime, timedelta
import os
import sys

import os
import sys


# ... continue with your script ...

def extract_features():

    return_period = 5
    
    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")
    stk_tickers = ['NVDA','AAPL', 'TMC','FDX']
    ccy_tickers = ['DEXJPUS', 'DEXUSUK']
    idx_tickers = ['SP500', 'DJIA', 'VIXCLS']
    
    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)
    #stk_data = web.DataReader(stk_tickers, 'yahoo')
    ccy_data = web.DataReader(ccy_tickers, 'fred', start=START_DATE, end=END_DATE)
    idx_data = web.DataReader(idx_tickers, 'fred', start=START_DATE, end=END_DATE)

    Y = np.log(stk_data.loc[:, ('Adj Close', 'NVDA')]).diff(return_period).shift(-return_period)
    Y.name = 'NVDA_Future_Return'
    
    # 4. Create base features (X) - Log returns of your other stocks and indices
    X1 = np.log(stk_data.loc[:, ('Adj Close', ('AAPL', 'TMC', 'FDX'))]).diff(return_period)
    X1.columns = X1.columns.droplevel()
    X2 = np.log(ccy_data).diff(return_period)
    X3 = np.log(idx_data).diff(return_period)
    X = pd.concat([X1, X2, X3], axis=1)
    
    # 5. Add 4 Custom Technical Features for NVDA
    # Feature 1: 14-day Simple Moving Average (Trend)
    X['NVDA_SMA_14'] = stk_data.loc[:, ('Adj Close', 'NVDA')].rolling(window=14).mean()
    # Feature 2: Volatility (Difference between High and Low)
    X['NVDA_Volatility'] = stk_data.loc[:, ('High', 'NVDA')] - stk_data.loc[:, ('Low', 'NVDA')]
    # Feature 3: Momentum (14-day percentage change)
    X['NVDA_Momentum_14'] = stk_data.loc[:, ('Adj Close', 'NVDA')].pct_change(14)
    # Feature 4: End of Quarter (Calendar feature)
    X['Is_Quarter_End'] = X.index.is_quarter_end.astype(int)
    
    # Combine, clean, and align the dataset
    
    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]
    Y = dataset.loc[:, Y.name]
    X = dataset.loc[:, X.columns]
    dataset.index.name = 'Date'
    #dataset.to_csv(r"./test_data.csv")
    features = dataset.sort_index()
    features = features.reset_index(drop=True)
    features = features.iloc[:,1:]
    return features


def get_bitcoin_historical_prices(days = 60):
    
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily' # Ensure we get daily granularity
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['Timestamp', 'Close Price (USD)'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()
    df = df[['Date', 'Close Price (USD)']].set_index('Date')
    return df







