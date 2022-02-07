import pandas as pd
import os
import talib


def icsa_shift(icsa_path='data/ICSA.csv', ohlcv_path='data/OHLC_NDX.csv'):
    """
    Load and shift the Initial Claims time series
    Initial Claims are published on Thursdays, but are refering to last Saturday
    That is why they require shifting in order not to commit forward looking bias
    Also ICSA values are discreet, they need to be forward filled with OHLCV data as the reference
    """
    icsa = pd.read_csv(icsa_path)
    icsa.columns = ['Date', 'ICSA']
    icsa.Date = pd.to_datetime(icsa.Date)
    icsa.set_index('Date', inplace=True)

    # shift the date 4 working days forward
    icsa = icsa.shift(4, freq='B')
    icsa.reset_index(inplace=True)

    # create one column df with dates from OHLCV dataframe
    ohlcv = pd.read_csv(ohlcv_path)
    df_date = pd.DataFrame(pd.to_datetime(ohlcv['Date']))
    del ohlcv

    # forward fill the values of ICSA for every day in OHLCV
    icsa = df_date.merge(icsa, how='left').fillna(method='ffill')
    del df_date

    if not os.path.isdir('data_preprocessed/'):
        os.mkdir('data_preprocessed/')
    icsa.to_csv('data_preprocessed/ICSA_preprocessed.csv', index=False)

    return 1


def technical_indicators(ohlcv_path="data/OHLC_NDX.csv"):
    """
    Calculate Technical Indicators from the ti_dict dictionary and export to .csv
    :param ohlcv_path: OHLCV dataset path
    """
    ohlcv = pd.read_csv(ohlcv_path)
    ti_dict = {
        'SMA5': talib.SMA(ohlcv.Close, 5),
        'SMA10': talib.SMA(ohlcv.Close, 10),
        'SMA50': talib.SMA(ohlcv.Close, 50),
        'EMA20': talib.EMA(ohlcv.Close, 20),
        'stoch5': talib.STOCH(ohlcv.High, ohlcv.Low, ohlcv.Close, 5, 3, 0, 3, 0)[0],
        'ADOSC': talib.ADOSC(ohlcv.High, ohlcv.Low, ohlcv.Close, ohlcv.Volume, fastperiod=3, slowperiod=10),
        'MACDhist': talib.MACD(ohlcv.Close, fastperiod=12, slowperiod=26, signalperiod=9)[2],
        'WILLR': talib.WILLR(ohlcv.High, ohlcv.Low, ohlcv.Close, timeperiod=14),
        'RSI': talib.RSI(ohlcv.Close, timeperiod=14),
        'MOM': talib.MOM(ohlcv.Close, timeperiod=10),
        'ROC': talib.ROC(ohlcv.Close, timeperiod=10),
        'OBV': talib.OBV(ohlcv.Close, ohlcv.Volume),
        'CCI': talib.CCI(ohlcv.High, ohlcv.Low, ohlcv.Close, timeperiod=14)
    }
    df_ti = pd.DataFrame(ti_dict, index=ohlcv.index)
    del ti_dict
    del ohlcv

    if not os.path.isdir('data_preprocessed/'):
        os.mkdir('data_preprocessed/')
    df_ti.to_csv('data_preprocessed/technical_indicators.csv', index=False)
    del df_ti

    return 1


def concatenate_dfs(df_dir='data_preprocessed/', exclude=['']):
    """
    Concatenate all of the .csv files in the given directory
    :param df_dir: Data Frame directory
    :param exclude: Exclude file names from the list of paths
    :return: Combined Dataframe withour NaN values
    """
    paths = [df_dir + x for x in os.listdir(df_dir) if x not in exclude]
    li = []
    for path in paths:
        df_buf = pd.read_csv(path)
        li.append(df_buf)
    del df_buf
    df_res = pd.concat(li, axis=1)
    df_res.dropna(inplace=True)
    del li
    df_res.to_csv("dataset.csv", index=False)
    return df_res


