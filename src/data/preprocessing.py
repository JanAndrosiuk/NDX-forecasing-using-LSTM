from src.init_setup import *
import pandas as pd
import os
import talib


class TrainPrep:

    def __init__(self) -> None:
        self.setup = Setup()
        self.config = self.setup.config
        self.logger = logging.getLogger("Input Prep")
        self.logger.addHandler(logging.StreamHandler())
        self.icsa_df_raw_path = self.setup.ROOT_PATH + self.config["raw"]["IcsaRawDF"]
        self.candle_df_raw_path = self.setup.ROOT_PATH + self.config["raw"]["CandleRawDF"]

    def prep_icsa(self) -> int:
        """
        Load and process Initial Claims time series data
        Initial Claims are published on Thursdays, but refer to the latest Saturday
        In other words, data with saturday timestamp is visible on the next Thursday at the earliest
        That is why ICSA records require forward time shift to avoid forward-looking bias
        Also ICSA values are published less frequently than candlestick data
        Therefore, records need to be forward filled
        """
        self.logger.info(f"Preparing ICSA dataset")

        icsa_df = pd.read_csv(self.icsa_df_raw_path)
        icsa_df.columns = ['Date', 'ICSA']
        icsa_df.Date = pd.to_datetime(icsa_df.Date)
        icsa_df.set_index('Date', inplace=True)

        # shift the date 4 working days forward
        icsa_df = icsa_df.shift(4, freq='B')
        icsa_df.reset_index(inplace=True)

        # create one column with dates from OHLCV dataframe
        candle_df = pd.read_csv(self.candle_df_raw_path)
        df_date = pd.DataFrame(pd.to_datetime(candle_df['Date']))
        del candle_df

        # forward-fill ICSA records for every day in OHLCV
        icsa_df = df_date.merge(icsa_df, how='left').fillna(method='ffill')
        del df_date

        # delete empty head of icsa values
        icsa_df.dropna(subset=['ICSA'], inplace=True)

        prep_dir = self.setup.ROOT_PATH + self.config["prep"]["DataPreprocessedDir"]
        if not os.path.isdir(prep_dir):
            os.mkdir(prep_dir)
        icsa_df.to_csv(self.setup.ROOT_PATH + self.config["prep"]["FFilledIcsaDfCsv"], index=False)
        icsa_df.to_pickle(self.setup.ROOT_PATH + self.config["prep"]["FFilledIcsaDfPkl"])

        return 0

    def prep_tis(self) -> int:
        """
        Calculate Technical Indicators from the ti_dict dictionary and export to .csv
        """
        self.logger.info(f"Preparing TIs dataset")

        candle_df = pd.read_csv(self.candle_df_raw_path)
        ti_dict = {
            'Date': candle_df["Date"],
            'SMA5': talib.SMA(candle_df.Close, 5),
            'SMA10': talib.SMA(candle_df.Close, 10),
            'SMA50': talib.SMA(candle_df.Close, 50),
            'EMA20': talib.EMA(candle_df.Close, 20),
            'stoch5': talib.STOCH(candle_df.High, candle_df.Low, candle_df.Close, 5, 3, 0, 3, 0)[0],
            'ADOSC': talib.ADOSC(candle_df.High, candle_df.Low, candle_df.Close,
                                 candle_df.Volume, fastperiod=3, slowperiod=10),
            'MACDhist': talib.MACD(candle_df.Close, fastperiod=12, slowperiod=26, signalperiod=9)[2],
            'WILLR': talib.WILLR(candle_df.High, candle_df.Low, candle_df.Close, timeperiod=14),
            'RSI': talib.RSI(candle_df.Close, timeperiod=14),
            'MOM': talib.MOM(candle_df.Close, timeperiod=10),
            'ROC': talib.ROC(candle_df.Close, timeperiod=10),
            'OBV': talib.OBV(candle_df.Close, candle_df.Volume),
            'CCI': talib.CCI(candle_df.High, candle_df.Low, candle_df.Close, timeperiod=14)
        }

        # Create a dataframe from TI dictionary
        tis_df = pd.DataFrame(ti_dict, index=candle_df.index)
        del ti_dict
        del candle_df

        # delete empty head of icsa values
        tis_df.dropna(inplace=True)

        # Save Technical Indicators dataframe
        prep_dir = self.setup.ROOT_PATH + self.config["prep"]["DataPreprocessedDir"]
        if not os.path.isdir(prep_dir):
            os.mkdir(prep_dir)
        tis_df.to_csv(self.setup.ROOT_PATH + self.config["prep"]["TisDfCsv"], index=False)
        tis_df.to_pickle(self.setup.ROOT_PATH + self.config["prep"]["TisDfPkl"])

        return 0

    def join_inputs(self) -> int:
        """
        Join ohlcv datasets with datasets from prep dir
        """

        prep_dir = self.setup.ROOT_PATH + self.config["prep"]["DataPreprocessedDir"]
        df_paths = []
        for f in os.listdir(prep_dir):
            if os.path.isfile(os.path.join(prep_dir, f)) and f.endswith('.pkl'):
                df_paths.append(prep_dir+f)

        self.logger.info(f"Joining datasets: {', '.join(map(str, df_paths))}")

        dfs = [pd.read_csv(self.candle_df_raw_path)]
        for f in df_paths:
            df = pd.read_pickle(f)
            dfs.append(df)
            del df
        for df in dfs:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index(keys="Date", inplace=True)
        self.logger.info(f"Datasets dims (no header or date index): {', '.join(map(str, [df.shape for df in dfs]))}")
        df_joined = dfs[0].join(dfs[1:], how="inner")
        df_joined.reset_index(inplace=True)

        # Transform y_train to qualitative variable
        self.logger.info(f"Transforming target variable")
        self.transform_target(df_joined)
        
        self.logger.info(f"Joined dataset dim (no header or date index): {df_joined.shape}")

        # Save pickle and csv
        input_dir = self.setup.ROOT_PATH + self.config["prep"]["DataInputDir"]
        if not os.path.isdir(input_dir):
            os.mkdir(input_dir)
        df_joined.to_pickle(self.setup.ROOT_PATH + self.config["prep"]["JoinedDfPkl"])
        df_joined.to_csv(self.setup.ROOT_PATH + self.config["prep"]["JoinedDfCsv"], index=False)

        return 0

    def transform_target(self, df) -> int:
        """
        Transforms Close prices to target variable
        If the problem is specified as regression, then shifted percentage change will be the target.
        In the case of classification: <...>.
        """
        var_target = self.config["model"]["VarTarget"]
        df[var_target] = df["Close"].pct_change(1) 
        
        if self.config["model"]["Problem"] not in ["classification", "regression"]: 
            self.logger.error("Unrecognized problem type. Available problem types: Regression/Classification")
        if self.config["model"]["Problem"] == "classification":
            df[var_target] = df[var_target].apply(lambda x: 1 if x>1 else 0)
        
        # Shift features one day backwards. Target change t0->t1 is assigned to features X(t1).
        df[var_target] = df[var_target].shift(-1)
        df.drop(df.tail(1).index, inplace=True)
        
        return 0
