from src.init_setup import *
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import pickle


class WindowSplit:

    def __init__(self) -> None:
        self.setup = Setup()
        self.config = self.setup.config
        self.logger = logging.getLogger("Window Split")
        self.logger.addHandler(logging.StreamHandler())
        self.logger.info("[[Window split module]]")
        self.params = {
            "target": self.config["model"]["VarTarget"],
            "features": self.config["model"]["Features"].split(', '),
            "lookback": int(self.config["model"]["Lookback"]),
            "train_window": int(self.config["model"]["TrainWindow"]),
            "test_window": int(self.config["model"]["TestWindow"])
        }
        self.df_joined_path = self.setup.ROOT_PATH + self.config["prep"]["JoinedDfPkl"]
        try:
            self.df_joined_tmp = pd.read_pickle(self.df_joined_path)
        except FileNotFoundError as fnf: 
            print(f'Could not find pickled of joined dataset, expected: {self.config["prep"]["JoinedDfPkl"]}')
            sys.exit(1)
        self.x_raw = self.df_joined_tmp[self.params["features"]].values
        self.y_raw = self.df_joined_tmp[self.params["target"]].values
        self.logger.info(f"X, y shape before split: {[self.x_raw.shape, self.y_raw.shape]}")
        self.close_prices = self.df_joined_tmp["Close"].values
        self.dates = self.df_joined_tmp["Date"].values
        self.logger.info(f"Close prices array, Date array shapes before split: " +
                         f"{[self.close_prices.shape, self.dates.shape]}")
        self.df_joined_tmp = None

    def generate_windows(self) -> int:
        """
        Split the dataset into windows in order to perform rolling LSTM model input
        The function returns a dictionary consisting:
            - input and target arrays for both training and test samples (only input for test) for each window
            - arrays of closing prices and dates for each window
            - an array of Min-Max scalers for each window
        """

        x_train, y_train, x_test, y_test = [], [], [], []
        close_train, close_test, date_train, date_test, scalers = [], [], [], [], []

        # for each recalibration window
        for i in tqdm(
            range(
                self.params["lookback"] + self.params["train_window"]
                , self.x_raw.shape[0] - self.params["test_window"]
                , self.params["test_window"]
            ), desc="Generating windows", leave=True
        ):

            x_buf, y_buf, x_test_buf, y_test_buf = [], [], [], []
            closes_train_buf, closes_test_buf, dates_train_buf, dates_test_buf = [], [], [], []

            # normalizing data, saving the scaler (only for input data, target data doesn't require normalization)
            scaler = MinMaxScaler()
            scaler.fit(self.x_raw[: i, :])
            scalers.append(scaler)

            # training periods:
            for j in range(i - self.params["train_window"], i, 1):
                x_transformed = scaler.transform(self.x_raw[j - self.params["lookback"]: j, :])
                x_buf.append(x_transformed)
                y_buf.append([[self.y_raw[j].copy()]])
                dates_train_buf.append(self.dates[j])
                closes_train_buf.append(self.close_prices[j].copy())

            # for test periods:
            for k in range(i, i + self.params["test_window"], 1):
                x_test_buf.append(scaler.transform(self.x_raw[k - self.params["lookback"]: k, :]))
                y_test_buf.append([[self.y_raw[k].copy()]])
                dates_test_buf.append(self.dates[k])  # equals to shape of dates
                closes_test_buf.append(self.close_prices[k].copy())

            # append buffers:
            x_train.append(x_buf)
            y_train.append(y_buf)
            x_test.append(x_test_buf)
            y_test.append(y_test_buf)
            close_train.append(closes_train_buf)
            close_test.append(closes_test_buf)
            date_train.append(dates_train_buf)
            date_test.append(dates_test_buf)

        # change to numpy arrays from list of lists and save them to dictionary
        windows_dict = {
            'x_train': np.asarray(x_train),
            'y_train': np.asarray(y_train, dtype=float),
            'x_test': np.asarray(x_test),
            'y_test': np.asarray(y_test),
            'closes_train': np.asarray(close_train),
            'closes_test': np.asarray(close_test),
            'dates_train': np.asarray(date_train),
            'dates_test': np.asarray(date_test),
            'scalers': np.asarray(scalers)
        }

        self.logger.info(f"\nTrain window dimensions (features, targets): {windows_dict['x_train'].shape}, "f"{windows_dict['y_train'].shape}")
        self.logger.info(f"Test window dimensions (features, targets): {windows_dict['x_test'].shape}, "f"{windows_dict['y_test'].shape}\n")

        # Pickle the result dictionary
        with open(self.config["prep"]["WindowSplitDict"], 'wb') as handle:
            pickle.dump(windows_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return 0
