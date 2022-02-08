from parameters import *
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_windows(input_matrix, closes, dates, targets):
    """
    Split the dataset into windows in order to perform rolling LSTM approach
    The function returns a dictionary consisting:
        - input and target arrays for both training and test samples (only input for test) for each window
        - arrays of closing prices and dates for each window
        - an array of Min-Max scalers for each window
    :param input_matrix: matrix of input variables - refer to features list in parameters.py
    :param closes: an array of closing prices
    :param dates: an array of dates
    :param targets: target variable array
    """

    X, y, X_test, y_test = [], [], [], []
    closes_train, closes_test, dates_train, dates_test, scalers = [], [], [], [], []

    # for each recalibration window
    for i in tqdm(
            range(lookback+train_period, input_matrix.shape[0]-test_period, test_period),
            desc="Generating windows", leave=True
    ):

        X_buf, y_buf, X_test_buf, y_test_buf = [], [], [], []
        closes_train_buf, closes_test_buf, dates_train_buf, dates_test_buf = [], [], [], []

        # normalizing data, and saving the scaler (only for input data, as target data doesn't require normalization)
        scaler = MinMaxScaler()
        scaler.fit(input_matrix[i-train_period-lookback:i, :])
        scalers.append(scaler)

        # for training periods:
        for j in range(i-train_period, i, 1):
            X_transformed = scaler.transform(input_matrix[j-lookback:j, :].copy())
            X_buf.append(X_transformed)
            y_buf.append([targets[j].copy()])
            dates_train_buf.append(dates[j])
            closes_train_buf.append(closes[j].copy())

        # for test periods:
        for k in range(i, i+test_period, 1):
            X_test_buf.append(scaler.transform(input_matrix[k-lookback:k, :].copy()))
            y_test_buf.append([targets[k].copy()])
            dates_test_buf.append(dates[k])  # equals to shape of dates
            closes_test_buf.append(closes[k].copy())

        # append buffers:
        X.append(X_buf)
        y.append(y_buf)
        X_test.append(X_test_buf)
        y_test.append(y_test_buf)
        closes_train.append(closes_train_buf)
        closes_test.append(closes_test_buf)
        dates_train.append(dates_train_buf)
        dates_test.append(dates_test_buf)

    # change to numpy arrays from list of lists and save them to dictionary
    windows_dict = {
        'X': np.asarray(X),
        'y': np.asarray(y),
        'X_test': np.asarray(X_test),
        'y_test': np.asarray(y_test),
        'closes_train': np.asarray(closes_train),
        'closes_test': np.asarray(closes_test),
        'dates_train': np.asarray(dates_train),
        'dates_test': np.asarray(dates_test),
        'scalers': np.asarray(scalers)
    }
    return windows_dict
