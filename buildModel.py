from parameters import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential


def model_builder(X, timesteps=lookback, n_feat=len(features)):
    """
    A function building the sequential Keras model
    model_builder parameters are described in parameters.py script
    The model uses stacked LSTM layers with a dropout set in each of them
    Last LSTM layer before the output layer always has to have return_sequences=False
    """
    model = tf.keras.Sequential([
        layers.LSTM(  # input layer
            units,
            batch_input_shape=(batch, timesteps, n_feat),
            stateful=True,
            dropout=dropout,
            return_sequences=True  # pass hidden layer outputs to the next LSTM layer
        ),
        layers.LSTM(  # 1st hidden layer
            units, batch_input_shape=(batch, timesteps, n_feat),
            activation=act, dropout=dropout, stateful=True, return_sequences=True
        ),
        layers.LSTM(  # 2nd hidden layer
            units, batch_input_shape=(batch, timesteps, n_feat),
            activation=act, dropout=dropout, stateful=True, return_sequences=False
        ),
        # layers.LSTM(  # 3rd hidden layer
        #     units, batch_input_shape=(batch, timesteps, n_feat),
        #     activation=act, dropout=dropout, stateful=True, return_sequences=False
        # ),
        # layers.LSTM(  # 4th hidden layer
        #     units, batch_input_shape=(batch, timesteps, n_feat),
        #     activation=act, dropout=dropout, stateful=True, return_sequences=False
        # ),
        layers.Dense(1)  # Output layer
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss
    )

    return model
