from src.init_setup import *
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import History
import json
import pickle
import time


class RollingLSTM:
    def __init__(self) -> None:
        self.setup = Setup()
        self.config = self.setup.config
        self.logger = logging.getLogger("Fit Predict")
        self.logger.addHandler(logging.StreamHandler())
        self.icsa_df_raw_path = self.setup.ROOT_PATH + self.config["raw"]["IcsaRawDF"]
        with open(self.config["prep"]["WindowSplitDict"], 'rb') as handle:
            self.window_dict = pickle.load(handle)
        self.logger.info(f"Loaded data dictionary!")
        self.logger.info(f"Train window dimensions (features, targets): {self.window_dict['x_train'].shape}, "
                         f"{self.window_dict['y_train'].shape}")
        self.logger.info(f"Test window dimensions (features, targets): {self.window_dict['x_test'].shape}, "
                         f"{self.window_dict['y_test'].shape}")
        self.x_train, self.y_train, self.x_test = (
            self.window_dict['x_train'], self.window_dict['y_train'], self.window_dict['x_test']
        )
        self.params = {
            "lookback": int(self.config["model"]["Lookback"]),
            "features": self.config["model"]["Features"].split(','),
            "epochs": int(self.config["model"]["NumberOfEpochs"]),
            "batch_size_train": int(self.config["model"]["BatchSizeTrain"]),
            "batch_size_test": int(self.config["model"]["BatchSizeTest"]),
            "units": int(self.config["model"]["LSTMUnits"]),
            "dropout": float(self.config["model"]["DropoutRate"]),
            "loss": self.config["model"]["LossFunction"],
            "activation": self.config["model"]["ActivationFunction"],
            "lr": float(self.config["model"]["LearningRate"]),
            "optimizer": self.config["model"]["Optimizer"],
            "n_hidden": int(self.config["model"]["NumberOfHiddenLayers"]),
        }
        self.predictions = []

    def model_fit_predict(self) -> int:
        """
        Training example was implemented according to machine-learning-mastery forum
        The function takes data from the dictionary returned from splitWindows.create_windows function
        https://machinelearningmastery.com/stateful-stateless-lstm-time-series-forecasting-python/
        """

        with tqdm(total=self.x_train.shape[0], desc="Fitting the model, saving predictions") as progress_bar:

            # Save model History for error metrics
            history = History()

            # build model
            self.logger.info(f'Building the model')
            model = self.model_builder()

            # print model summary
            model.summary(print_fn=self.logger.info)

            # Fit and predict for each window
            for i in range(self.x_train.shape[0]):

                # TRAIN (FIT) model for each epoch
                # history = current_model.fit(
                #     x_train[i], y_train[i],
                #     epochs=_epochs, batch_size=batch,
                #     verbose=0, shuffle=False, validation_split=0.1,
                #     callbacks=[history]
                # )
                # print(x_train[i].shape, x_train[i].dtype, y_train[i].shape, y_train[i].dtype)

                for e in range(self.params["epochs"]):
                    model.fit(
                        self.x_train[i], self.y_train[i],
                        epochs=1, batch_size=self.params["batch_size_train"],
                        verbose=0, shuffle=False,
                        callbacks=[history]
                    )

                self.predictions.append(
                    model.predict(self.x_test[i], batch_size=self.params["batch_size_test"], verbose=0)
                )

                model.reset_states()

                progress_bar.update(1)

        # Save predictions
        self.logger.info(f'Saving predictions')
        output_dir = self.setup.ROOT_PATH + self.config["prep"]["DataOutputDir"]
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        with open(self.config["prep"]["PredictionsArray"], 'wb') as handle:
            pickle.dump(np.asarray(self.predictions), handle, protocol=pickle.HIGHEST_PROTOCOL)

        return 0

    def model_builder(self) -> Sequential:
        """
        A function building the sequential Keras model
        model_builder parameters are described in parameters.py script
        The model uses stacked LSTM layers with a dropout set in each of them
        Last LSTM layer before the output layer always has to have return_sequences=False
        """
        batch_input_shape = (self.params["batch_size_train"], self.params["lookback"], len(self.params["features"]))

        model = tf.keras.Sequential([
            layers.LSTM(  # input layer
                self.params["units"],
                batch_input_shape=batch_input_shape,
                stateful=True,
                dropout=self.params["dropout"],
                return_sequences=True  # pass hidden layer outputs to the next LSTM layer
            ),
            layers.LSTM(  # 1st hidden layer
                self.params["units"], batch_input_shape=batch_input_shape,
                activation=self.params["activation"], dropout=self.params["dropout"],
                stateful=True, return_sequences=True
            ),
            layers.LSTM(  # 2nd hidden layer
                self.params["units"], batch_input_shape=batch_input_shape,
                activation=self.params["activation"], dropout=self.params["dropout"],
                stateful=True, return_sequences=False
            ),
            # layers.LSTM(  # 3rd hidden layer
            #     self.params["units"], batch_input_shape=batch_input_shape,
            #     activation=self.params["activation"], dropout=self.params["dropout"],
            #     stateful=True, return_sequences=False
            # ),
            # layers.LSTM(  # 4th hidden layer
            #     self.params["units"], batch_input_shape=batch_input_shape,
            #     activation=self.params["activation"], dropout=self.params["dropout"],
            #     stateful=True, return_sequences=False
            # ),
            layers.Dense(1)  # Output layer
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.params["lr"]),
            loss=self.params["loss"]
        )

        return model

    def save_results(self) -> int:
        """
        Save results and parameters to results directory
        """
        timestamp = time.strftime("%Y-%m-%d_%H-%M")
        self.logger.info(f"Saving evaluation data and model description with timestamp: {timestamp}")

        with open(self.config["prep"]["PredictionsArray"], 'rb') as handle:
            preds = pickle.load(handle)

        df_pred_eval = pd.DataFrame(
            zip(
                self.window_dict['dates_test'].reshape(-1),
                preds.reshape(-1),
                self.window_dict['closes_test'].reshape(-1)
            ),
            columns=['Date', 'Pred', 'Real']
        )

        # Save results to csv and pkl
        df_pred_eval.to_csv(
            f'{self.config["prep"]["DataOutputDir"]}model_eval_data_{timestamp}.csv'
            , index=False
        )
        df_pred_eval.set_index("Date", inplace=True)
        df_pred_eval.to_pickle(f'{self.config["prep"]["DataOutputDir"]}model_eval_data_{timestamp}.pkl')

        model_desc = {
            'Model Name': f'{time.strftime("%Y-%m-%d_%H-%M")}',
            'Features': f'{self.params["features"]}',
            'Probability threshold': f'{self.config["model"]["TargetThreshold"]}',
            'Look-back period': f'{self.params["lookback"]}',
            'Training period': f'{self.config["model"]["TrainWindow"]}',
            'Test period': f'{self.config["model"]["TestWindow"]}',
            'LSTM layer units': f'{self.params["units"]}',
            'Dropout rate': f'{self.params["dropout"]}',
            'Activation function': f'{self.params["activation"]}',
            'Initial learning rate': f'{self.params["lr"]}',
            'loss function': f'{self.params["loss"]}',
            'Number of epochs': f'{self.params["epochs"]}',
            'Train Batch size': f'{self.params["batch_size_train"]}',
            'Test Batch size': f'{self.params["batch_size_test"]}',
            'Optimizer': f'{self.params["optimizer"]}',
            'Number of hidden layers': f'{self.params["n_hidden"]}'
        }

        # Save model description to .json
        report_dir = self.setup.ROOT_PATH + self.config["prep"]["ReportDir"]
        if not os.path.isdir(report_dir):
            os.mkdir(report_dir)
        with open(
                f'{report_dir}model_config_{timestamp}.json', "w"
        ) as fp:
            json.dump(model_desc, fp, indent=4, sort_keys=False)

        return 0
