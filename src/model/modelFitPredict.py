from src.init_setup import *
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import History, EarlyStopping, ModelCheckpoint
import keras_tuner
from keras_tuner.tuners import RandomSearch
import json
import pickle
import time


class RollingLSTM:
    def __init__(self) -> None:
        self.setup = Setup()
        self.config = self.setup.config
        self.logger = logging.getLogger("Fit Predict")
        self.logger.addHandler(logging.StreamHandler())
        print = self.logger.info
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
            "epochs": int(self.config["model"]["Epochs"]),
            "validation_window": int(self.config["model"]["ValidationWindow"]),
            "batch_size_train": int(self.config["model"]["BatchSizeTrain"]),
            "batch_size_validation": int(self.config["model"]["BatchSizeValidation"]),
            "batch_size_test": int(self.config["model"]["BatchSizeTest"]),
            "units": int(self.config["model"]["LSTMUnits"]),
            "units_min": int(self.config["model"]["LSTMUnits"]),
            "units_max": int(self.config["model"]["LSTMUnitsMax"]),
            "dropout": float(self.config["model"]["DropoutRate"]),
            "dropout_min": float(self.config["model"]["DropoutRateMin"]),
            "dropout_max": float(self.config["model"]["DropoutRateMax"]),
            "loss": self.config["model"]["LossFunction"],
            "activation": self.config["model"]["ActivationFunction"],
            "lr": float(self.config["model"]["LearningRate"]),
            "lr_tune": [float(idx) for idx in self.config["model"]["LearningRateTune"].split(',')],
            "optimizer": self.config["model"]["Optimizer"],
            "n_hidden": int(self.config["model"]["HiddenLayers"]),
            "tune_trials": int(self.config["model"]["HyperParamTuneTrials"]),
        }
        for item in self.params.items(): self.logger.info(f"{item}")
        self.predictions = []

    def model_fit_predict(self) -> int:
        """
        Training example was implemented according to machine-learning-mastery forum
        The function takes data from the dictionary returned from splitWindows.create_windows function
        https://machinelearningmastery.com/stateful-stateless-lstm-time-series-forecasting-python/
        """

        # For each train-test windoww
        with tqdm(total=self.x_train.shape[0], desc="[Cross Window Progress Bar]") as progress_bar:

            # Save model History for error metrics
            history = History()

            # build and return model
            # self.logger.info(f'Building the model')
            # model = self.model_builder()

            # print model summary
            # model.summary(print_fn=self.logger.info)

            # Fit and predict for each window
            self.logger.info(f'Fitting the model, saving predictions')
            for i in range(self.x_train.shape[0]):
                
                # Hyperparameter tuning
                self.logger.info("Building a model to tune")
                tuner = RandomSearch(
                    self.model_builder
                    , objective="val_loss"
                    , max_trials = self.params["tune_trials"]
                    , directory = "models", overwrite = True
                    , project_name = f"model_window_{i}"
                )
                tuner.search_space_summary()
                
                x_train = self.x_train[i][:-self.params["validation_window"]]
                y_train = self.y_train[i][:-self.params["validation_window"]]
                x_val = self.x_train[i][-self.params["validation_window"]:]
                y_val = self.y_train[i][-self.params["validation_window"]:]
                
                self.logger.info("Tuning the model")
                self.logger.info(f"Train window dimensions (features, targets): {x_train.shape}, " + f"{y_train.shape}")
                self.logger.info(f"Validation window dimensions (features, targets): {x_val.shape}, " + f"{y_val.shape}")
                tuner.search(
                    x_train, y_train
                    , validation_data = (x_val, y_val)
                    , epochs = self.params["epochs"]
                    , batch_size=self.params["batch_size_validation"] # it has to be validation one since there is no other way to specify, and obviously batch size <= sample size
                    , shuffle = False
                )
                optimal_hp = tuner.get_best_hyperparameters(1)[0] # num_trials arg -> how robust should the tune process be to random seed
                self.logger.info(f"Hyperparams picked by Random Search: {optimal_hp.values}")
                
                # Build the tuned model and train it; use early stopping for epochs
                tuned_model = tuner.hypermodel.build(optimal_hp)
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2*self.params["epochs"])
                history = tuned_model.fit(
                    x_train, y_train
                    , validation_data = (x_val, y_val)
                    , epochs=self.params["epochs"]
                    , batch_size=self.params["batch_size_train"]
                    , shuffle=False
                    , verbose=1
                    , callbacks=[es]
                )

                self.predictions.append(
                    tuned_model.predict(self.x_test[i], batch_size=self.params["batch_size_test"], verbose=0)
                )

                # Reset weight matrices between each recalibration window
                tuned_model.reset_states()

                progress_bar.update(1)

        # Save predictions
        self.logger.info(f'Saving predictions')
        output_dir = self.setup.ROOT_PATH + self.config["prep"]["DataOutputDir"]
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        with open(self.config["prep"]["PredictionsArray"], 'wb') as handle:
            pickle.dump(np.asarray(self.predictions), handle, protocol=pickle.HIGHEST_PROTOCOL)

        return 0

    def model_builder(self, hp) -> Sequential:
        """
        A function building the sequential Keras model
        model_builder parameters are described in parameters.py script
        The model uses stacked LSTM layers with a dropout set in each of them
        Last LSTM layer before the output layer always has to have return_sequences=False
        """
        batch_input_shape = (self.params["batch_size_validation"], self.params["lookback"], len(self.params["features"]))

        # Define to-be-tuned-hyperparameters
        hp_units = hp.Int("units", min_value=self.params["units_min"], max_value=self.params["units_max"], step=16)
        hp_dropout = hp.Float("dropout", min_value=self.params["dropout_min"], max_value=self.params["dropout_max"], step=0.05) 
        hp_lr = hp.Choice("learning_rate", values=self.params["lr_tune"])
        
        model = tf.keras.Sequential([
            layers.LSTM(  # input layer
                hp_units
                , batch_input_shape=batch_input_shape
                , stateful=True
                , dropout=hp_dropout
                , return_sequences=True  # pass hidden layer outputs to the next LSTM layer
            ),
            layers.LSTM(  # 1st hidden layer
                hp_units
                , batch_input_shape=batch_input_shape
                , activation=self.params["activation"], dropout=hp_dropout,
                stateful=True, return_sequences=True
            ),
            layers.LSTM(  # 2nd hidden layer
                hp_units
                , batch_input_shape=batch_input_shape
                , activation=self.params["activation"], dropout=hp_dropout,
                stateful=True, return_sequences=False
            ),
            # layers.LSTM(  # 3rd hidden layer
            #     self.params["units"]
            # #     , batch_input_shape=batch_input_shape
            #     , activation=self.params["activation"], dropout=self.params["dropout"],
            #     stateful=True, return_sequences=False
            # ),
            # layers.LSTM(  # 4th hidden layer
            #     self.params["units"]
            # #     , batch_input_shape=batch_input_shape
            #     , activation=self.params["activation"], dropout=self.params["dropout"],
            #     stateful=True, return_sequences=False
            # ),
            layers.Dense(1)  # Output layer
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
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
