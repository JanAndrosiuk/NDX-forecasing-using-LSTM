from src.init_setup import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import multiprocessing as mp
from multiprocessing import Process, Manager, Pool
from tqdm import tqdm # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import pynvml # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.callbacks import History, EarlyStopping, ModelCheckpoint # type: ignore
import tensorboard
import keras_tuner # type: ignore
from keras_tuner.tuners import RandomSearch # type: ignore
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
        self.tensorboard_logger = self.config["logger"]["TensorboardLoggerPath"]
        self.icsa_df_raw_path = self.setup.ROOT_PATH + self.config["raw"]["IcsaRawDF"]
        with open(self.config["prep"]["WindowSplitDict"], 'rb') as handle:
            self.window_dict = pickle.load(handle)
        self.logger.info(f"Loaded data dictionary!")
        self.logger.info(f'GPU DETECTED: [{tf.test.is_gpu_available(cuda_only=True)}]')
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

    def model_fit_predict(self, i, shared_pred_dict): # shared_pred_dict
        """
        Training example was implemented according to machine-learning-mastery forum
        The function takes data from the dictionary returned from splitWindows.create_windows function
        https://machinelearningmastery.com/stateful-stateless-lstm-time-series-forecasting-python/
        """
        
        windows_count = self.x_train.shape[0]
        start_time = time.time()
        
        log_dir = self.tensorboard_logger # + time.strftime("%Y-%m-%d_%H-%M-%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

        # Hyperparameter tuning
        # self.logger.info("Building a model to tune")
        tuner = RandomSearch(
            self.model_builder
            , objective="val_loss"
            , max_trials = self.params["tune_trials"]
            , directory = "models", overwrite = True
            , project_name = f"model_window_{i}"
        )
        if i == 0: tuner.search_space_summary()
        
        # self.logger.info("[{i}/{windows_count}] Tuning the model")
        if i==0:
            self.logger.info(f"Train window dimensions (features, targets): {self.x_train.shape}, " + f"{self.y_train.shape}")
            validation_set_shapes = (self.x_train[i][-self.params["validation_window"]:].shape, self.y_train[i][-self.params["validation_window"]:].shape)
            self.logger.info(f"Validation window dimensions (features, targets): {validation_set_shapes[0]}, " + f"{validation_set_shapes[1]}")
        tuner.search(
            self.x_train[i][:-self.params["validation_window"]]
            , self.y_train[i][:-self.params["validation_window"]]
            , validation_data = (
                self.x_train[i][-self.params["validation_window"]:]
                , self.y_train[i][-self.params["validation_window"]:]
            )
            , epochs = self.params["epochs"]
            , batch_size=self.params["batch_size_validation"] # it has to be validation one since there is no other way to specify, and obviously batch size <= sample size
            , shuffle = False
            , callbacks=[tensorboard_callback]
            , verbose = 1
        )
        optimal_hp = tuner.get_best_hyperparameters(1)[0] # num_trials arg -> how robust should the tune process be to random seed
        # self.logger.info(f"[{i}/{windows_count}] Hyperparams picked by Random Search: {optimal_hp.values}. Fitting the tuned model")
        
        # Build the tuned model and train it; use early stopping for epochs
        tuned_model = tuner.hypermodel.build(optimal_hp)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.params["epochs"])
        history = tuned_model.fit(
            self.x_train[i][:-self.params["validation_window"]]
            , self.y_train[i][:-self.params["validation_window"]]
            , validation_data = (
                self.x_train[i][-self.params["validation_window"]:]
                , self.y_train[i][-self.params["validation_window"]:]
            )
            , epochs=self.params["epochs"]
            , batch_size=self.params["batch_size_train"]
            , shuffle=False
            , verbose=1
            , callbacks=[es, tensorboard_callback]
        )

        # generate array of predictions from ith window and save it to dictionary (shared between processes)
        shared_pred_dict[i] = tuned_model.predict(self.x_test[i], batch_size=self.params["batch_size_test"], verbose=0)
        # current_predictions = tuned_model.predict(self.x_test[i], batch_size=self.params["batch_size_test"], verbose=0)

        # log status
        # self.logger.info(f'[{i}/{windows_count}] FINISHED, EXEC TIME: {time.time() - start_time}')

        # return current_predictions
        return 0

    def model_fit_predict_multiprocess(self):
        
        mp.set_start_method('spawn', force=True)
        windows_count = self.x_train.shape[0]

        # Create separate process for each window. Save predictions to a dictionary shared between processes.
        with Manager() as manager:
            shared_pred_dict = manager.dict()
            processes = []
            for i in range(self.x_train.shape[0]):
                p = Process(target=self.model_fit_predict, args=(i, shared_pred_dict))  # Passing the list
                processes.append(p)
                start_time = time.time()
                p.start()
                p.join()
                self.logger.info(f'[{i}/{windows_count}] FINISHED, TOTAL FINISHED: [{len(shared_pred_dict)}/{windows_count}] [{i}]-th EXEC TIME: {time.time() - start_time}')
                self.get_gpu_mem_usage(i)
            self.predictions = [shared_pred_dict[key] for key in sorted(shared_pred_dict.keys())]

        # with Pool(processes=1) as pool: # safe option; 1 by 1
        # # self.predictions = pool.map(self.model_fit_predict, [i for i in range(self.x_train.shape[0])])
        # # self.predictions = [ent for sublist in self.predictions for ent in sublist]
        #     for i in range(self.x_train.shape[0]):
        #         curr_pred = pool.apply_async(self.model_fit_predict, [i])
        #         self.predictions.append(curr_pred.get())
        #         curr_pred.wait()
        #         # [result.wait() for result in results]
        #         # pool.close()
        #         # pool.join()
        # self.predictions = [x for sublist in self.predictions for x in sublist]

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

    def get_gpu_mem_usage(self, i) -> None:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        self.logger.info(f"POST [{i}] GPU memory usage: {np.round(info.used/info.total*100, 2)}%")