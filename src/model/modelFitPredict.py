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
        with open(self.config["prep"]["WindowSplitDict"], 'rb') as handle: self.window_dict = pickle.load(handle)
        self.logger.info(f"Loaded data dictionary!")
        self.logger.info(f'GPU DETECTED: [{tf.test.is_gpu_available(cuda_only=True)}]')
        self.logger.info(f"Train window dimensions (features, targets): {self.window_dict['x_train'].shape}, "
                         f"{self.window_dict['y_train'].shape}")
        self.logger.info(f"Test window dimensions (features, targets): {self.window_dict['x_test'].shape}, "
                         f"{self.window_dict['y_test'].shape}")
        self.x_train, self.y_train, self.x_test = (
            self.window_dict['x_train'], self.window_dict['y_train'], self.window_dict['x_test']
        )
        for key in self.config["model"]: self.logger.info(f"{key}: {self.config['model'][key]}")
        self.early_stopping = 2.5e-4
        self.predictions = []
        self.timestamp = time.strftime("%Y-%m-%d_%H-%M")

    def model_fit_predict(self, i, shared_pred_dict): # shared_pred_dict
        """
        Training example was implemented according to machine-learning-mastery forum
        The function takes data from the dictionary returned from splitWindows.create_windows function
        https://machinelearningmastery.com/stateful-stateless-lstm-time-series-forecasting-python/
        """
        # start_time = time.time()
        log_dir = self.tensorboard_logger # + time.strftime("%Y-%m-%d_%H-%M-%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=10, write_graph=True)

        # Hyperparameter tuning
        # self.logger.info("Building a model to tune")
        tuner = RandomSearch(
            self.model_builder
            , objective="val_loss"
            , max_trials = int(self.config["model"]["HyperParamTuneTrials"])
            , directory = "models", overwrite = True
            , project_name = f"model_window_{i}"
        )
        if i == 0: tuner.search_space_summary()
        
        # self.logger.info("[{i}/{self.x_train.shape[0]}] Tuning the model")
        validation_window_size = int(self.config["model"]["ValidationWindow"])
        if i==0:
            self.logger.info(f"Train window dimensions (features, targets): {self.x_train.shape}, " + f"{self.y_train.shape}")
            validation_set_shapes = (self.x_train[i][-validation_window_size:].shape, self.y_train[i][-validation_window_size:].shape)
            self.logger.info(f"Validation window dimensions (features, targets): {validation_set_shapes[0]}, " + f"{validation_set_shapes[1]}")
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3, min_delta=self.early_stopping)
        tuner.search(
            self.x_train[i][:-validation_window_size]
            , self.y_train[i][:-validation_window_size]
            , validation_data = (
                self.x_train[i][-validation_window_size:]
                , self.y_train[i][-validation_window_size:]
            )
            , epochs = int(self.config["model"]["Epochs"])
            , batch_size = int(self.config["model"]["BatchSizeValidation"]) # it has to be validation one since there is no other way to specify, and obviously batch size <= sample size
            , shuffle = False
            , callbacks = [es, tensorboard_callback]
            , verbose = 1
        )
        optimal_hp = tuner.get_best_hyperparameters(1)[0] # num_trials arg -> how robust should the tune process be to random seed
        report_dir = self.setup.ROOT_PATH + self.config["prep"]["ReportDir"]
        with open(f'{report_dir}model_hyperparams_{self.timestamp}.json', "a") as fp: 
            json.dump(optimal_hp.values, fp, indent=4, sort_keys=False)
            fp.write(os.linesep)
        # self.logger.info(f"[{i}/{self.x_train.shape[0]}] Hyperparams picked by Random Search: {optimal_hp.values}. Fitting the tuned model")
        
        # Build the tuned model and train it; use early stopping for epochs
        tuned_model = tuner.hypermodel.build(optimal_hp)
        tuned_model.fit(
            self.x_train[i][:-validation_window_size]
            , self.y_train[i][:-validation_window_size]
            , validation_data = (
                self.x_train[i][-validation_window_size:]
                , self.y_train[i][-validation_window_size:]
            )
            , epochs = int(self.config["model"]["Epochs"])
            , batch_size = int(self.config["model"]["BatchSizeTrain"])
            , shuffle = False
            , verbose = 1
            , callbacks = [es, tensorboard_callback]
        )

        # generate array of predictions from ith window and save it to dictionary (shared between processes)
        shared_pred_dict[i] = tuned_model.predict(self.x_test[i], batch_size=int(self.config["model"]["BatchSizeTest"]), verbose=0)

        # self.logger.info(f'[{i}/{self.x_train.shape[0]}] FINISHED, EXEC TIME: {time.time() - start_time}')

        return 0

    def model_fit_predict_multiprocess(self):  
        mp.set_start_method('spawn', force=True)

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
                self.logger.info(f'[{i}/{self.x_train.shape[0]}] FINISHED, TOTAL FINISHED: [{len(shared_pred_dict)}/{self.x_train.shape[0]}] [{i}]-th EXEC TIME: {time.time() - start_time}')
                self.get_gpu_mem_usage(i)
            self.predictions = [shared_pred_dict[key] for key in sorted(shared_pred_dict.keys())]

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
        batch_input_shape = (int(self.config["model"]["BatchSizeValidation"])
                             , int(self.config["model"]["Lookback"])
                             , len(self.config["model"]["Features"].split(', ')))
        problem = self.config["model"]["Problem"]
        
        # Define hyperparameters grid
        hp_lr               = hp.Choice("learning_rate", values=[float(x) for x in self.config["model"]["LearningRate"].split(', ')])
        loss_fun_classification = {
            "Hinge" : tf.keras.losses.Hinge()
        }
        loss_fun_regression = {
            "MAPE"  : tf.keras.losses.MeanAbsolutePercentageError()
            , "MSE" : tf.keras.losses.MeanSquaredError()
        }
        available_optimizers = {
            "Adam"        : tf.keras.optimizers.Adam(learning_rate=hp_lr)
            , "RMSprop"   : tf.keras.optimizers.RMSprop(learning_rate=hp_lr)
            , "Adadelta"  : tf.keras.optimizers.Adadelta(learning_rate=hp_lr)
        }
        
        if problem == "regression"      :   hp_loss_function = loss_fun_regression[hp.Choice("loss_fun", values=self.config["model"]["LossFunctionRegression"].split(', '))]
        elif problem == "classification":   hp_loss_function = loss_fun_classification[hp.Choice("loss_fun", values=self.config["model"]["LossFunctionClassification"].split(', '))]
        hp_optimizer        = available_optimizers[hp.Choice("optimizer", self.config["model"]["Optimizer"].split(', '))]
        hp_units            = hp.Int("units", min_value=int(self.config["model"]["LSTMUnits"]), max_value=int(self.config["model"]["LSTMUnitsMax"]), step=16)
        hp_hidden_layers    = hp.Int("hidden_layers", min_value=int(self.config["model"]["HiddenLayersMin"]), max_value=int(self.config["model"]["HiddenLayersMax"]), step=16)
        hp_dropout          = hp.Float("dropout", min_value=float(self.config["model"]["DropoutRateMin"]), max_value=float(self.config["model"]["DropoutRateMax"]), step=0.05) 
        
        # Sequential model definition:
        layer_list = []
        for _ in range(hp_hidden_layers-1): 
            layer_list.append(layers.LSTM(
                hp_units
                , batch_input_shape=batch_input_shape
                , activation=self.config["model"]["ActivationFunction"]
                , stateful=True
                , dropout=hp_dropout
                , return_sequences=True
            ))
        layer_list.extend([
            layers.LSTM(hp_units, batch_input_shape=batch_input_shape
                , activation=self.config["model"]["ActivationFunction"], dropout=hp_dropout
                , stateful=True, return_sequences=False)
            , layers.Dense(1)
        ])
        model = tf.keras.Sequential(layer_list)
        model.compile(
            optimizer = hp_optimizer
            , loss = hp_loss_function  
        )

        return model

    def save_results(self) -> int:
        """
        Save results and parameters to results directory
        """
        self.logger.info(f"Saving evaluation data and model description with timestamp: {self.timestamp}")
        
        # Load predictions array
        with open(self.config["prep"]["PredictionsArray"], 'rb') as handle: preds = pickle.load(handle)
        df_pred_eval = pd.DataFrame(
            zip(self.window_dict['dates_test'].reshape(-1), preds.reshape(-1), self.window_dict['closes_test'].reshape(-1)),
            columns=['Date', 'Pred', 'Real']
        )

        # Save results to csv and pkl
        df_pred_eval.to_csv(f'{self.config["prep"]["DataOutputDir"]}model_eval_data_{self.timestamp}.csv', index=False)
        df_pred_eval.set_index("Date", inplace=True)
        df_pred_eval.to_pickle(f'{self.config["prep"]["DataOutputDir"]}model_eval_data_{self.timestamp}.pkl')
        
        # Save model description to .json
        model_desc = {
            'Model Name': f'{time.strftime("%Y-%m-%d_%H-%M")}',
            'Features': f'{self.config["model"]["Features"].split(", ")}',
            # 'Probability threshold': f'{self.config["model"]["TargetThreshold"]}',
            'Look-back period': f'{int(self.config["model"]["Lookback"])}',
            'Training period': f'{self.config["model"]["TrainWindow"]}',
            'Test period': f'{self.config["model"]["TestWindow"]}',
            'Train Batch size': f'{int(self.config["model"]["BatchSizeTrain"])}',
            'Test Batch size': f'{int(self.config["model"]["BatchSizeTest"])}',
            'Number of epochs': f'{int(self.config["model"]["Epochs"])}'
            # 'LSTM layer units': f'{int(self.config["model"]["LSTMUnits"])}',
            # 'Dropout rate': f'{float(self.config["model"]["DropoutRate"])}',
            # 'Activation function': f'{self.config["model"]["ActivationFunction"]}',
            # 'Learning rate': f'{float(self.config["model"]["LearningRate"])}',
            # 'loss function': f'{self.config["model"]["LossFunction"]}',
            # 'Optimizer': f'{self.config["model"]["Optimizer"]}',
            # 'Number of hidden layers': f'{int(self.config["model"]["HiddenLayers"])}'
        }
        report_dir = self.setup.ROOT_PATH + self.config["prep"]["ReportDir"]
        if not os.path.isdir(report_dir): os.mkdir(report_dir)
        with open(f'{report_dir}model_config_{self.timestamp}.json', "w") as fp: json.dump(model_desc, fp, indent=4, sort_keys=False)

        return 0

    def get_gpu_mem_usage(self, i) -> None:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        self.logger.info(f"POST [{i}] GPU memory usage: {np.round(info.used/info.total*100, 2)}%")
