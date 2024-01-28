from src.init_setup import * # type: ignore
import os
import psutil # type: ignore
import shutil
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
from keras.utils import to_categorical
import tensorboard # type: ignore
import keras_tuner # type: ignore
from keras_tuner.tuners import RandomSearch # type: ignore
from keras.callbacks import History
import json
import csv
import pickle
import time
import re
import glob
import datetime


class RollingLSTM:
    def __init__(self) -> None:
        self.setup = Setup() # type: ignore
        self.config = self.setup.config
        self.timestamp = time.strftime("%Y-%m-%d_%H-%M")
        self.export_path = f'{self.setup.ROOT_PATH}{self.config["prep"]["ExportDir"]}{self.timestamp}/'
        if not os.path.isdir(self.export_path): os.mkdir(self.export_path)
        self.logger = logging.getLogger("Fit Predict") # type: ignore
        self.logger.addHandler(logging.StreamHandler()) # type: ignore
        self.logger.info("[[Model Fit & Predict module]]\n")
        print = self.logger.info
        self.tensorboard_logger = self.config["logger"]["TensorboardLoggerPath"]
        
        self.icsa_df_raw_path = self.setup.ROOT_PATH + self.config["raw"]["IcsaRawDF"]
        with open(self.config["prep"]["WindowSplitDict"], 'rb') as handle: self.window_dict = pickle.load(handle)
        self.logger.info(f'\nSuccessfully loaded split dictionary\n'\
                         f'GPU DETECTED: [{tf.test.is_gpu_available(cuda_only=True)}]\n'\
                         f'TRAIN (features; targets): ({self.window_dict["x_train"].shape}; {self.window_dict["y_train"].shape})\n'\
                         f'TEST: ({self.window_dict["x_test"].shape}; {self.window_dict["y_test"].shape})\n')
        self.x_train, self.y_train, self.x_test = (self.window_dict['x_train'], self.window_dict['y_train'], self.window_dict['x_test'])
        
        self.logger.info(f"Parameters summary:\n{'-'*49}")
        for key in self.config["model"]: self.logger.info(f"{key}: {self.config['model'][key]}")
        self.logger.info(f"{'-'*49}\n")
            
        self.early_stopping_min_delta = 0.0
        self.predictions = []

    def model_fit_predict(self, i, shared_pred_dict) -> int:
        '''
        Training example was implemented according to machine-learning-mastery forum
        The function takes data from the dictionary returned from splitWindows.create_windows function
        https://machinelearningmastery.com/stateful-stateless-lstm-time-series-forecasting-python/
        '''
        start_time = time.time()
        log_dir = self.tensorboard_logger # + time.strftime("%Y-%m-%d_%H-%M-%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=10, write_graph=True)
        history = History()
        
        validation_window_size = int(self.config["model"]["ValidationWindow"])
        if self.config["model"]["Problem"] == "classification":
            y_train = to_categorical(self.y_train[i][:-validation_window_size], num_classes=3)[:,0,:]
            y_val = to_categorical(self.y_train[i][-validation_window_size:], num_classes=3)[:,0,:]
        elif self.config["model"]["Problem"] == "regression":
            y_train = self.y_train[i][:-validation_window_size]
            y_val = self.y_train[i][-validation_window_size:]
        else:
            logging.error("Wrong problem type! Check config")
            raise

        if i == 0:
            validation_set_shapes = (self.x_train[i][-validation_window_size:].shape, self.y_train[i][-validation_window_size:].shape)
            print(f"Validation set size: ({validation_set_shapes[0]}, {validation_set_shapes[1]})\n\nSTARTING TRAINING PROCESS\n")
        
        # Validation & Early Stopping setup
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=3, restore_best_weights=True, min_delta=self.early_stopping_min_delta)
        
        # Hyperparameter tuning -> compare previous best model with current hyper-model
        tuner_previous_best = RandomSearch(
            self.model_builder
            , hyperparameters = keras_tuner.HyperParameters(), tune_new_entries = False
            , objective = "val_loss", max_trials = 1
            , directory = self.config["prep"]["TunerHistoryDir"], overwrite = True, project_name = f"current_window_model_with_previous_best_params"
        )
        print(f"[{i}/{self.x_train.shape[0]}] Evaluating previous model -> previous best parameters")
        tuner_previous_best.search(
            self.x_train[i][:-validation_window_size]
            , y_train
            , validation_data = (self.x_train[i][-validation_window_size:], y_val)
            , epochs = int(self.config["model"]["Epochs"])
            , batch_size = int(self.config["model"]["BatchSizeValidation"]) # it has to be validation one since there is no other way to specify, and obviously batch size <= sample size
            , shuffle = False, callbacks = [es, tensorboard_callback], verbose = 1
        )
        tuner_current = RandomSearch(
            self.model_builder
            , objective="val_loss"
            , max_trials = int(self.config["model"]["HyperParamTuneTrials"])
            , directory = self.config["prep"]["TunerHistoryDir"], overwrite = True, project_name = f"current_window_model"
        )
        print(f"[{i}/{self.x_train.shape[0]}] Tuning the model")
        tuner_current.search(
            self.x_train[i][:-validation_window_size]
            , y_train
            , validation_data = (self.x_train[i][-validation_window_size:], y_val)
            , epochs = int(self.config["model"]["Epochs"])
            , batch_size = int(self.config["model"]["BatchSizeValidation"]) # it has to be validation one since there is no other way to specify, and obviously batch size <= sample size
            , shuffle = False, callbacks = [es, tensorboard_callback], verbose = 1
        )
        
        # Compare previous best model and current tuned model, retrieve best hyperparameters
        current_best_score = tuner_current.oracle.get_best_trials(1)[0].get_state()["score"]
        previous_best_score = tuner_previous_best.oracle.get_best_trials(1)[0].get_state()["score"]
        if previous_best_score < current_best_score:
            hp_optimal = tuner_previous_best.get_best_hyperparameters(1)[0]
            print(f"Previous model outfperforms current tuned model: Current score: {current_best_score}, Previous score: {previous_best_score} => ")
        else: hp_optimal = tuner_current.get_best_hyperparameters(1)[0]
        
        # Save best combination
        report_dir = self.setup.ROOT_PATH + self.config["prep"]["ModelParamDir"]
        with open(f'{report_dir}optimal_hyperparams_{self.timestamp}.csv', "a") as fp: 
            writer = csv.writer(fp, delimiter="\t",lineterminator="\n")
            if i == 0:
                writer.writerow(hp_optimal.values.keys())
            writer.writerow(hp_optimal.values.values())
        shutil.copy2(f'{report_dir}optimal_hyperparams_{self.timestamp}.csv', self.export_path)
        
        # Use optimal hyperparams to train the model on the whole window period
        print(f"[{i}/{self.x_train.shape[0]}] Fitting the tuned model")
        tuned_model = tuner_current.hypermodel.build(hp_optimal)
        tuned_model.fit(
            self.x_train[i][:-validation_window_size], y_train
            , validation_data = (self.x_train[i][-validation_window_size:], y_val)
            , epochs = int(self.config["model"]["Epochs"]), batch_size = int(self.config["model"]["BatchSizeTrain"])
            , shuffle = False, verbose = 1, callbacks = [es, tensorboard_callback, history]
        )
        
        fir_history_json_path = f'{self.setup.ROOT_PATH}{self.config["prep"]["FitHistoryDir"]}fit_history_{self.timestamp}.csv'
        temp_history_df = pd.DataFrame(history.history)
        temp_history_df["window_index"] = i
        temp_history_df.to_csv(fir_history_json_path, mode='a', index=False, header=False)
        
        # generate array of predictions from ith window and save it to dictionary (shared between processes)
        predictions_array = tuned_model.predict(self.x_test[i], batch_size=int(self.config["model"]["BatchSizeTest"]), verbose=0)
        if self.config["model"]["Problem"] == "regression": shared_pred_dict[i] = predictions_array
        elif self.config["model"]["Problem"] == "classification": 
            classes = [0, 1, -1]
            shared_pred_dict[i] = [classes[most_probable_class] for most_probable_class in predictions_array.argmax(axis=-1)]
            print(predictions_array)
            print(predictions_array.argmax(axis=-1))
            print(shared_pred_dict[i])
        
        return 0

    def model_fit_predict_multiprocess(self, save=True) -> int:
        '''
        Executes model_fit_predict as separate processes. Processes share predictions dictionary.
        '''  
        mp.set_start_method('spawn', force=True)
        
        # Initialize fit history csv
        history_csv_path = f'{self.setup.ROOT_PATH}{self.config["prep"]["FitHistoryDir"]}fit_history_{self.timestamp}.csv'
        pd.DataFrame(columns=['loss', 'val_loss', 'window_index'])\
            .to_csv(history_csv_path, index=False)
        
        # Create separate process for each window. Save predictions to a dictionary shared between processes.
        with Manager() as manager:
            shared_pred_dict = manager.dict()
            for i in range(self.x_train.shape[0]):
                p = Process(target=self.model_fit_predict, args=(i, shared_pred_dict))
                start_time = time.time()
                p.start()
                p.join()
                self.logger.info(f'[{i}/{self.x_train.shape[0]}] FINISHED, TOTAL FINISHED: [{len(shared_pred_dict)}/{self.x_train.shape[0]}] [{i}]-th EXEC TIME: {time.time() - start_time}')
                self.get_gpu_mem_usage(i)
            self.predictions = [shared_pred_dict[key] for key in sorted(shared_pred_dict.keys())]

        if save == True:
            self.logger.info(f'Saving predictions')
            output_dir = self.setup.ROOT_PATH + self.config["prep"]["DataOutputDir"]
            if not os.path.isdir(output_dir): os.mkdir(output_dir)
            with open(self.config["prep"]["PredictionsArray"], 'wb') as handle: 
                pickle.dump(np.asarray(self.predictions, dtype=object), handle, protocol=pickle.HIGHEST_PROTOCOL)
            shutil.copy2(history_csv_path, self.export_path)
            shutil.copy2(f'{self.setup.ROOT_PATH}config.ini', self.export_path)

        return 0
    
    def model_builder(self, hp) -> Sequential:
        '''
        A function building the sequential Keras model
        model_builder parameters are described in parameters.py script
        The model uses stacked LSTM layers with a dropout set in each of them
        Last LSTM layer before the output layer always has to have return_sequences=False
        '''
        batch_input_shape = (int(self.config["model"]["BatchSizeValidation"])
                             , int(self.config["model"]["Lookback"])
                             , len(self.config["model"]["Features"].split(', ')))
        problem = self.config["model"]["Problem"]
        
        # If params from previous model not found, use default dictionary
        previous_hp_csv = f'{self.config["prep"]["ModelParamDir"]}optimal_hyperparams_{self.timestamp}.csv'
        if os.path.isfile(previous_hp_csv):
            with open(previous_hp_csv, newline="\n") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                hp_default_dict = list(reader)[-1]
        else: hp_default_dict = {
            "learning_rate": self.config["model"]["DefaultLearningRate"],
            "loss_fun": self.config["model"]["DefaultLossFunction"],
            "optimizer": self.config["model"]["DefaultOptimizer"],
            "units": self.config["model"]["DefaultUnits"],
            "hidden_layers": self.config["model"]["DefaultHiddenLayers"],
            "dropout": self.config["model"]["DefaultDropout"]
        }
            
        # Hyperparameters grid
        hp_lr                   = hp.Choice("learning_rate", default=float(hp_default_dict["learning_rate"]), values=[float(x) for x in self.config["model"]["LearningRate"].split(', ')])
        loss_fun_classification = {
            "categorical_crossentropy" : tf.keras.losses.CategoricalCrossentropy()
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
        if problem == "regression"      :   
            hp_loss_fun_name    = hp.Choice("loss_fun", default=hp_default_dict["loss_fun"], values=self.config["model"]["LossFunctionRegression"].split(', '))
            hp_loss_fun         = loss_fun_regression[hp_loss_fun_name]
            hp_activation       = self.config["model"]["ActivationFunctionRegression"]
            hp_dense_units      = 1
        elif problem == "classification":   
            hp_loss_fun_name    = hp.Choice("loss_fun", default=hp_default_dict["loss_fun"], values=self.config["model"]["LossFunctionClassification"].split(', '))
            hp_loss_fun         = loss_fun_classification[hp_loss_fun_name]
            hp_activation       = self.config["model"]["ActivationFunctionClassification"]
            hp_dense_units      = 3
        hp_optimizer            = available_optimizers[hp.Choice("optimizer", default=hp_default_dict["optimizer"], values=self.config["model"]["Optimizer"].split(', '))]
        hp_units                = hp.Int("units", default=int(hp_default_dict["units"]), min_value=int(self.config["model"]["LSTMUnitsMin"]), max_value=int(self.config["model"]["LSTMUnitsMax"]), step=32)
        hp_hidden_layers        = hp.Int("hidden_layers", default=int(hp_default_dict["hidden_layers"]), min_value=int(self.config["model"]["HiddenLayersMin"]), max_value=int(self.config["model"]["HiddenLayersMax"]), step=1)
        hp_dropout              = hp.Float("dropout", default=float(hp_default_dict["dropout"]), min_value=float(self.config["model"]["DropoutRateMin"]), max_value=float(self.config["model"]["DropoutRateMax"]), step=0.05)
        self.early_stopping_min_delta = {
            "MSE": float(self.config["model"]["LossMinDeltaMSE"]), 
            "MAPE": float(self.config["model"]["LossMinDeltaMAPE"]),
            "categorical_crossentropy": float(self.config["model"]["LossMinDeltaCategoricalCrossEntropy"])
            }[hp_loss_fun_name]
        
        # Sequential model
        layer_list = []
        for _ in range(hp_hidden_layers-1): 
            layer_list.append(layers.LSTM(hp_units, batch_input_shape=batch_input_shape, stateful=True, dropout=hp_dropout, return_sequences=True))
        layer_list.extend([
            layers.LSTM(hp_units, batch_input_shape=batch_input_shape, dropout=hp_dropout, stateful=True, return_sequences=False),
            layers.Dense(hp_dense_units, activation=hp_activation)])
        model = tf.keras.Sequential(layer_list)
        
        model.compile(optimizer = hp_optimizer, loss = hp_loss_fun  )

        return model

    def save_results(self) -> int:
        '''
        Save results and parameters to results directory
        '''
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
        self.logger.info("Success\n")
        
        return 0
    
    def get_latest_file(directory, pattern, extension, timestamp_pattern="%Y-%m-%d_%H-%M"):
        '''
        Retrieves latest filepath of given directory, file name pattern, timestamp pattern, and extension
        '''
        files = glob.glob(f"{directory}{pattern}*.{extension}")
        if not files: return None
        dates = [re.search(directory+pattern+'(.+?)\.'+extension, f).group(1) for f in files]
        latest_date = sorted(dates, key=lambda x: datetime.strptime(x, timestamp_pattern), reverse=True)[0]
        return f'{directory}{pattern}{latest_date}.{extension}'

    def get_gpu_mem_usage(self, i) -> None:
        '''
        Checks GPU memory usage between window models
        '''
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        self.logger.info(f"POST [{i}] GPU memory usage: {np.round(info.used/info.total*100, 2)}%")
