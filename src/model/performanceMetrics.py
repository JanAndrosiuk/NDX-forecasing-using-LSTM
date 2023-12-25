from src.init_setup import *
import sys
import os
import pickle
import shutil
import numpy as np
from collections import Counter
import json
import time


class PerformanceMetrics:
    def __init__(self) -> None:
        self.setup = Setup()
        self.config = self.setup.config
        self.logger = logging.getLogger("Performance Metrics")
        self.logger.addHandler(logging.StreamHandler())
        self.eval_data = None
        self.eval_data_timestamp = None
        self.pred_thres = float(self.config["model"]["PredictionThreshold"])
        self.tr_cost = float(self.config["evaluation"]["TransactionCost"])
        self.export_path = ""

    def calculate_metrics(self, custom_timestamp=None):
        """
        Calculate ARC (for buy-and-hold strategy and LSTM strategy), ASD, IR, MLD (only for LSTM strategy)
        :return: equity line array for further visualization
        """

        # Load evaluation data for performance metrics
        self.load_eval_data(custom_timestamp)
        equity_line_array, returns_array = (
            self.equity_line(self.eval_data["Pred"].values, self.eval_data["Real"].values) 
        )
        pos_counter = Counter(np.sign(self.eval_data["Pred"].values))
        self.logger.info(f'[Positions] Long: {pos_counter[1]}, Short: {pos_counter[-1]}, 0: {pos_counter[0]}')

        # Save performance statistics to a dictionary
        metrics = {
            "[ARC_BH]": str(np.round(self.arc(self.eval_data["Real"].values) * 100, 2)) + "%",
            "[ARC_EQ]": str(np.round(self.arc(equity_line_array) * 100, 2)) + "%",
            "[ASD_EQ]": str(np.round(self.asd(equity_line_array), 4)),
            "[IR_EQ]": str(np.round(self.ir(equity_line_array), 4)),
            "[MLD_EQ]": str(np.round(self.mld(returns_array) * 100, 2)) + "% of the year"
        }

        # Save results to .json file
        performance_metrics_path = f'{self.setup.ROOT_PATH}{self.config["prep"]["ModelMetricsDir"]}performance_metrics_{self.eval_data_timestamp}.json'
        with open(performance_metrics_path, 'w') as fp:
            json.dump(metrics, fp, indent=4, sort_keys=False)
        shutil.copy2(performance_metrics_path, self.export_path)
        eq_line_path = f'{self.setup.ROOT_PATH}{self.config["prep"]["DataOutputDir"]}eq_line_{self.eval_data_timestamp}.pkl'
        self.logger.info(f'Saving Equity Line array: {eq_line_path}')
        with open(eq_line_path, 'wb') as handle:
            pickle.dump(np.asarray(equity_line_array), handle, protocol=pickle.HIGHEST_PROTOCOL)
        shutil.copy2(eq_line_path, self.export_path)

        self.logger.info(f'[ARC_BH]:    {metrics["[ARC_BH]"]}')
        self.logger.info(f'[ARC_EQ]:    {metrics["[ARC_EQ]"]}')
        self.logger.info(f'[ASD_EQ]:    {metrics["[ASD_EQ]"]}')
        self.logger.info(f'[IR_EQ]:     {metrics["[IR_EQ]"]}')
        self.logger.info(f'[MLD_EQ]:    {metrics["[MLD_EQ]"]}')

        return equity_line_array

    def load_eval_data(self, custom_timestamp = None) -> int:
        files = os.listdir(self.config["prep"]["DataOutputDir"])
        
        if custom_timestamp is None:
            pickles = [f'{self.config["prep"]["DataOutputDir"]}{f}' for f in files if ("model_eval_data" in f and f.endswith(".pkl"))]
            try: latest_file = max(pickles, key=os.path.getmtime)
            except ValueError as ve:
                self.logger.error("No file available. Please rerun the whole process / load data first.")
                sys.exit(1)
            self.logger.info(f"Found latest eval data pickle: {latest_file}")
            self.eval_data_timestamp = latest_file[-20:-4]
        else: 
            self.eval_data_timestamp = custom_timestamp
            custom_eval_data_path = f'{self.config["prep"]["DataOutputDir"]}model_eval_data_{custom_timestamp}.pkl'
            self.logger.debug(f'Trying to load eval data: {custom_eval_data_path}')
            try: 
                with open(custom_eval_data_path, 'rb') as handle: self.eval_data = pickle.load(handle)
            except FileNotFoundError as ve: 
                self.logger.error("Model evaluation data not found.")
                sys.exit(1)
            self.logger.info(f"Loaded model evaluation data pickle: {custom_eval_data_path}")

        self.export_path = f'{self.setup.ROOT_PATH}{self.config["prep"]["ExportDir"]}{self.eval_data_timestamp}/'
        if not os.path.isdir(self.export_path): os.mkdir(self.export_path)

        return 0

    def equity_line(self, predictions: np.array, actual_values: np.array) -> tuple:
        """
        Calculate returns from investment based on predictions for price change, and actual values
        used params:
            - threshold required to consider a prediction as reliable or not (0 by default)
            - transaction cost of changing the investment position. Counts double.

        table of possible cases:
        | previous | current | threshold    | decision   |
        |----------|---------|--------------|------------|
        | L        | L       | abs(x)>thres | L (keep)   |
        | L        | L       | abs(x)<thres | L (keep)   |
        | L        | S       | abs(x)>thres | S (change) |
        | L        | S       | abs(x)<thres | L (keep)   |
        | S        | L       | abs(x)>thres | L (change) |
        | S        | L       | abs(x)<thres | S (keep)   |
        | S        | S       | abs(x)>thres | S (keep)   |
        | S        | S       | abs(x)<thres | S (keep)   |

        :param predictions: array of values between [-1, 1]
        :param actual_values: array of actual values
        :return: returns array, counter of transactions below threshold, array of position indicators
        """
        returns_array = []
        
        eq_line_array = [actual_values[0]]
        eq_line_second_return = (actual_values[1] - actual_values[0]) / actual_values[0]
        if (predictions[0] > 0): eq_line_array.append(eq_line_array[0] * (1 + eq_line_second_return))
        elif (predictions[0] < 0): eq_line_array.append(eq_line_array[0] * (1 - eq_line_second_return))
        else: eq_line_array.append(eq_line_array[0])
        
        self.logger.debug(f'Real values (B&H) -> First 5: {actual_values[:5]}, Last 5: {actual_values[-5:]}')
        
        for i in range(1, actual_values.shape[0]-1, 1):
            # r = (actual_values[i] - actual_values[i-1])/actual_values[i-1]
            # print(self.eval_data.iloc[[i]])
            # print(f'{i} {predictions[i-1]} {predictions[i]} {r} {eq_line_array[i-1]}')
            
            if (predictions[i] == 0): _return_rate = 0 - int(predictions[i-1] == 0) * self.tr_cost

            # L/L
            elif (predictions[i-1] >= 0 and predictions[i] > 0): _return_rate = (actual_values[i+1] - actual_values[i]) / actual_values[i] - int(predictions[i] == 0) * self.tr_cost
            
            # L/S
            elif (predictions[i-1] >= 0 and predictions[i] < 0): _return_rate = (actual_values[i] - actual_values[i+1]) / actual_values[i] - 2 * self.tr_cost + int(predictions[i] == 0) * self.tr_cost
                
            # S/S
            elif (predictions[i-1] <= 0 and predictions[i] < 0): _return_rate = (actual_values[i] - actual_values[i+1]) / actual_values[i] - int(predictions[i] == 0) * self.tr_cost

            # S/L
            elif (predictions[i-1] <= 0 and predictions[i] > 0): _return_rate = (actual_values[i+1] - actual_values[i]) / actual_values[i] - 2 * self.tr_cost + int(predictions[i] == 0) * self.tr_cost

            returns_array.append(_return_rate)
            eq_line_array.append(eq_line_array[i] * (1 + _return_rate))
            
        return np.asarray(eq_line_array), np.asarray(returns_array)

    def asd(self, equity_array, scale=252):
        """
        Annualized Standard Deviation
        :param equity_array: array of investment return for each timestep
        :param scale: number of days required for normalization. By default, in a year there are 252 trading days.
        :return: ASD as percentage
        """

        # Differentiate the returns array to get percentages instead of nominal values
        return_diffs = self.diff_array(equity_array)

        return np.std(return_diffs) * np.sqrt(scale)

    @staticmethod
    def diff_array(values_array: np.array, cost: float = 0) -> np.array:
        """
        Calculates discrete differences as percentages of the previous value
        :param values_array: Real prices array.
        :param cost: Constant cost for each timestep
        :return: Differentiated array
        """
        results_array = []
        for i in range(values_array.shape[0]):
            if i == 0:
                continue
            else:
                results_array.append((values_array[i] - values_array[i-1]) / values_array[i-1] - cost)
        return np.asarray(results_array)

    @staticmethod
    def arc(equity_array: np.array, scale: int = 252) -> float:
        """
        Annualized Return Ratio
        :param equity_array: equity line
        :param scale: number of days required for normalization. By default in a year there are 252 trading days.
        :return: ARC as percentage
        """
        return (equity_array[-1] / equity_array[0]) ** (scale / len(equity_array)) - 1

    def ir(self, equity_array: np.array, scale=252) -> float:
        """
        Information Ratio
        :param equity_array: Equity Line array
        :param equity_array: array of investment return for each timestep
        :param scale: number of days required for normalization. By default in a year there are 252 trading days.
        """
        return self.arc(equity_array, scale) / self.asd(equity_array, scale)

    @staticmethod
    def mld(returns_array: np.array, scale: int = 252) -> float:
        """
        Maximum Loss Duration
        Maximum number of time steps when the returns were below 0
        :param returns_array: array of investment returns
        :param scale: number of days required for normalization. By default, in a year there are 252 trading days.
        :return: MLD
        """
        max_loss = 0
        curr = 0
        for i in range(returns_array.shape[0]):

            # If first returns is negative, add this occurrence to max loss counter
            # If it's positive, continue
            if i == 0 and returns_array[0] < 0:
                curr += 1
                max_loss = curr

            # If the equity continues dropping
            elif (i > 0) and (returns_array[i-1] < 0) and (returns_array[i] < 0):
                curr += 1
                if max_loss < curr:
                    max_loss = curr

            # If the equity stops dropping
            elif (i > 0) and (returns_array[i-1] < 0) and (returns_array[i] > 0):
                curr = 0

        # Normalize over the number of trading days in a year
        return max_loss / scale
