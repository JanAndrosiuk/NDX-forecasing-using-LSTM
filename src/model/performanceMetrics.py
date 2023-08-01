from src.init_setup import *
import os
import pickle
import numpy as np
import json
from termcolor import colored


class PerformanceMetrics:
    def __init__(self) -> None:
        self.setup = Setup()
        self.config = self.setup.config
        self.logger = logging.getLogger("Performance Metrics")
        self.eval_data = None
        self.eval_data_timestamp = None
        self.threshold = float(self.config["model"]["TargetThreshold"])
        self.tr_cost = float(self.config["evaluation"]["TransactionCost"])

    def calculate_metrics(self):
        """
        Calculate ARC (for buy-and-hold strategy and LSTM strategy), ASD, IR, MLD (only for LSTM strategy)
        :return: equity line array for further visualization
        """

        # Load evaluation data for performance metrics
        self.load_latest_eval_data()
        returns_array = self.returns(self.eval_data["Pred"].values, self.eval_data["Real"].values)[0]
        equity_line = self.eq_line(returns_array, self.eval_data["Real"].values[0])

        # Save performance statistics to a dictionary
        metrics = {
            "[ARC_BH]": str(np.round(self.arc(self.eval_data["Real"].values) * 100, 2)) + "%",
            "[ARC_EQ]": str(np.round(self.arc(equity_line) * 100, 2)) + "%",
            "[ASD_EQ]": str(np.round(self.asd(equity_line), 4)),
            "[IR_EQ]": str(np.round(self.ir(equity_line), 4)),
            "[MLD_EQ]": str(np.round(self.mld(returns_array) * 100, 2)) + "% of the year"
        }

        # Save results to .json file
        report_dir = self.setup.ROOT_PATH + self.config["prep"]["ReportDir"]
        if not os.path.isdir(report_dir):
            os.mkdir(report_dir)
        with open(f'{report_dir}performance_metrics_{self.eval_data_timestamp}.json', 'w') as fp:
            json.dump(metrics, fp, indent=4, sort_keys=False)

        # Print results
        print(colored("[ARC_BH]", 'blue'), metrics['[ARC_BH]'])
        print(colored("[ARC_EQ]", 'blue'), metrics['[ARC_EQ]'])
        print(colored("[ASD_EQ]", 'blue'), metrics['[ASD_EQ]'])
        print(colored("[IR_EQ]", 'blue'), metrics['[IR_EQ]'])
        print(colored("[MLD_EQ]", 'blue'), metrics['[MLD_EQ]'])

        return equity_line

    def load_latest_eval_data(self) -> int:
        files = os.listdir(self.config["prep"]["DataOutputDir"])
        pickles = [f'{self.config["prep"]["DataOutputDir"]}{f}' for f in files
                   if ("model_eval_data" in f and f.endswith(".pkl"))]
        self.logger.debug(f'Found pickles: {pickles}')
        latest_file = max(pickles, key=os.path.getmtime)
        self.logger.info(f"Found latest eval data pickle: {latest_file}")

        # Save timestamp from the filename in a static way
        self.eval_data_timestamp = latest_file[-20:-4]

        with open(latest_file, 'rb') as handle:
            self.eval_data = pickle.load(handle)

        return 0

    def returns(self, predictions, actual_values):
        """
        Calculate returns from investment based on predictions for price change, and actual values
        used params:
            - threshold required to consider a prediction as reliable or not
            - transaction cost of changing the investment position. Counts double.

        :param predictions: array of values between [-1, 1]
        :param actual_values: array of actual values
        :return: returns array, counter of transactions below threshold, array of position indicators
        """
        positions = [1]  # store positions (1 for long, 0 for stay, -1 for short), first is always long
        counter = 0  # count positions inside the threshold
        returns_array = []  # output array
        for i in range(actual_values.shape[0]):  # for i in actual values

            # first position is always long
            if i == 0:
                continue

            # if prediction is outside of threshold and isn't the first one
            elif (predictions[i] > self.threshold or predictions[i] < -self.threshold) and (i != 0):

                # LONG BUY
                if predictions[i] > self.threshold:

                    # if previous was long, don't include the transaction cost
                    if positions[-1] == 1:
                        returns_array.append(actual_values[i] - actual_values[i - 1])
                        positions.append(1)

                    # if previous was short, include the transaction cost for changing the position
                    else:
                        returns_array.append(actual_values[i] - actual_values[i - 1] - 2 * self.tr_cost)
                        positions.append(1)

                # SHORT SELL
                elif predictions[i] < -self.threshold:

                    # if previous was short, don't include transactions cost
                    if positions[-1] == -1 and i != 1:
                        returns_array.append(actual_values[i - 1] - actual_values[i])
                        positions.append(-1)

                    # if previous was long, include the transaction cost for changing the position
                    else:
                        returns_array.append(actual_values[i - 1] - actual_values[i] - 2 * self.tr_cost)
                        positions.append(-1)

            # if predictions is inside threshold, but isn't the first one
            else:
                counter += 1
                returns_array.append(0)
                positions.append(0)

        return np.asarray(returns_array), counter, positions

    @staticmethod
    def eq_line(returns_array, _n_value):
        """
        Calculate the equity line of investment
        Equity Line informs about the change of accumulated capital
        :param returns_array: array of returns from investment
        :param _n_value: Initial value of capital
        :return: returns an array where each element represents the capital at the given timestep
        """

        # first value is the initial value of the capital
        equity_array = [_n_value]

        # Walk through every return in returns array
        # For each daily investment return, add (the current equity value + daily change) to the array
        for i, x in enumerate(returns_array):
            equity_array.append(equity_array[i] + x)

        return np.asarray(equity_array)

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
        Calculates dicrete differences as percentages of the previous value
        :param values_array: Real prices array.
        :param cost: Constant cost for each timestep
        :return: np.array
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
