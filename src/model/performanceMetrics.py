from parameters import *
import os
import numpy as np
from pandas import read_csv
import json
from termcolor import colored


def diff_array(values_array, cost=0):
    """
    Calculates dicrete differences as percentages of the previous value
    :param values_array: Real prices array. However, could be any array.
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


def returns(predictions, actual_values, thres=threshold, tr_cost=0.0005):
    """
    Calculate returns from investment based on predictions for price change, and actual values
    :param predictions: array of values between [-1, 1]
    :param actual_values: array of actual values
    :param thres: threshold of considering a prediction as reliable
    :param tr_cost: transaction cost of changing the investment position. Counts double.
    :return: np.array
    """
    positions = [1]  # store positions (1 for long, 0 for stay, -1 for short), first is always long
    counter = 0  # count positions inside the threshold
    returns_array = []  # output array
    for i in range(actual_values.shape[0]):  # for i in actual values

        # first position is always long
        if i == 0:
            continue

        # if prediction is outside of threshold and isn't the first one
        elif (predictions[i] > thres or predictions[i] < -thres) and (i != 0):

            # LONG BUY
            if predictions[i] > thres:

                # if previous was long, don't include the transaction cost
                if positions[-1] == 1:
                    returns_array.append(actual_values[i] - actual_values[i-1])
                    positions.append(1)

                # if previous was short, include the transaction cost for changing the position
                else:
                    returns_array.append(actual_values[i] - actual_values[i-1] - 2 * tr_cost)
                    positions.append(1)

            # SHORT SELL
            elif predictions[i] < -thres:

                # if previous was short, don't include transactions cost
                if positions[-1] == -1 and i != 1:
                    returns_array.append(actual_values[i-1] - actual_values[i])
                    positions.append(-1)

                # if previous was long, include the transaction cost for changing the position
                else:
                    returns_array.append(actual_values[i-1] - actual_values[i] - 2 * tr_cost)
                    positions.append(-1)

        # if predictions is inside threshold, but isn't the first one
        else:
            counter += 1
            returns_array.append(0)
            positions.append(0)
    return np.asarray(returns_array), counter, positions


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


def arc(equity_array, scale=252):
    """
    Annualized Return Ratio
    :param equity_array: equity line
    :param scale: number of days required for normalization. By default in a year there are 252 trading days.
    :return: ARC as percentage
    """
    return (equity_array[-1] / equity_array[0]) ** (scale / len(equity_array)) - 1


def asd(equity_array, scale=252):
    """
    Annualized Standard Deviation
    :param equity_array: array of investment return for each timestep
    :param scale: number of days required for normalization. By default in a year there are 252 trading days.
    :return: ASD as percentage
    """

    # Differentiate the returns array to get percentages instead of nominal values
    return_diffs = diff_array(equity_array)

    return np.std(return_diffs) * np.sqrt(scale)


def ir(equity_array, scale=252):
    """
    Information Ratio
    :param equity_array: Equity Line array
    :param equity_array: array of investment return for each timestep
    :param scale: number of days required for normalization. By default in a year there are 252 trading days.
    """
    return arc(equity_array, scale) / asd(equity_array, scale)


def mld(returns_array, scale=252):
    """
    Maximum Loss Duration
    Maximum number of time steps when the returns were below 0
    :param returns_array: array of investment returns
    :param scale: number of days required for normalization. By default in a year there are 252 trading days.
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


def calculate_metrics(results_path=f'results/{name}_results.csv'):
    """
    Calculate ARC (for buy-and-hold strategy and LSTM strategy), ASD, IR, MLD (only for LSTM strategy)
    :param results_path: path to a dataframe with results (modelFitPredict.save_results)
    :return: equity line array for further visualization
    """

    # Load results and calculate necessary data to calculate performance metrics
    df_res = read_csv(results_path)
    returns_array = returns(df_res.Pred.values, df_res.Real.values)[0]
    equity_line = eq_line(df_res.Pred.values, df_res.Real.values[0])

    # Save performance statistics to a dictionary
    stats = {
        "[ARC_BH]": str(np.round(arc(df_res.Real.values) * 100, 2)) + "%",
        "[ARC_EQ]": str(np.round(arc(equity_line) * 100, 2)) + "%",
        "[ASD_EQ]": str(np.round(asd(equity_line), 4)),
        "[IR_EQ]": str(np.round(ir(equity_line), 4)),
        "[MLD_EQ]": str(np.round(mld(returns_array)*100, 2)) + "% of the year"
    }

    # Save results to .json file
    if not os.path.isdir('results/'):
        os.mkdir('results/')
    with open(f'results/{name}_metrics.json', 'w') as fp:
        json.dump(stats, fp, indent=4, sort_keys=False)

    # Print results
    print(colored("[ARC_BH]", 'blue'), stats['[ARC_BH]'])
    print(colored("[ARC_EQ]", 'blue'), stats['[ARC_EQ]'])
    print(colored("[ASD_EQ]", 'blue'), stats['[ASD_EQ]'])
    print(colored("[IR_EQ]", 'blue'), stats['[IR_EQ]'])
    print(colored("[MLD_EQ]", 'blue'), stats['[MLD_EQ]'])

    return equity_line
