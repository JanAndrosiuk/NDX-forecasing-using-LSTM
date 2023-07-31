from modelFitPredict import windows_dict
from parameters import *
import os
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import json


def pred_hist(df_results_path=f'results/{name}_results.csv', bins=500, show_results=False):
    """
    Plot the histogram of predicted values
    :param df_results_path: dataframe of results, currently used path by default
    :param bins: number of histogram bins, 500 by default
    :param show_results: True if results should be shown, False by default
    :return: 1 if the process was successful
    """
    df_results = read_csv(df_results_path)
    plt.figure(figsize=(8, 4))
    df_results['Pred'].plot.hist(bins=bins, color="darkslategrey", ec="darkslategrey")
    plt.xlabel("Predictions", **csfont)
    plt.ylabel("Frequency", **csfont)
    plt.title("Histogram of model predictions", fontsize=13, **csfont)

    # Save histogram to .png file
    if not os.path.isdir('visualizations/'):
        os.mkdir('visualizations/')

    plt.savefig(f'visualizations/{name}_predictions_histogram.png')

    if show_results:
        plt.show()

    return 1


def plot_equity_line(
    predictions, equity_line, param_dict_path=f'results/{name}_parameters.json',
    show_results=False
):
    """
    Function responsible for plotting the Equity Line
    :param predictions: array of predictions (modelFitPredict.model_fit_predict)
    :param equity_line: array of equity line values (performanceMetrics.calculate_metrics)
    :param param_dict_path: path to parameters dictionary (created in modelFitPredict.save_results)
    :param show_results: True if results should be shown
    :return: 1 if the process was successful
    """

    # Load model parameters and attach them to the plot
    with open(param_dict_path, 'r') as fp:
        param_dict = json.load(fp)
    txt = ''''''
    for k, v in param_dict.items():
        txt += str(k) + ": " + str(v) + '\n'

    # Plot the equity line
    plt.figure(figsize=(20, 14))
    plt.text(0, int(np.max(equity_line[1:]) * 0.6), txt, fontsize=14)
    plt.plot(windows_dict['dates_test'].reshape(-1), equity_line[1:])
    plt.plot(windows_dict['dates_test'].reshape(-1), windows_dict['closes_test'].reshape(-1), color='black')

    # Add vertical lines indicating time steps
    xcoords = [x for x in range(predictions.reshape(-1).shape[0]) if (x + 1) % 125 == 0 or x == 0]
    for xc in xcoords:
        plt.axvline(x=xc, c='black', ls='--', alpha=0.2, lw=1)

    # Correct x axis ticks and set the title
    x_ticks = range(0, windows_dict['dates_test'].reshape(-1).shape[0], 250)
    plt.xticks(x_ticks, fontsize=10, rotation=60)
    ax = plt.gca()
    ax.set_title('Equity Line', va='center', fontsize=15, fontweight='bold')

    # Save the plot to .png file
    if not os.path.isdir('visualizations/'):
        os.mkdir('visualizations/')
    plt.savefig(f'visualizations/{name}_equity_line.png')

    # Show results
    if show_results:
        plt.show()

    return 1
