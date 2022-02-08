#!/usr/bin/env python
# -*- coding: utf-8 -*-
from modelFitPredict import *
from performanceMetrics import *
from visualizeResults import *


def main():

    # Preprocessing
    icsa_shift()
    technical_indicators()
    concatenate_dfs()

    # Training the model, saving predictions for each window
    predictions = model_fit_predict()

    # Saving results and parameters
    save_results(predictions)

    # Calculate and save performance metrics
    equity_line = calculate_metrics()

    # Visualize results and save those visualizations
    pred_hist()
    plot_equity_line(predictions, equity_line)

    return 0


if __name__ == "__main__":
    main()
