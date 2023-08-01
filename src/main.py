#!/usr/bin/env python
# -*- coding: utf-8 -*-
import src.data as pr
import src.model as m
# from performanceMetrics import *
# from visualizeResults import *


def main():

    # Preprocessing
    # prep = pr.TrainPrep()
    # prep.prep_icsa()
    # prep.prep_tis()
    # prep.join_inputs()

    # Data split into train-test windows
    # ws = pr.WindowSplit()
    # ws.generate_windows()

    # Fit, Predict, save predictions
    # fp = m.RollingLSTM()
    # # fp.model_fit_predict()
    # fp.save_results()

    # # Calculate and save performance metrics
    metrics = m.PerformanceMetrics()
    metrics.load_latest_eval_data()
    metrics.calculate_metrics()

    # # Visualize results and save those visualizations
    # pred_hist()
    # plot_equity_line(predictions, equity_line)

    return 0


if __name__ == "__main__":
    main()
