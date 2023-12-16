#!/usr/bin/env python
# -*- coding: utf-8 -*-
import src.data as pr
import src.model as mod
import src.visualization as vis


def main():

    # Preprocessing
    prep = pr.TrainPrep()
    prep.prep_icsa()
    prep.prep_tis()
    prep.join_inputs()

    # Data split into train-test windows
    ws = pr.WindowSplit()
    ws.generate_windows()

    # Fit, Predict, save predictions
    fp = mod.RollingLSTM()
    fp.model_fit_predict_multiprocess(save=True)
    fp.save_results()

    # Get performance metrics
    metrics = mod.PerformanceMetrics()
    metrics.load_eval_data()
    metrics.calculate_metrics()

    # Visualize results
    vs = vis.Plots()
    vs.load_performance_data()
    vs.hist()
    vs.equity_line()

    return 0


if __name__ == "__main__":
    main()
