#!/usr/bin/env python
# -*- coding: utf-8 -*-
import src.data as pr
import src.model as mod
import src.visualization as vis
from datetime import datetime


def main():
    start_time = datetime.now()
    
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
    custom_timestamp = None
    metrics = mod.PerformanceMetrics()
    metrics.calculate_metrics(custom_timestamp)

    # Visualize results
    vs = vis.Plots(start_time)
    vs.load_performance_data(custom_timestamp)
    vs.hist()
    vs.equity_line()
    # vs.multi_equity_line([
    #     "2024-01-28_02-04"
    #     ,"2024-01-28_03-01"
    #     ,"2024-01-28_03-58"
    #     ,"2024-01-28_04-54"
    #     ,"2024-01-28_05-50"
    #     ,"2024-01-28_10-54"
    #     ,"2024-01-28_13-02"
    #     ,"2024-01-28_13-59"
    #     ,"2024-01-28_14-56"
    #     ,"2024-01-28_15-54"
    #     ,"2024-01-28_16-50"
    #     ,"2024-01-28_17-48"
    #     ,"2024-01-28_18-45"
    #     ,"2024-01-28_19-43"
    # ])

    return 0


if __name__ == "__main__":
    main()
