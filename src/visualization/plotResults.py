from src.init_setup import *
import sys
import os
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil.relativedelta import relativedelta
import json
import shutil

class Plots:
    def __init__(self, program_start_time: datetime) -> None:
        self.setup = Setup()
        self.config = self.setup.config
        self.logger = logging.getLogger("Generate Visualizations")
        self.logger.addHandler(logging.StreamHandler())
        self.logger.info("[[Visualizations module]]")  
        self.eval_data = None
        self.timestamp = None
        self.model_config_dict, self.perf_metr_dict = None, None
        self.eq_line_array = None
        self.window_dict = None
        self.vis_dir = self.config["prep"]["VisualizationsDir"]
        if not os.path.isdir(self.vis_dir): os.makedirs(self.vis_dir)
        self.export_path = "" 
        plt.rcParams["font.family"] = "monospace"
        plt.rcParams['axes.titlepad'] = 20
        self.program_start_time = program_start_time
        

    def load_performance_data(self, custom_timestamp=None) -> int:
        if custom_timestamp is None:
            files = os.listdir(self.config["prep"]["DataOutputDir"])
            pickles = [f'{self.config["prep"]["DataOutputDir"]}{f}' for f in files
                    if ("model_eval_data" in f and f.endswith(".pkl"))]
            self.logger.debug(f'Found pickles: {pickles}')
            try: latest_file = max(pickles, key=os.path.getmtime)
            except ValueError as ve:
                print("No file available. Please rerun the whole process / load data first.")
                sys.exit(1)
            self.timestamp = latest_file[-20:-4]
            eval_data_path = latest_file
        else:
            self.timestamp = custom_timestamp
            eval_data_path = f'{self.config["prep"]["DataOutputDir"]}model_eval_data_{self.timestamp}.pkl'
            
        self.export_path = f'{self.setup.ROOT_PATH}{self.config["prep"]["ExportDir"]}{self.timestamp}/'
        if not os.path.isdir(self.export_path): os.mkdir(self.export_path)
        
        # Load model evaluation data
        self.logger.info(f"Loading latest eval data pickle:\t{eval_data_path}")
        with open(eval_data_path, 'rb') as handle:
            self.eval_data = pickle.load(handle)
        self.logger.info("Loaded\n")    
            
        # Load split windows dictionary
        with open(self.config["prep"]["WindowSplitDict"], 'rb') as handle:
            self.window_dict = pickle.load(handle)
        
        # Load peroformance metrics dictionary
        perf_metr_dict_path = f'{self.config["prep"]["ModelMetricsDir"]}performance_metrics_{self.timestamp}.json'
        self.logger.info(f"Loading performance metrics json:\t{perf_metr_dict_path}")
        with open(perf_metr_dict_path, 'rb') as handle:
            self.perf_metr_dict = json.load(handle)
        self.logger.info("Loaded\n")

        # Load model equity line
        model_eqline_dict_path = f'{self.config["prep"]["DataOutputDir"]}eq_line_{self.timestamp}.pkl'
        # model_eqline_dict_path = f'reports/export/{self.timestamp}/eq_line_{self.timestamp}.pkl'
        self.logger.info(f"Loading model equity line:\t\t{model_eqline_dict_path}")
        with open(model_eqline_dict_path, 'rb') as handle:
            self.eq_line_array = pickle.load(handle)
        self.logger.info("Loaded\n")

        return 0

    def hist(self, bins: int = 500, show_results: bool = False) -> int:
        """
        Plot the histogram of predicted values
        :param bins: number of histogram bins, 500 by default
        :param show_results: True if results should be shown, False by default
        :return: 0 if the process was successful
        """

        plt.figure(figsize=(8, 4))
        self.eval_data["Pred"].plot.hist(bins=bins, color="darkslategrey", ec="darkslategrey")
        plt.xlabel("Predictions")
        plt.ylabel("Frequency")
        plt.title("Histogram of model predictions", fontsize=13)

        # Save histogram to .png file
        plt.savefig(f'{self.vis_dir}predictions_histogram_{self.timestamp}.png')
        shutil.copy2(f'{self.vis_dir}predictions_histogram_{self.timestamp}.png', self.export_path)
        self.logger.info(f"Saved Predictions histogram:\t\t{self.vis_dir}predictions_histogram_{self.timestamp}.png")

        if show_results:
            plt.show()

        return 0

    def equity_line(self, show_results=False):
        """
        Function responsible for plotting the Equity Line
        Requires:
            - array of predictions
            - array of equity line values
            - model parameters (config) dictionary
        :param show_results: True if results should be shown
        :return: 0 if the process was successful
        """

        date_flat_vector = self.window_dict['dates_test'].reshape(-1)
        
        # Load model parameters and attach them to the plot
        m = self.perf_metr_dict
        
        txt = ''''''
        txt += f"{'-'*49}\n"
        txt += f"| Metric\t| Buy & Hold\t| Strategy\t|\n".expandtabs()
        txt += f"{'-'*49}\n"
        extra_tab = "\t" if float(m["EQ_ARC"]) >= 0 else ""
        txt += f'| aRC\t\t| {m["BH_ARC"]}%\t| {m["EQ_ARC"]}%\t{extra_tab}|\n'.expandtabs()
        txt += f'| aSD\t\t| {m["BH_ASD"]}\t| {m["EQ_ASD"]}\t|\n'.expandtabs()
        txt += f'| MLD\t\t| {m["BH_MLD"]}%\t\t| {m["EQ_MLD"]}%\t\t|\n'.expandtabs()
        txt += f'| IR\t\t| {m["BH_IR"]}\t| {m["EQ_IR"]}\t|\n'.expandtabs()
        txt += f'| IR**\t\t| {m["BH_IR**"]}\t| {m["EQ_IR**"]}\t|\n'.expandtabs()
        txt += f'| Positions\t| 1\t\t| {m["POS_CNT"]}\t\t|\n'.expandtabs()
        txt += f"{'-'*49}"
        txt_console_print = txt
        txt = f"{self.timestamp}\n{txt}"
        
        # Plot the equity line
        plt.figure(figsize=(25, 14))
        plt.text(
            date_flat_vector[0], int(np.max([self.eq_line_array, self.eval_data["Real"]]) * 0.75),
            txt, fontsize=12
        )
        
        plt.plot(date_flat_vector, self.eq_line_array.reshape(-1))
        plt.plot(date_flat_vector, self.window_dict['closes_test'].reshape(-1), color='black')

        ax = plt.gca()
        ax.set_title('Equity Line: Buy&Hold vs LSTM', va='center', fontsize=18)
        
        min_year = np.datetime64(date_flat_vector[0], 'Y').astype(int) + 1970
        max_year = np.datetime64(date_flat_vector[-1], 'Y').astype(int) + 1970
        
        # X-ticks and vertical lines
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        base = datetime.date(min_year-1, 1, 1)
        date_list = [base + relativedelta(years=x) for x in range(1, max_year-min_year+3)]
        for date in date_list:
            ax.vlines(date, ymin=0, ymax=max(self.window_dict['closes_test'].reshape(-1)),
                ls='--', alpha=0.3, linewidth=0.5, colors='black')

        # Save the plot to .png file
        plt.savefig(f'{self.vis_dir}equity_line_{self.timestamp}.png')
        shutil.copy2(f'{self.vis_dir}equity_line_{self.timestamp}.png', self.export_path)
        self.logger.info(f"Saved Equity Line plot:\t\t\t{self.vis_dir}equity_line_{self.timestamp}.png")

        if show_results:
            plt.show()
        
        self.logger.info(f"\nRESULTS:\n{txt_console_print}\n")
        program_execution_delta = datetime.datetime.now() - self.program_start_time
        program_execution_time = (datetime.datetime(1970,1,1,0,0,0) + program_execution_delta).strftime("%H:%M")
        self.logger.info(f"Program ended successfully:\t{time.strftime('%d-%m-%Y %H:%M')}\nExecution time:\t\t\t{program_execution_time}\n{'-'*120}")
        
        return 0
    
    def multi_equity_line(self, timestamps_to_load):
        
        # Load equity line arrays
        equity_line_arrays_list = []
        for t in timestamps_to_load:
            model_eqline_dict_path = f'{self.config["prep"]["DataOutputDir"]}eq_line_{t}.pkl'
            with open(model_eqline_dict_path, 'rb') as handle:
                equity_line_arrays_list.append(pickle.load(handle))
                
        default_eq_line_shape = equity_line_arrays_list[0].shape
        for el in equity_line_arrays_list: 
            if el.shape != default_eq_line_shape:
                self.logger.error("When ploting multiple equity lines, equity line arrays must be equal by shape.")
                raise Exception
        
        # Load split windows dictionary (regenerate if the latest is not up-to-date)
        with open(self.config["prep"]["WindowSplitDict"], 'rb') as handle: self.window_dict = pickle.load(handle)
        date_flat_vector = self.window_dict['dates_test'].reshape(-1)
        
        # Load Buy&Hold data from 1st timestamp entry
        eval_data_path = f'{self.config["prep"]["DataOutputDir"]}model_eval_data_{timestamps_to_load[0]}.pkl'
        with open(eval_data_path, 'rb') as handle: buy_and_hold_array = pickle.load(handle)["Real"]
        
        # Load performance metrics dictionaries
        performance_metric_dictionaries_list = []
        for t in timestamps_to_load:
            perf_metr_dict_path = f'{self.config["prep"]["ModelMetricsDir"]}performance_metrics_{t}.json'
            with open(perf_metr_dict_path, 'rb') as handle:
                performance_metric_dictionaries_list.append(json.load(handle))
        
        # Create a dictionary with average test results
        mean_res_dict = {
            "EQ_ARC": 0.0
            ,"EQ_ASD": 0.0
            ,"EQ_MLD": 0.0
            ,"EQ_IR": 0.0
            ,"EQ_IR**": 0.0
            ,"POS_CNT": 0.0
        }
        for perf_dict in performance_metric_dictionaries_list:
            for k in perf_dict.keys():
                if k in mean_res_dict.keys():
                    mean_res_dict[k] += float(perf_dict[k])/len(timestamps_to_load)
        
        mean_res_dict["EQ_ARC"] = str(format(mean_res_dict["EQ_ARC"], '.2f'))
        mean_res_dict["EQ_ASD"] = str(format(mean_res_dict["EQ_ASD"], '.4f'))
        mean_res_dict["EQ_MLD"] = str(format(mean_res_dict["EQ_MLD"], '.2f'))
        mean_res_dict["EQ_IR"] = str(format(mean_res_dict["EQ_IR"], '.4f'))
        mean_res_dict["EQ_IR**"] = str(format(mean_res_dict["EQ_IR**"], '.4f'))
        mean_res_dict["POS_CNT"] = str(format(mean_res_dict["POS_CNT"], '.0f'))
        
        txt = ''''''
        txt += f"{'-'*49}\n"
        txt += f"| Metric\t| Buy & Hold\t| Strategy mean\t|\n".expandtabs()
        txt += f"{'-'*49}\n"
        extra_tab = "\t" if float(mean_res_dict["EQ_ARC"]) >= 0 else ""
        txt += f'| aRC\t\t| {performance_metric_dictionaries_list[0]["BH_ARC"]}%\t| {mean_res_dict["EQ_ARC"]}%\t{extra_tab}|\n'.expandtabs()
        txt += f'| aSD\t\t| {performance_metric_dictionaries_list[0]["BH_ASD"]}\t| {mean_res_dict["EQ_ASD"]}\t|\n'.expandtabs()
        txt += f'| MLD\t\t| {performance_metric_dictionaries_list[0]["BH_MLD"]}%\t\t| {mean_res_dict["EQ_MLD"]}%\t\t|\n'.expandtabs()
        txt += f'| IR\t\t| {performance_metric_dictionaries_list[0]["BH_IR"]}\t| {mean_res_dict["EQ_IR"]}\t|\n'.expandtabs()
        txt += f'| IR**\t\t| {performance_metric_dictionaries_list[0]["BH_IR**"]}\t| {mean_res_dict["EQ_IR**"]}\t|\n'.expandtabs()
        txt += f'| Positions\t| 1\t\t| {mean_res_dict["POS_CNT"]}\t\t|\n'.expandtabs()
        txt += f"{'-'*49}"
        
        # Plot the equity line
        plt.figure(figsize=(25, 14))
        plt.text(
            date_flat_vector[0], int(np.max([equity_line_arrays_list[0], buy_and_hold_array]) * 0.75),
            txt, fontsize=12
        )
        
        for eq_line_array in equity_line_arrays_list:
            plt.plot(date_flat_vector, eq_line_array.reshape(-1))
        plt.plot(date_flat_vector, self.window_dict['closes_test'].reshape(-1), color='black', linewidth=3.0)

        ax = plt.gca()
        ax.set_title('Equity Line: Buy&Hold vs LSTM', va='center', fontsize=18)
        
        min_year = np.datetime64(date_flat_vector[0], 'Y').astype(int) + 1970
        max_year = np.datetime64(date_flat_vector[-1], 'Y').astype(int) + 1970
        
        # X-ticks and vertical lines
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        base = datetime.date(min_year-1, 1, 1)
        date_list = [base + relativedelta(years=x) for x in range(1, max_year-min_year+3)]
        for date in date_list:
            ax.vlines(date, ymin=0, ymax=max(self.window_dict['closes_test'].reshape(-1)),
                ls='--', alpha=0.3, linewidth=0.5, colors='black')
        
        # Save the plot to .png file
        plt.savefig(f'{self.vis_dir}equity_line_summary_{timestamps_to_load[-1]}.png')
        self.export_path = f'{self.setup.ROOT_PATH}{self.config["prep"]["ExportDir"]}{timestamps_to_load[-1]}/'
        shutil.copy2(f'{self.vis_dir}equity_line_summary_{timestamps_to_load[-1]}.png', self.export_path)
        self.logger.info(f"Saved Equity Line summary plot for {len(timestamps_to_load)} test cases:\t{self.vis_dir}equity_line_summary{timestamps_to_load[-1]}.png")

        self.logger.info(f"\nRESULTS:\n{txt}\n")        
        return 0
