from src.init_setup import *
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json


class Plots:
    def __init__(self) -> None:
        self.setup = Setup()
        self.config = self.setup.config
        self.logger = logging.getLogger("Generate Visualizations")
        self.logger.addHandler(logging.StreamHandler())
        self.eval_data = None
        self.timestamp = None
        self.model_config_dict = None
        self.eq_line_array = None
        self.window_dict = None
        self.vis_dir = self.config["prep"]["VisualizationsDir"]
        if not os.path.isdir(self.vis_dir):
            os.mkdir(self.vis_dir)
        self.csfont = {'fontname': 'Times New Roman'}

    def load_latest_data(self) -> int:
        files = os.listdir(self.config["prep"]["DataOutputDir"])
        pickles = [f'{self.config["prep"]["DataOutputDir"]}{f}' for f in files
                   if ("model_eval_data" in f and f.endswith(".pkl"))]
        self.logger.debug(f'Found pickles: {pickles}')
        latest_file = max(pickles, key=os.path.getmtime)

        # Save timestamp to read other files from the run
        self.timestamp = latest_file[-20:-4]

        # Load evaluation data
        self.logger.info(f"Loading latest eval data pickle: {latest_file}")
        with open(latest_file, 'rb') as handle:
            self.eval_data = pickle.load(handle)

        # Load split windows dictionary
        with open(self.config["prep"]["WindowSplitDict"], 'rb') as handle:
            self.window_dict = pickle.load(handle)

        # Load model config dictionary
        model_config_dict_path = f'{self.config["prep"]["ReportDir"]}model_config_{self.timestamp}.json'
        self.logger.info(f"Loading model config json: {model_config_dict_path}")
        with open(model_config_dict_path, 'rb') as handle:
            self.model_config_dict = json.load(handle)

        # Load model equity line
        model_eqline_dict_path = f'{self.config["prep"]["DataOutputDir"]}eq_line_{self.timestamp}.pkl'
        self.logger.info(f"Loading model equity line: {model_eqline_dict_path}")
        with open(model_eqline_dict_path, 'rb') as handle:
            self.eq_line_array = pickle.load(handle)

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
        plt.xlabel("Predictions", **self.csfont)
        plt.ylabel("Frequency", **self.csfont)
        plt.title("Histogram of model predictions", fontsize=13, **self.csfont)

        # Save histogram to .png file
        plt.savefig(f'{self.vis_dir}predictions_histogram_{self.timestamp}.png')

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

        # Load model parameters and attach them to the plot
        txt = ''''''
        for k, v in self.model_config_dict.items():
            txt += str(k) + ": " + str(v) + '\n'

        # Plot the equity line
        plt.figure(figsize=(25, 14))
        plt.text(
            self.window_dict['dates_test'].reshape(-1)[0], int(np.max(self.eq_line_array) * 0.6),
            txt, fontsize=12, fontdict=self.csfont
        )
        plt.plot(self.window_dict['dates_test'].reshape(-1), self.eq_line_array.reshape(-1))
        plt.plot(self.window_dict['dates_test'].reshape(-1), self.window_dict['closes_test'].reshape(-1), color='black')

        ax = plt.gca()
        ax.set_title(
            'Equity Line - Buy&Hold vs LSTM model', va='center', fontsize=15, fontweight='bold', fontdict=self.csfont
        )
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        plt.gcf().autofmt_xdate()

        # Add vertical lines indicating time-steps
        xticks = ax.get_xticks()
        for xtick in xticks:
            ax.vlines(
                x=xtick,
                ymin=0,
                ymax=max(self.window_dict['closes_test'].reshape(-1)),
                ls='--', alpha=0.3, linewidth=0.5, colors='black'
            )

        # Save the plot to .png file
        plt.savefig(f'{self.vis_dir}equity_line_{self.timestamp}.png')

        if show_results:
            plt.show()

        return 0
