#!/usr/bin/env python
# -*- coding: utf-8 -*-
from preprocessing import *
# from splitWindows import *
from modelFitPredict import *


def main():

    # Preprocessing
    # icsa_shift()
    # technical_indicators()
    # concatenate_dfs()

    # Training the model, saving predictions for each window
    predictions = model_fit_predict()

    # Saving results and parameters
    save_results(predictions)

    return 0

if __name__ == "__main__":
    main()
