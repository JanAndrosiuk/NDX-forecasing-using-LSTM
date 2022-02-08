#!/usr/bin/env python
# -*- coding: utf-8 -*-
from preprocessing import *
# from splitWindows import *
from modelFitPredict import *


def main():

    # PREPROCESSING
    # icsa_shift()
    # technical_indicators()
    # concatenate_dfs()

    # TRAINING THE MODEL, SAVING PREDICTIONS FOR EACH WINDOW
    predictions = model_fit_predict()
    print(
        len(predictions),
        predictions[-5:]
    )
    return 0

if __name__ == "__main__":
    main()
