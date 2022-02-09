## About the Project

Here I present the wrapper for evaluating rolling stacked LSTM model on historical financial data. The default model uses Nasdaq100 index data between 01-10-1985 and 11-01-2021 (*data/OHLC_NDX.csv*). Aparth from the candlestick data (OHLCV), it also uses *TA-Lib* library for technical indicators.

## Specific requirements

- Tensorflow (with GPU acceleration enabled, preferably) and other python libraries listed in `requirements.txt`
- TA-Lib library [[installation]](https://blog.quantinsti.com/install-ta-lib-python/)
- Financial time series data - candle stick data [[example source - yahoo finance]](https://finance.yahoo.com/quote/%5ENDX/history?p=%5ENDX)
- Initial Claims time series data - [[source - Federal Reserve Bank of St. Louis]](https://fred.stlouisfed.org/series/ICSA)

## Description of modules

`parameters.py` - set model parameters regarding name of the model, LSTM parameters, train and test lengths, used features, etc.

`preprocessing.py` - adding technical indicators data and processes Initial Claims data. After that it combines all of the data frames to *data_preprocessed/*

`splitWindows.py` - splits the data into equal sized chunks prepared for the rolling LSTM, calculating performance metrics, and visualizations

`buildModel.py` - compiles the stacked LSTM model framework

`modelFitPredict.py` - trains the model, and generates predictions for each window. After that, saves the results to *results/*

`performanceMetrics.py` - calculates and saves performance metrics regarding hypothetical investment returns

`visualizeResults.py` - visualizations. E.g. Equity Line for each time step

`main.py` - combines all the modules

## Remarks

Although the method didn't yield very significant remarks, it may serve as the base template for more extensive analysis.

Further improvements may include:

- [ ] Averaging the results from many runtimes (random seed cannot be currently set due to the large amount of stochastic processes)
- [ ] Hyperparameters tuning between windows
- [ ] Including genetic algorithm / hillclimb algorithm in hyperparameter tuning between windows

## License

MIT License | Copyright (c) 2021 Jan Androsiuk
