## About the Project

Rolling LSTM modelling framework for candlestick data with an addition of technical indicators.

## Requirements
- common python packages: `pip install -r requirements.txt` 
- **Tensorflow** python package (preferably with GPU acceleration)
- **TA-Lib** python package [[installation]](https://blog.quantinsti.com/install-ta-lib-python/)
- Example financial time series data: (NDX OHLCV candlestick) [[yahoo finance]](https://finance.yahoo.com/quote/%5ENDX/history?p=%5ENDX)
- Initial Claims time series data [[Initial Claims - Federal Reserve Bank of St. Louis]](https://fred.stlouisfed.org/series/ICSA)

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

Further improvements to be included:

- [ ] Averaging the results from many runtimes (random seed cannot be currently set due to the large amount of stochastic processes)
- [ ] Hyper-param tuning between windows
- [ ] Real time approach

## License

MIT License | Copyright (c) 2021 Jan Androsiuk
