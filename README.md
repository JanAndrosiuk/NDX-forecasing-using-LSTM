## About the Project

Rolling LSTM modelling framework for candlestick data with an addition of technical indicators.

## Requirements
- common python packages: `pip install -r requirements.txt` 
- **Tensorflow** python package (preferably with GPU acceleration)
- **TA-Lib** python package [[installation]](https://blog.quantinsti.com/install-ta-lib-python/)
- Example financial time series data: (NDX OHLCV candlestick) [[yahoo finance]](https://finance.yahoo.com/quote/%5ENDX/history?p=%5ENDX)
- Initial Claims time series data [[Initial Claims - Federal Reserve Bank of St. Louis]](https://fred.stlouisfed.org/series/ICSA)

## Description of modules
tbd
## Remarks

Further improvements to be included:

- [ ] Averaging the results from many runtimes (random seed cannot be currently set due to the large amount of stochastic processes)
- [ ] Hyper-param tuning between windows
- [ ] Real time approach

## License

MIT License | Copyright (c) 2021 Jan Androsiuk
