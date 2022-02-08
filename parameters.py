import time
name = f'Model_{time.strftime("%Y%m%d-%H%M%S")}'

# below one can choose which parameters will be included in the model training
# it can be either Technical Indicators data, OHLCV data, or just closing prices of an index
features = [
    'ICSA', 'SMA5', 'SMA10', 'SMA50', 'EMA20', 'stoch5', 'ADOSC',
   'MACDhist', 'WILLR', 'RSI', 'MOM', 'ROC', 'OBV', 'CCI'
]
# features = ['Open', 'High', 'Low', 'Close', 'Volume]
# features = ['Close']

# a threshold referring to the minimal price change that indicates a rise or fall
# this parameter is used in preprocessing.transform_target
threshold = 0.001

train_period = 504  # in days
test_period = 252  # in days
lookback = 63  # in days, how many previous days should be taken into account

batch = 252  # batch size
units = 30  # number of LSTM units
epochs = 100
lr = 0.01  # learning rate
dropout = 0.1  # dropout rate on each of LSTM layers
loss = 'mean_squared_error'  # loss function
act = 'tanh'  # activation function

# opt and hidden parameters below are only set to be included in dictionary.
# to take effect, those have to be changed in buildModel.model_builder
opt = 'Adam'  # Optimizer used in the learning process
hidden = 2  # Number of the model hidden layers

desc = {
    'Model Name': name,
    'Features': f'{features}',
    'Probability threshold': f'{threshold}',
    'Look-back period': f'{lookback}',
    'Training period': f'{train_period}',
    'Test period': f'{test_period}',
    'LSTM layer units': f'{units}',
    'Dropout rate': f'{dropout}',
    'Activation function': act,
    'Initial learning rate': f'{lr}',
    'loss function': loss,
    'Number of epochs': f'{epochs}',
    'Batch size': f'{batch}',
    'Optimizer': opt,
    'Number of hidden layers': f'{hidden}'
}
