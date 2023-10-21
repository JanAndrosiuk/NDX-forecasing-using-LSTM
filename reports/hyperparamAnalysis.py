import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt

# hyperparamLogList = glob.glob('./reports/model_hyperparams*.json')
hyperparamLogList = ['./reports/model_hyperparams_2023-10-16_09-15.json']
for f in hyperparamLogList:
    with open(f, "r") as fh:
        data = json.load(fh)
    print(f)
    df = pd.DataFrame.from_dict(data)
    print(df.tail(10))
    # print(
    #     df.agg({'learning_rate' : ['min', 'mean', 'max'], 'units' : ['min', 'mean', 'max'], 'dropout': ['min', 'mean', 'max'], 'hidden_layers': ['min', 'mean', 'max']})
    # )
    # print(
    #     df[['loss_fun', 'optimizer']].value_counts()
    # )
    df[['learning_rate', 'units', 'dropout']].hist(bins=70)
    plt.title(f)
    plt.show()