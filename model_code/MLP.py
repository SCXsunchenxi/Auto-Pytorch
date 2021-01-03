from autoPyTorch import AutoNetRegression

# Other imports for later usage
import pandas as pd
import numpy as np
import os as os
import openml
import json
from sklearn.model_selection import train_test_split


# data with six strains
def load_data(plane):
    '''
     :param plane: choose plane P123-P27
    '''

    print('[***] Plane ' + plane)

    dir = '/Users/sunchenxi/Documents/Github/PLANE611/data_per_plane/'
    data = pd.read_csv(dir + plane + '_data.csv', encoding='utf-8')
    # M = pd.read_csv(dir+plane+'_mean.csv', encoding='utf-8')
    # S = pd.read_csv(dir+plane+'_sd.csv', encoding='utf-8')

    X = np.asarray(data.get(data.columns.values.tolist()[1:31]))
    Y = np.asarray(data.get(data.columns.values.tolist()[31:]))

    print('[***] Data is loaded')

    return X, Y

if __name__ == "__main__":

    autonet = AutoNetRegression(config_preset="tiny_cs", result_logger_dir="logs/")

    # Get the current configuration as dict
    current_configuration = autonet.get_current_autonet_config()

    # Get the ConfigSpace object with all hyperparameters, conditions, default values and default ranges
    hyperparameter_search_space = autonet.get_hyperparameter_search_space()

    # load data
    plane='P123'
    X, Y = load_data(plane)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    results_fit = autonet.fit(X_train=X_train,
                              Y_train=Y_train,
                              validation_split=0.3,
                              max_runtime=300,
                              min_budget=60,
                              max_budget=100,
                              refit=True)

    # Save fit results as json
    with open("logs/results_fit.json", "w") as file:
        json.dump(results_fit, file)

