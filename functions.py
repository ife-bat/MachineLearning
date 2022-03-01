import numpy as np
import pandas as pd

'''_______________________Data work functions__________________________'''

# Function to select a dataset from a cell dataframe
def select_dataset(df, column):
    
    if not isinstance(column, (list, tuple)):
        column = [column]
        
    s = df.loc[:, column]
    print(f"Shape of selected packed dataset: {s.shape}")
    
    s = s.dropna()
    print(f"Shape of selected packed dataset without NaNs: {s.shape}")
    if s.empty:
        print("Non values found")
        return
    
    s = s.T.apply(pd.Series.explode).set_index("cycle_index")
    print(f"Shape of selected unpacked dataset: {s.shape}")

    return s

'''___________________________ Model functions__________________________'''

# Function to get errors
def get_errors(y_train, y_test, y_train_pred, y_test_pred):
    mse_cycles_train = mean_squared_error(np.power(10, y_train), np.power(10, y_train_pred), squared=False)
    mse_cycles_test = mean_squared_error(np.power(10, y_test), np.power(10, y_test_pred), squared=False)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_test_pred)
    
    return mse_cycles_train, mse_cycles_test, mape_train, mape_test
