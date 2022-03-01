import numpy as np
import pandas as pd

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


