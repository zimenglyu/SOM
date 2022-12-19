import numpy as np
# from quicksom.som import SOM
import pandas as pd

# Get data
data_path = '/Users/zimenglyu/Downloads/mga_cyclone_10.csv'
data = pd.read_csv(data_path)
data = data.drop(columns=['DateTime'])
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
data =  data.to_numpy()
np.save("/Users/zimenglyu/Documents/git/SOM/input.npy", data)
