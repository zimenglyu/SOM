import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import pearsonr   
from scipy.stats import zscore   
from sklearn.preprocessing import StandardScaler

umatrix_path = "/Users/zimenglyu/Documents/git/SOM/umatrix_plots/"
CT_matrix_path = "/Users/zimenglyu/Documents/git/SOM/l2_distance.csv"

umatrix_files = glob(umatrix_path + "MGAspectra_distance*.csv")

def get_correlation(x1, x2):
    x1 = x1.flatten()
    x2 = x2.flatten()
    pearson = np.corrcoef(x1, x2)
    # pearson_2 = pearsonr(umatrix_dist, CT_dist)
    print (pearson[0,1])

def distance_zscore(x1, x2):
    # distance = x1-x2
    zscore_x1 = StandardScaler().fit_transform(x1)
    zscore_x2 = StandardScaler().fit_transform(x2)
    z = zscore(zscore_x1-zscore_x2)
    print(np.nansum(np.absolute(z)))
    
for file in umatrix_files:
    print(file)
    umatrix_dist = pd.read_csv(file).to_numpy()
    CT_dist = pd.read_csv(CT_matrix_path).to_numpy()
    get_correlation(umatrix_dist, CT_dist)
    distance_zscore(umatrix_dist, CT_dist)
    
    
    
