import susi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from susi.SOMPlots import plot_umatrix, plot_som_histogram
import argparse

def normalize_data(X):
    data = X.drop(columns=['DateTime'])
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    data = data.to_numpy()
    return data

def save_umatrix(u_matrix, args, y_pred=None):
    plot_umatrix(u_matrix, args.n_rows, args.n_columns)
    if y_pred is not None:
        plt.scatter(y_pred[:,1] *2, y_pred[:, 0]*2, c='r')
    plot_name = "{}_umatrix_{}_{}_e{}.png".format(args.dataset_name, args.n_rows, args.n_columns, args.n_epochs)
    plot_title = "{} umatrix".format(args.dataset_name)
    plt.title(plot_title)
    plt.savefig(plot_name, bbox_inches='tight')
    
def save_som_hist(bums, args):
    plot_som_histogram(bums, args.n_rows, args.n_columns)
    plot_name = "{}_hist_{}_{}_e{}.png".format(args.dataset_name, args.n_rows, args.n_columns, args.n_epochs)
    plot_title = "{} hist".format(args.dataset_name)
    plt.title(plot_title)
    plt.savefig(plot_name, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for SOM')

    parser.add_argument('--train_spectra', required=True, help='input training dataset with spectra')
    parser.add_argument('--test_spectra', required=False, help='test dataset with spectra')
    parser.add_argument('--test_lab_result', required=False, help='test data with lab results')
    parser.add_argument('--n_columns', required=True, type=int, help='number of columns for SOM')
    parser.add_argument('--n_rows', required=True, type=int, help='number of rows for SOM')
    parser.add_argument('--n_epochs', required=True, type=int, help='number of epochs for SOM training')
    parser.add_argument('--plot_umatrix', required=False, default=False, type=bool, help='option to plot and save umatrix plot')
    parser.add_argument('--plot_hist', required=False, default=False, type=bool, help='option to plot and save SOM hist graph')
    parser.add_argument('--dataset_name', required=False, default="sample", help='name of the dataset')

    args = parser.parse_args()
    # spectra columns with all zeros should be removed before normalization 
    train_data = pd.read_csv(args.train_spectra)
    test_spectra = pd.read_csv(args.test_spectra)
    test_lab_result = pd.read_csv(args.test_lab_result)
    test_data_points = pd.merge(test_spectra, test_lab_result, on="DateTime")

    data = normalize_data(train_data)
    test_data = normalize_data(test_data_points)
    # spectra columns with all zeros should be removed before normalization, so the actual number of input columns is not 512
    num_spectra_columns = len(data[0])
    
    som = susi.SOMClustering(n_rows=args.n_rows, n_columns=args.n_columns, n_iter_unsupervised=args.n_epochs)
    som.fit(data)
        
    if args.plot_umatrix:
        if args.test_spectra is not None and args.test_lab_result is not None:
            prediction = som.transform(test_data[:, :num_spectra_columns])
        else:
            prediction = None
        u_matrix = som.get_u_matrix()
        save_umatrix(u_matrix, args, prediction)
    
    if args.plot_hist:
        bums = som.get_bmus(data)
        save_som_hist(bums, args)
