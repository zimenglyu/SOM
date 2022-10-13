from os import umask
import susi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from susi.SOMPlots import plot_umatrix
import argparse
from typing import List, Tuple
import matplotlib
import math

def normalize_data(X):
    data = X.drop(columns=['DateTime'])
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    data = data.to_numpy()
    return data

def count_occurance(X, num_row, num_column):
    count = np.zeros((num_row, num_column))
    for x in X:
        count[x[0], x[1]] += 1
    return count

# def shortest_path_matrix(bum_list, u_matrix, n_rows, n_columns):
#     num_points = len(bum_list)
#     shortest_distance = np.zeros((num_points, num_points))
#     for i in range(num_points):
#         for j in range(num_points):
#             bum1 = bum_list[i]
#             bum2 = bum_list[j]
#             shortest_distance[i,j] = find_shortest_path(bum1, bum2, u_matrix)
#     return shortest_distance

# def find_shortest_path(bum1, bum2, u_matraix):
#     m = abs(bum1[0] - bum2[0])
#     n = abs(bum1[1] - bum2[1])
#     if m == 0 and n == 0:
#         return 0
#     elif m == 0:
        


def save_umatrix(u_matrix, args, y_pred=None):
    plot_umatrix(u_matrix, args.n_rows, args.n_columns)
    if y_pred is not None:
        plt.scatter(y_pred[:,1] *2, y_pred[:, 0]*2, c='r')
    up = []
    medium = []
    down = []
    leftup = []
    leftdown = []
    leftmedium = []
    for i in range(len(y_pred)):
        if [y_pred[i,1] *2, y_pred[i, 0]*2] not in up:  
            plt.text(y_pred[i,1] *2, y_pred[i, 0]*2, str(i+2))
            up.append([y_pred[i,1] *2, y_pred[i, 0]*2])
        else:
            if [y_pred[i,1] *2, y_pred[i, 0]*2] not in medium:
                plt.text(y_pred[i,1] *2, y_pred[i, 0]*2-0.5, str(i+2))
                medium.append([y_pred[i,1] *2, y_pred[i, 0]*2])
            else:
                if [y_pred[i,1] *2, y_pred[i, 0]*2] not in down:
                    plt.text(y_pred[i,1] *2, y_pred[i, 0]*2+0.5, str(i+2))
                    down.append([y_pred[i,1] *2, y_pred[i, 0]*2])
                else:
                    if [y_pred[i,1] *2, y_pred[i, 0]*2] not in leftup:
                        plt.text(y_pred[i,1] *2-1, y_pred[i, 0]*2, str(i+2))
                        leftup.append([y_pred[i,1] *2, y_pred[i, 0]*2])
                    else:
                        if [y_pred[i,1] *2, y_pred[i, 0]*2] not in leftdown:
                            plt.text(y_pred[i,1] *2-1, y_pred[i, 0]*2+0.5, str(i+2))
                            leftdown.append([y_pred[i,1] *2, y_pred[i, 0]*2])
                        else:
                            plt.text(y_pred[i,1] *2-1, y_pred[i, 0]*2-0.5, str(i+2))
                            leftmedium.append([y_pred[i,1] *2, y_pred[i, 0]*2])
    plot_name = "{}_umatrix_{}_{}_e{}.png".format(args.dataset_name, args.n_rows, args.n_columns, args.n_epochs)
    plot_title = "{} umatrix".format(args.dataset_name)
    plt.title(plot_title)
    plt.savefig(plot_name, bbox_inches='tight')
    
def save_som_hist(bums, args, y_pred, max_count = 100):

    matplotlib.rcParams.update({'font.size': 22})
    ax = plot_som_histogram(bums, args.n_rows, args.n_columns, n_datapoints_cbar=max_count)
    if y_pred is not None:
        ax.scatter(y_pred[:,0]+0.2, y_pred[:, 1]+0.2, c='r', linewidths=5)
    up = []
    medium = []
    down = []
    leftup = []
    leftdown = []
    leftmedium = []
    for i in range(len(y_pred)):
        if [y_pred[i,0], y_pred[i, 1]] not in up:  
            ax.text(y_pred[i,0] , y_pred[i, 1], str(i+2))
            up.append([y_pred[i,0] , y_pred[i, 1]])
        else:
            if [y_pred[i,0] , y_pred[i, 1]] not in medium:
                ax.text(y_pred[i,0] , y_pred[i, 1]-0.2, str(i+2))
                medium.append([y_pred[i,0] , y_pred[i, 1]])
            else:
                if [y_pred[i,0] , y_pred[i, 1]] not in down:
                    ax.text(y_pred[i,0] , y_pred[i, 1]+0.2, str(i+2))
                    down.append([y_pred[i,0] , y_pred[i, 1]])
                else:
                    if [y_pred[i,0] , y_pred[i, 1]] not in leftup:
                        ax.text(y_pred[i,0] -0.4, y_pred[i, 1], str(i+2))
                        leftup.append([y_pred[i,0] , y_pred[i, 1]])
                    else:
                        if [y_pred[i,0] , y_pred[i, 1]] not in leftdown:
                            ax.text(y_pred[i,0] -0.4, y_pred[i, 1]+0.2, str(i+2))
                            leftdown.append([y_pred[i,0] , y_pred[i, 1]])
                        else:
                            ax.text(y_pred[i,0] -0.4, y_pred[i, 1]-0.2, str(i+2))
                            leftmedium.append([y_pred[i,0] , y_pred[i, 1]])
    plot_name = "{}_hist_{}_{}_e{}.png".format(args.dataset_name, args.n_rows, args.n_columns, args.n_epochs)
    plot_title = "{} hist".format(args.dataset_name)
    # ax.title(plot_title)
    plt.savefig(plot_name, bbox_inches='tight')

def plot_som_histogram(
    bmu_list: List[Tuple[int, int]],
    n_rows: int,
    n_columns: int,
    n_datapoints_cbar: int = 5,
    fontsize: int = 10,
) -> plt.Axes:
    """Plot 2D Histogram of SOM.

    Plot 2D Histogram with one bin for each SOM node. The content of one
    bin is the number of datapoints matched to the specific node.

    Parameters
    ----------
    bmu_list  : list of (int, int) tuples
        Position of best matching units (row, column) for each datapoint
    n_rows : int
        Number of rows for the SOM grid
    n_columns : int
        Number of columns for the SOM grid
    n_datapoints_cbar : int, optional (default=5)
        Maximum number of datapoints shown on the colorbar
    fontsize : int, optional (default=22)
        Fontsize of the labels

    Returns
    -------
    ax : pyplot.axis
        Plot axis

    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    # colormap
   
    # cmaplist = [cmap(i) for i in range(cmap.N)]
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    #     "mcm", cmaplist, cmap.N
    # )
    
    # cmap = mpl.cm.cool
    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=1, vmax=n_datapoints_cbar)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=ax, orientation='vertical', label='Number of datapoints')

    # bounds = np.arange(0.0, n_datapoints_cbar + 1, 1.0)
    # norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    ax2 = fig.add_axes([0.96, 0.12, 0.03, 0.76])
    cbar = matplotlib.colorbar.ColorbarBase(
        ax2,
        cmap=cmap,
        norm=norm,
        spacing="proportional",
        format="%1i",
        extend="max",
    )
    cbar.ax.set_ylabel("Number of datapoints", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    ax.hist2d(
        [x[0] for x in bmu_list],
        [x[1] for x in bmu_list],
        bins=[n_rows, n_columns],
        cmin=1,
        cmap=cmap,
        norm=norm,
    )


    # for label in cbar.ax.xaxis.get_ticklabels()[::2]:
    #     label.set_visible(False)

    ax.set_xlabel("SOM columns", fontsize=fontsize)
    ax.set_ylabel("SOM rows", fontsize=fontsize)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
        
    ax.xaxis.label.set_color('tab:gray')
    ax.tick_params(axis='x', colors='tab:gray')
    ax.yaxis.tick_left()
    # to be compatible with plt.imshow:
    # ax.invert_yaxis()

    # plt.grid(b=False)

    return ax

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
    
    n_columns = args.n_columns
    n_rows = args.n_rows
    
    # spectra columns with all zeros should be removed before normalization 
    train_data = pd.read_csv(args.train_spectra)
    test_spectra = pd.read_csv(args.test_spectra)
    test_lab_result = pd.read_csv(args.test_lab_result)
    test_data_points = pd.merge(test_spectra, test_lab_result, on="DateTime")

    data = normalize_data(train_data)
    test_data = normalize_data(test_data_points)
    # spectra columns with all zeros should be removed before normalization, so the actual number of input columns is not 512
    num_spectra_columns = len(data[0])
    
    som = susi.SOMClustering(n_rows=n_rows, n_columns=n_columns, n_iter_unsupervised=args.n_epochs)
    som.fit(data)
    
    if args.test_spectra is not None and args.test_lab_result is not None:
        prediction = som.transform(test_data[:, :num_spectra_columns])
    else:
        prediction = None
        
    if args.plot_umatrix:
        u_matrix = som.get_u_matrix()
        # print("umatrix size is: {}".format(np.array(u_matrix).shape))
        # print(u_matrix)
        save_umatrix(u_matrix, args, prediction)
    
    if args.plot_hist:
        bums = som.get_bmus(data)
        # print(len(bums))
        bums = np.array(bums)
        print(bums.shape)
        count = count_occurance(bums, n_rows, n_columns)
        # print(count)
        # print("som of count is {}".format(np.sum(count)))
        # print("maximum count is {}".format(np.max(count)))
        
            
        # print(Counter(bums).keys() )# equals to list(set(words))
        # print(Counter(bums).values() )# counts the elements' frequency
        save_som_hist(bums, args, prediction, int(np.max(count)))
