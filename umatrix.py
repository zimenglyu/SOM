from os import umask
import susi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from susi.SOMPlots import plot_umatrix
import argparse
from typing import List, Tuple
import matplotlib
import sys
from shortest_path import Graph, dijkstra
import pandas as pd
from sklearn.metrics import mean_squared_error

def normalize_data(X, norm_method):
    data = X.drop(columns=['DateTime'])
    if (norm_method == "minmax"):
    # data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0) )
    elif (norm_method == "std"):
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    else:
        sys.exit("Error: wrong data normalization type")
    data = data.to_numpy()
    return data

def count_occurance(X, num_row, num_column):
    count = np.zeros((num_row, num_column))
    for x in X:
        count[x[0], x[1]] += 1
    return count

def shortest_path_matrix(bum_list, u_matrix, n_rows, n_columns):
    num_points = len(bum_list)
    shortest_distance = np.zeros((num_points, num_points)) - 1
    distance_graph = make_graph(u_matrix, n_rows,n_columns)
    for i in range(num_points):
        for j in range(num_points):
            bum1 = bum_list[i]
            bum2 = bum_list[j]
            shortest_distance[i,j] = find_shortest_path(bum1, bum2,distance_graph, n_columns)
    return shortest_distance

def find_shortest_path( bum1, bum2, distance_graph, n_col):
    m = abs(bum1[0] - bum2[0])
    n = abs(bum1[1] - bum2[1])
    if m == 0 and n == 0:
        return 0
    else:
        start = get_graph_index(bum1[0], bum1[1], n_col)
        end = get_graph_index(bum2[0], bum2[1], n_col)
        distance_graph.reset()
        D = dijkstra(distance_graph, start)
        return D[end]    
    
def get_graph_index(i,j,row_size):
    graph_i = i*row_size + j
    return graph_i

def make_graph(umatrix,m,n):
    distance_graph = Graph(m*n)
    d_m = len(u_matrix)-1

    for i in range(m):
        for j in range(n):
            graph_i = get_graph_index(i,j, n)
            if (i + 1 < m):
                graph_i_up = get_graph_index(i+1,j, n)
                distance_graph.add_edge(graph_i, graph_i_up, umatrix[d_m-(i*2+1),j*2])
            if (j + 1 < n): 
                graph_i_right = get_graph_index(i,j+1, n)
                distance_graph.add_edge(graph_i, graph_i_right, umatrix[d_m-(i*2),j*2+1])
    return distance_graph
               


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
    
def inverse_number(x, fenmu):
    if ( x == 0 ):
        # incase the number x is 0
        return 1/(fenmu)
    else: 
        return 1/x

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

    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=1, vmax=n_datapoints_cbar)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=ax, orientation='vertical', label='Number of datapoints')
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
    ax.set_xlabel("SOM columns", fontsize=fontsize)
    ax.set_ylabel("SOM rows", fontsize=fontsize)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
        
    ax.xaxis.label.set_color('tab:gray')
    ax.tick_params(axis='x', colors='tab:gray')
    ax.yaxis.tick_left()

    return ax

def get_correlation(x1, x2):
    x1 = x1.flatten()
    x2 = x2.flatten()
    pearson = np.corrcoef(x1, x2)
    # pearson_2 = pearsonr(umatrix_dist, CT_dist)
    print (pearson[0,1])

# generate random array, float between [0, 1]
def get_random_array(m,n):
    a = np.random.rand(m,n)
    return a

def save_shortest_distance_csv(args, shortest_distance):
    if (args.save_shortest_distance_csv):
        distance = pd.DataFrame(shortest_distance)
        distance.index += 2
        distance.columns += 2
        name = "{}_distance_{}_{}_e{}".format(args.dataset_name, args.n_rows, args.n_columns, args.n_epochs)
        distance.to_csv(name + ".csv")

def get_input_data(input_data_path, norm_method): 
    train_data = pd.read_csv(input_data_path)
    # print(test_data_points)
    train_data_norm = normalize_data(train_data, norm_method)
    return train_data_norm

def get_test_data(args, norm_method):
    # spectra columns with all zeros should be removed before normalization 
    test_spectra = pd.read_csv(args.test_spectra)
    test_lab_result = pd.read_csv(args.test_lab_result)
    test_lab_result = test_lab_result.sample(frac=1).reset_index(drop=True)   # shuffle data
    
    test_data_points = pd.merge(test_lab_result, test_spectra, on="DateTime")
    test_data = normalize_data(test_data_points, norm_method)
    test_ground_truth = normalize_data(test_lab_result, norm_method)
    
    return test_data, test_ground_truth

def get_ct_data(args, norm_method):
    train_data = pd.read_csv(args.ct_train)
    train_data_norm = normalize_data(train_data, norm_method)
    test_data = pd.read_csv(args.ct_test)
    test_data_norm = normalize_data(test_data, norm_method)
    return train_data_norm, test_data_norm

def estimate_value_ct(shortest_distance, num_neighbors, num_vali_points, num_properties, fenmu):
    weighted_neighbor_pred = np.zeros((num_vali_points, num_properties))
    average_neighbor_pred = np.zeros((num_vali_points, num_properties))
    random_neighbor_pred = np.zeros((num_vali_points, num_properties))
    num_neighbors = int(num_neighbors)
    for i in range(num_vali_points):
        close_index = np.argsort(shortest_distance[i])
        random_index = np.copy(close_index)
        np.random.shuffle(random_index)
        close_index = close_index[:num_neighbors]
        random_index = random_index[:num_neighbors]

        for k in range(num_properties):
            distance_sum_w = 0
            numerator_w = 0
            numerator_a = 0
            line_count = 0
            for j in close_index:
                part_distance = inverse_number(shortest_distance[i,j], fenmu)
                distance_sum_w += part_distance
                numerator_w += part_distance * test_ground_truth[j, k]
                
                numerator_a += test_ground_truth[j, k]
                line_count += 1
                
            average_neighbor_pred[i,k] = numerator_a / line_count
            weighted_neighbor_pred[i,k] = numerator_w / distance_sum_w
            
            numerator_r = 0
            line_count = 0
            for j in random_index:               
                numerator_r += test_ground_truth[j, k]
                line_count += 1
            random_neighbor_pred[i,k] = numerator_r / line_count
    return weighted_neighbor_pred, average_neighbor_pred, random_neighbor_pred
    
def estimate_value(shortest_distance, num_neighbors, num_vali_points, num_properties, fenmu):
    weighted_neighbor_pred = np.zeros((num_vali_points, num_properties))
    average_neighbor_pred = np.zeros((num_vali_points, num_properties))
    random_neighbor_pred = np.zeros((num_vali_points, num_properties))
    num_neighbors = int(num_neighbors)
    for i in range(num_vali_points):
        close_index = np.argsort(shortest_distance[i])
        random_index = np.copy(close_index)
        np.random.shuffle(random_index)
        close_index = close_index[:num_neighbors]
        random_index = random_index[:num_neighbors]

        for k in range(num_properties):
            distance_sum_w = 0
            numerator_w = 0
            numerator_a = 0
            line_count = 0
            for j in close_index:
                part_distance = inverse_number(shortest_distance[i,j], fenmu)
                distance_sum_w += part_distance
                numerator_w += part_distance * test_ground_truth[j, k]
                
                numerator_a += test_ground_truth[j, k]
                line_count += 1
                
            average_neighbor_pred[i,k] = numerator_a / line_count
            weighted_neighbor_pred[i,k] = numerator_w / distance_sum_w
            
            numerator_r = 0
            line_count = 0
            for j in random_index:               
                numerator_r += test_ground_truth[j, k]
                line_count += 1
            random_neighbor_pred[i,k] = numerator_r / line_count
    return weighted_neighbor_pred, average_neighbor_pred, random_neighbor_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for SOM')

    parser.add_argument('--train_spectra', required=False, help='input training dataset with spectra')
    parser.add_argument('--test_spectra', required=False, help='test dataset with spectra')
    parser.add_argument('--test_lab_result', required=False, help='test data with lab results')
    parser.add_argument('--ct_train', required=False, default=False, help='path to coal tracker train data')
    parser.add_argument('--ct_test', required=False, default=False, help='path to coal tracker test data')
    parser.add_argument('--n_columns', required=True, type=int, help='number of columns for SOM')
    parser.add_argument('--n_rows', required=True, type=int, help='number of rows for SOM')
    parser.add_argument('--n_epochs', required=True, type=int, help='number of epochs for SOM training')
    parser.add_argument('--plot_umatrix', required=False, default=False, type=bool, help='option to plot and save umatrix plot')
    parser.add_argument('--plot_hist', required=False, default=False, type=bool, help='option to plot and save SOM hist graph')
    parser.add_argument('--dataset_name', required=False, default="sample", help='name of the dataset')
    parser.add_argument('--num_neighbors', required=False, default=5, help='number of neighbors used for estimating values')
    parser.add_argument('--save_shortest_distance_csv', required=False, default=False, help='if write the shortest distance path to csv')
    
    args = parser.parse_args()
    n_columns = args.n_columns
    n_rows = args.n_rows
    num_neighbors = args.num_neighbors
    print("num neighbors {}".format(num_neighbors))
    # num_validation_points = args.vali_points
    
    # train_data = get_input_data(args.train_spectra, "minmax")
    # test_data, test_ground_truth = get_test_data(args, "minmax")
    
    train_data, test_data_all = get_ct_data(args, "minmax")
    test_data = test_data_all[:, 3:]
    test_ground_truth = test_data_all[:, :3]
    
    num_spectra_columns = len(train_data[0])
    num_vali_points = len(test_ground_truth)
    
    
    som = susi.SOMClustering(n_rows=n_rows, n_columns=n_columns, n_iter_unsupervised=args.n_epochs)
    som.fit(train_data)
            
    # if args.test_spectra is not None and args.test_lab_result is not None:
    prediction = som.transform(test_data[:, :num_spectra_columns])
    # else:
    #     prediction = None
        

    u_matrix = som.get_u_matrix()
         
    if args.plot_hist:
        bums = som.get_bmus(train_data)
        bums = np.array(bums)
        count = count_occurance(bums, n_rows, n_columns)
        save_som_hist(bums, args, prediction, int(np.max(count)))
            
    shortest_distance = shortest_path_matrix(prediction, u_matrix, n_rows, n_columns)
    save_shortest_distance_csv(args, shortest_distance)
    
    num_properties = len(test_ground_truth[0])
    print("num properties is {}, use {} neighbors".format(num_properties, num_neighbors))
    weighted_acc = []
    avg_acc = []
    random_neighbor_acc = []
    random_acc = []
    
    for fenmu in [0.1]:
        for i in range(1):
            test_data, test_ground_truth = get_test_data(args, "minmax")
            prediction = som.transform(test_data[:, :num_spectra_columns])
            shortest_distance = shortest_path_matrix(prediction, u_matrix, n_rows, n_columns)

            weighted_neighbor_pred, average_neighbor_pred, random_neighbor_pred = estimate_value(shortest_distance, num_neighbors, num_vali_points, num_properties, fenmu)
            weighted_som_acc = mean_squared_error(weighted_neighbor_pred, test_ground_truth)
            average_som_acc = mean_squared_error(average_neighbor_pred, test_ground_truth)
            random_som_acc = mean_squared_error(random_neighbor_pred, test_ground_truth)

            random_guess_array = get_random_array(num_vali_points, num_properties)
            random_guess_acc = mean_squared_error(random_guess_array, test_ground_truth)
            weighted_acc.append(weighted_som_acc)
            avg_acc.append(average_som_acc)
            random_neighbor_acc.append(random_som_acc)
            random_acc.append(random_guess_acc)
        print("inverse number is {}".format(fenmu))
        print("20 repeat average using {} neighbors: ".format(num_neighbors))
        print("weighted neighbor is {}, average_neighbor is {}, random neighbor acc is {}".format( np.mean(weighted_acc), np.mean(avg_acc), np.mean(random_neighbor_acc)))
        print("random guessing acc is {}".format(np.mean(random_acc)))