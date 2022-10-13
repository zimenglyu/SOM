from minisom import MiniSom  
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


data_path = '/Users/zimenglyu/Downloads/mga_cyclone_10.csv'
data = pd.read_csv(data_path)
data = data.drop(columns=['DateTime'])

data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

som_shape = (3, 3)
# print(data[0:5])
data = data.to_numpy()
som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=0.5, learning_rate=0.5, neighborhood_function='gaussian', random_seed=10) # initialization of 6x6 SOM
som.train_batch(data, 5000, verbose=True)
# som.train(data, 100) # trains the SOM with 100 iterations

# print(som.winner(data[0]))
# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in data]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

for c in np.unique(cluster_index):
    plt.scatter(data[cluster_index == c, 0],
                data[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)

# plotting centroids
for centroid in som.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
                s=40,  color='k', label='centroid')
plt.legend();
plt.show()