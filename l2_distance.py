import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler


file_name = "/Users/zimenglyu/Documents/datasets/microbeam/May_2021_Field_Test_fuel_properties_10.csv"

relevantproperties = [
    "Total Moisture as rec'd %",
    "Ash as rec'd %",
    "HHV as rec'd BTU/lb",
    "SiO2 dry %",
    "CaO dry %",
    "Na2O dry %",
    "B/A calc'd",
]

data = pd.read_csv(file_name)
data = data[relevantproperties]
print(data.columns)
zscorenormdata = StandardScaler().fit_transform(data)
minmaxnormdata = MinMaxScaler().fit_transform(data)
# normalized_data=(data-data.mean(axis=0))/data.std(axis=0)
# normalized_data=(data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))

normalized_data = zscorenormdata
num_points = normalized_data.shape[0]
distance = np.zeros((num_points, num_points))

for i in range(num_points):
    for j in range(num_points):
        distance[i][j] = np.linalg.norm(normalized_data[i]-normalized_data[j]) # by default l2 distance

DF = pd.DataFrame(distance)
DF.index += 2 
DF.columns += 2
DF.to_csv("l2_distance.csv")