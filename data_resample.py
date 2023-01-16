import pandas as pd
import math
datadir = '/Users/zimenglyu/Documents/datasets/microbeam/2022_MGA_CT_10.csv'
data = pd.read_csv(datadir, parse_dates=["DateTime"], index_col="DateTime")
# data = data[(data['CycloneNumber']==10)]
# data.dropna(inplace=True)
# data['DateTime']= pd.to_datetime(data['DateTime'])
# print(data.loc[40])

new_data = data.resample("10T").mean()
new_data = new_data.apply (pd.to_numeric, errors='coerce')
new_data = new_data.dropna()
# print(new_data.iloc[40, :])
# print(new_data.head)
new_data.to_csv("/Users/zimenglyu/Documents/datasets/microbeam/2022_MGA_CT_10_resample.csv")
# path1 = "/Users/zimenglyu/Documents/datasets/microbeam/May_2021_Field_Test/FT_2021_CT_10.csv"
# data1 = pd.read_csv(path1, parse_dates=["DateTime"], index_col="DateTime")
# path2 = "/Users/zimenglyu/Documents/datasets/microbeam/may_2021_field_test_10.csv"
# data2 = pd.read_csv(path2, parse_dates=["DateTime"], index_col="DateTime")
# data3 = pd.merge(data1, data2, on="DateTime")
# data3.to_csv("/Users/zimenglyu/Documents/datasets/microbeam/May_2021_Field_Test/FT_2021_CT_spectra_test.csv")

# data_dir = "/Users/zimenglyu/Documents/datasets/microbeam/MGA_CT_10/resample/MGA-CT_10_test.csv"
# data = pd.read_csv(data_dir)
# for num_row in [30, 40, 50, 75, 100]:
#     total_rows = data.shape[0]
#     num_files = math.floor(total_rows/num_row)

#     for i in range(num_files):
#         sub_data = data[i*num_row: (i+1)*num_row]
#         sub_data.to_csv("/Users/zimenglyu/Documents/datasets/microbeam/MGA_CT_10/resample/{}/MGA-CT_10_test_".format(num_row) + str(i) + ".csv", index=False)