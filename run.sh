# train_spectra='/Users/zimenglyu/Downloads/mga_cyclone_10.csv'
# dataset_name=MGAspectra

train_spectra='/Users/zimenglyu/Documents/datasets/microbeam/may_2021_field_test_10.csv'
dataset_name=2021FieldTest

for n in 5 6 7 8 9 10
do

    python umatrix.py --train_spectra $train_spectra \
    --test_spectra /Users/zimenglyu/Documents/datasets/microbeam/may_2021_field_test_10.csv \
    --test_lab_result /Users/zimenglyu/Documents/datasets/microbeam/May_2021_Field_Test_fuel_properties_10.csv \
    --n_columns $n --n_rows $n --n_epochs 10000 \
    --plot_umatrix True --plot_hist True --dataset_name $dataset_name
done