# train_spectra='/Users/zimenglyu/Downloads/mga_cyclone_10.csv'
# dataset_name=MGAspectra

train_spectra='/Users/zimenglyu/Documents/datasets/microbeam/may_2021_field_test_10.csv'
dataset_name=2021FieldTest

# for n in 7
# do
#     for neighbor in 3 4
#     do

#         python umatrix.py --train_spectra $train_spectra \
#         --test_spectra /Users/zimenglyu/Documents/datasets/microbeam/may_2021_field_test_10.csv \
#         --test_lab_result /Users/zimenglyu/Documents/datasets/microbeam/May_2021_Field_Test_fuel_properties_10_test.csv \
#         --n_columns $n --n_rows $n --n_epochs 10000 \
#         --plot_umatrix True --plot_hist True --dataset_name $dataset_name \
#         --num_neighbors $neighbor
#     done
# done

for n in 7
do
    for neighbor in 5
    do

        python umatrix.py  \
        --ct_train /Users/zimenglyu/Documents/datasets/microbeam/MGA-CT_MGA_10.csv \
        --ct_test /Users/zimenglyu/Documents/datasets/microbeam/MGA-CT_MGA_10_test.csv \
        --n_columns $n --n_rows $n --n_epochs 10000 \
        --plot_umatrix True --dataset_name $dataset_name \
        --num_neighbors $neighbor
    done
done