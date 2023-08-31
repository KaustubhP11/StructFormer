import shutil, os
import random

input_path = "/data2/venkat/parser_code/new_pklfiles_2k_12k/"
test_output_path = "/data2/venkat/parser_code/new_pklfiles_2k_12k_test/"
train_output_path = "/data2/venkat/parser_code/new_pklfiles_2k_12k_train/"


all_files = [name for name in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, name)) and name.endswith('.pkl')]
random.shuffle(all_files)

test_set = random.sample(all_files, len(all_files)//10)
train_set = []

train_set = [ item for item in all_files if item not in test_set]


for item in test_set:
    shutil.copy(input_path+item, test_output_path)
for item in train_set:
    shutil.copy(input_path+item, train_output_path)


    # dpython3 train_mlm.py --output_dir /data1/venkat/program/master/tarun/scratch_train/output_preTraining8/
