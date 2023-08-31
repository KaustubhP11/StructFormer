import os
from random import shuffle
from math import floor
import shutil
from pathlib import Path

all_files=os.listdir("/data2/venkat/parser_code/new_pklfiles_2k_12k/")
# all_files.remove("test")
# all_files.remove("train")
shuffle(all_files)
split=0.9
split_index=floor(len(all_files)*split)
train=all_files[:split_index]
test=all_files[split_index:]

for fname in train:
    shutil.copy2(os.path.join("/data2/venkat/parser_code/new_pklfiles_2k_12k/",fname),"/data2/venkat/parser_code/dataset_pkl_all/train/")

print("train done")

for fname in test:
    shutil.copy2(os.path.join("/data2/venkat/parser_code/new_pklfiles_2k_12k/",fname),"/data2/venkat/parser_code/dataset_pkl_all/test/")

print("test done")
