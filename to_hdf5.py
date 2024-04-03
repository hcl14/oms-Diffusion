import h5py
import numpy as np
import os

base_path = './'  # dataset path
save_path = 'dresscode.hdf5'  # path to save the hdf5 file
hf = h5py.File(save_path, 'a')  # open the file in append mode

def store_annots(annots_path, category, hf):
    with open(annots_path, 'rb') as f:
        binary_data = f.read()
    binary_data_np = np.asarray(binary_data)
    dset = hf.create_dataset(category, data=binary_data_np)

grp = hf.create_group('annots')
train_path = '/workspace/DressCode_train/DressCode_1024/annots/train_pairs_230729.txt'
ratio_path = '/workspace/DressCode_train/DressCode_1024/annots/person_clothes_ratio_upper_train.txt'
test_paired_path = '/workspace/DressCode_train/DressCode_1024/annots/test_pairs_paired_230729.txt'
test_unpaired_path = '/workspace/DressCode_train/DressCode_1024/annots/test_pairs_unpaired_230729.txt'
store_annots(train_path, 'train', grp)
store_annots(ratio_path, 'ratio', grp)
store_annots(test_paired_path, 'test_paired', grp)
store_annots(test_unpaired_path, 'test_unpaired', grp)

for i in os.listdir(base_path):   # read all the As'
    if i not in ['upper', 'lower', 'dresses']:
        continue

    vid_name = os.path.join(base_path, i)
    grp = hf.create_group(vid_name)  # create a hdf5 group.  each group is one of ['upper', 'lower', 'dresses']
    for j in os.listdir(vid_name):  # read all subfolders inside group
        track = os.path.join(vid_name, j)
        subgrp = grp.create_group(j)  # create a subgroup for the above created group. each small
        for k in os.listdir(track):   # find all images inside subfolders.
            # can be file of any type. we are reading it in binary format.
            img_path = os.path.join(track, k)
            with open(img_path, 'rb') as img_f:
                binary_data = img_f.read()
            binary_data_np = np.asarray(binary_data)
            dset = subgrp.create_dataset(k, data=binary_data_np)
hf.close()
