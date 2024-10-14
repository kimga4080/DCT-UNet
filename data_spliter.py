"""
Split the dataset into train, valication, and test dataset
"""

import numpy as np
from sklearn.model_selection import KFold
import random

def basic_CV_split(img_paths, fold=5, fold_ind=1, random_state=7, tr_val_ratio=0.8, *args, **kwargs):
    '''
    # Function for split CV fold and output tr/val/test path on specific fold index
    Args:
        img_path: path of image file
        fold: split of train, validation, test (int) 
        fold_ind: index of fold to split dataset (int)
        random_state = state for same split (int)
        val_test_ratio: ratio between validation and test (float)
        args, kwargs: any other parameters that might need to be passed in
    '''

    ind_ascend = np.arange(0, len(img_paths), 1)
    kf5 = KFold(n_splits=fold, random_state=random_state, shuffle=True)
    fold_iter = 0
    for tr_val_index, test_index in kf5.split(ind_ascend):
        fold_iter = fold_iter + 1

        if fold_iter == fold_ind:
            random.seed(7)
            random.shuffle(test_index)
            train_index = tr_val_index[:int(len(tr_val_index)*tr_val_ratio)]
            val_index = tr_val_index[int(len(tr_val_index)*tr_val_ratio):]

            tr_path = [img_paths[i] for i in train_index] 
            val_path = [img_paths[i] for i in val_index]
            test_path = [img_paths[i] for i in test_index] 

    return tr_path, val_path, test_path


def data_path_spliter(img_paths, fold=5, fold_ind=1, random_state=7, val_test_ratio=0.8, *args, **kwargs):
    '''
    # Function for output tr/val/test path
    Args:
        img_path: path of image file
        select_files: selected file (list)
        data_type: MR1, MR2, both, both_split (str)
        fold: split of train, validation, test (int) -> split dataset as fold, and used 1 fold for val/test, and remaining for train
        fold_ind: index of fold to split dataset (int)
        random_state = state for same split (int)
        val_test_ratio: ratio between validation and test (float)
        args, kwargs: any other parameters that might need to be passed in
    '''


    tr_paths, val_paths, test_paths = basic_CV_split(img_paths, fold, fold_ind, random_state, val_test_ratio)

    train_dicts = [{'image': tr_path,'label': tr_path[:-9] + 'label.nii'} for tr_path in tr_paths]
    val_dicts = [{'image': val_path,'label': val_path[:-9] + 'label.nii'} for val_path in val_paths]
    test_dicts = [{'image': test_path,'label': test_path[:-9] + 'label.nii'} for test_path in test_paths]

    return train_dicts, val_dicts, test_dicts