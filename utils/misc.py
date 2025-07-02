import os, re
import os.path as osp
import random
import time
import json
import pickle
import numpy as np


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def mkdir_and_rename(path):
    """
    Make directory. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


def check_pairings(setting, config_dir, pairings):
    file_name = "pairing_gt_partial_full.json" if setting == "partial_full" else "pairing_gt_full_full_and_partial_partial.json"
    with open(os.path.join(config_dir, file_name), 'r') as json_file:
        gt_pairings = json.load(json_file)
    return pairings == gt_pairings


def sort_list(l):
    try:
        return list(sorted(l, key=lambda x: int(re.search(r'\d+(?=\.)', x).group())))
    except AttributeError:
        return sorted(l)


def read_pickle(path):
    with open(path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def write_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def save_error_to_file(file_path):
    import traceback
    with open(file_path, "a") as file:
        traceback.print_exc(file=file)
        file.write("\n")

