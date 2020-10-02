import os
import re
import pickle


def get_source_root():
    cur_path = os.path.abspath('.')
    root_path = re.findall(r"^(.*hmer).*$", cur_path)[0]
    return root_path


def save_object(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def load_object(file):
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj


def get_path(rel_path):
    if rel_path.startswith("/"):
        return rel_path
    return os.path.join(get_source_root(), rel_path)
