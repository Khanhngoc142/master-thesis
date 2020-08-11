import os
import re


def get_source_root():
    cur_path = os.path.abspath('.')
    root_path = re.findall(r"^(.*hmer).*$", cur_path)[0]
    return root_path