import hashlib
import json
import operator
import pathlib

import pickle
import shutil

import numpy
from scipy.io import loadmat, savemat

SOURCE_PATH = pathlib.Path(__file__).parents[1]
PROJECT_PATH = SOURCE_PATH.parent
DATA_PATH = PROJECT_PATH / "data"


def load_as_txt(file):
    with open(file) as f:
        return f.read()


def save_text(text: str, file):
    with open(file, 'w') as f:
        f.write(text)


def load_pkl(file):
    # return numpy.load(file, allow_pickle=True)
    with open(file, 'rb') as f:
        return pickle.load(f)


def load_npy_as_dict(file):
    data = numpy.load(file)
    if isinstance(data, dict):
        return data
    else:
        raise ValueError("load data is not a dict!")


def save_to_pkl(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def load_json_as_dict(file):
    with open(file) as json_file:
        data = json.load(json_file)
    return data


def save_dict_as_json(obj, file):
    with open(file, 'w') as json_file:
        json.dump(obj, json_file, indent=4)


def get_all_files(folder):
    if not isinstance(folder, pathlib.Path):
        folder = pathlib.Path(folder)
    return [f for f in folder.rglob("*") if f.is_file()]


def get_newest_file(folder):
    files = pathlib.Path(folder).glob("*")
    latest_file = max([f for f in files], key=lambda item: item.stat().st_ctime)
    return latest_file


def load_mat(file):
    return loadmat(file)


def save_mat(file, data):
    if type(data) is dict:
        savemat(file, data)
    else:
        savemat(file, {'data': data})


def cal_file_md5(file_path):
    return _cal_file_md5(file_path).hex()


def _cal_file_md5(file_path):
    with open(file_path, "rb") as file:
        file_hash = hashlib.md5()
        while True:
            chunk = file.read(8192)
            if chunk:
                file_hash.update(chunk)
            else:
                break
    return file_hash.digest()


def cal_dir_md5(dir_path):
    return cal_file_md5_sum(get_all_files(dir_path))


def cal_file_md5_sum(file_list):
    bytes_list = [_cal_file_md5(file) for file in file_list]
    byte = bytes_list[0]
    for b in bytes_list[1:]:
        byte = bytes(map(operator.xor, byte, b))
    return byte.hex()


def fullname(o):
    klass = o.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__
    return module + '.' + klass.__qualname__


def get_size(path: pathlib.Path):
    if path.is_file():
        return path.stat().st_size
    else:
        return sum([get_size(p) for p in path.iterdir()])


def remove(path: pathlib.Path):
    if path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)
