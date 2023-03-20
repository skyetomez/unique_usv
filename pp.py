import pathlib
import os

import tensorflow as tf
import numpy as np

from numpy.typing import NDArray
from typing import Tuple, List

root_dir = "/work/skylerthomas_umass_edu/current_projects/uniqueness"
root_dir = pathlib.Path(root_dir)

cont_dir = root_dir / "continuous"
disc_dir = root_dir / "discrete"
# test_dir = cont_dir / "le_22_028"

SMALLEST_VEC_LEN = 2176202


def _gen_paths_from_dir_(path) -> Tuple[List[int], List[str]]:  # type: ignore
    trans = {
        "le_22_036": 0,
        "le_22_036_day2": 1,
        "le_22_036_day4": 2,
        "le_22_037_day2": 3,
        "le_22_037_day3": 4,
        "le_22_037_day4": 5,
        "le_22_039_day4": 6,
        "le_22_040_day2": 7,
    }
    label = trans[path.name]  # path.name
    print(path.name)
    _path = cont_dir.joinpath(path)
    os.chdir(_path)

    paths = [str(n.absolute()) for n in _path.glob("*.npy")]
    labels = [label] * len(list(paths))

    return labels, paths


def _get_dataset(path):
    data = []
    labels = []
    for p in path.iterdir():
        l, d = _gen_paths_from_dir_(p)
        data.append(d)
        labels.append(l)

    def _categorical_(y):
        tmp = [l for n in labels for l in n]
        tmp = tf.keras.utils.to_categorical(tmp)
        print(f"labels array is {tmp.shape}")
        return tmp.astype(np.float32)

    def _load_(x):
        def _resize_(x):
            arr = np.resize(x, 2176202)  # 2176202
            return arr

        tmp = [l for n in data for l in n]
        arr = list(
            map(
                lambda path: np.concatenate(
                    np.load(path, allow_pickle=True), dtype=np.float32
                ),
                tmp,
            )
        )
        arr = np.array(
            list(map(lambda matrix: _resize_(matrix), arr)), dtype=np.float32
        )
        print(f"data array has shape {arr.shape}")
        return arr

    data = _load_(data)
    labels = _categorical_(labels)

    print(f"total length of the data array is {len(data)}")
    print(f"total length of the labels array is {len(labels)}")
    return data, labels


def _shuffle_(data: NDArray, labels: NDArray) -> List[Tuple[NDArray, NDArray]]:
    from random import sample

    tmp = list(zip(data, labels))

    for _ in range(11):
        tmp = sample(tmp, len(tmp))

    return tmp


def _train_test_split_(ds, percent: float = 0.65, val: bool = False):
    split = int(len(ds) * percent)
    train = ds[:split]
    test = ds[split:]
    return train, test


def load_discrete_dataset(path):
    # Load the numpy files
    print(f"Loading dataset from path {path}")
    data, labels = _get_dataset(path)
    print(f"shuffling data set")
    tmp = _shuffle_(data, labels)
    print(f"creating splits")
    train, test = _train_test_split_(tmp, percent=0.65)
    print("returning test and train set ")
    return tf.data.Dataset.from_tensor_slices(train, test)


# def train_test_split(ds, percent: float = 0.5, val: bool = False):
#    train_set, test_set = tf.keras.utils.split_dataset(ds, right_size=percent)
#    if not val:
#        val_per = 1 - percent // 2
#        test_set, val_set = tf.keras.utils.split_dataset(test_set, left_size=val_per)
#        print(
#            f"making train {percent}%, test {val_per}%  and val {val_per}% split sets"
#        )
#        return train_set, test_set, val_set
#    print(f"making train {percent}% and test splits {1-percent}%")
#    return train_set, test_set
