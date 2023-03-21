import os
import sys
import pathlib

import numpy as np
import tensorflow as tf
from nn import build_cnn1d_model, get_checkpoints
from pp import load_discrete_dataset

from typing import Tuple
from numpy.typing import NDArray


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# os.environ["TF_ENABLE_ONEDNN_OPTS"] = 0

# ---------------- constants ---------------
GLOBAL_BATCH_SIZE = 64
PREFETCH_BUFFER_SIZE = 3299 + 1  # cmdline magic
NUM_EPOCHS = 10

# ---------------- load data set ----------------
root_dir = "/work/skylerthomas_umass_edu/current_projects/uniqueness"
root_dir = pathlib.Path(root_dir)

path = pathlib.Path(os.getenv("START"))

if path is None:
    path = root_dir / "discrete"


print("Loading Dataset")
train_set, test_set, val_set = load_discrete_dataset(path=path)
# samples are time segments from the CWT per LE-22_036.. etc.
# (label, sample) , (dir_name, npy)


# ------------ help functions ------------
def getdata(dataset)-> Tuple[NDArray,NDArray]:
    x, y = zip(*dataset)
    x,y = np.array(x), np.array(y)
    return x, y


# ------------ build model ------------
model = build_cnn1d_model()


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # Check optimizers
    loss=tf.keras.losses.BinaryCrossentropy(),  # Check losses
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.FalseNegatives(),
    ],  # Check metrics
)

model_callbacks = get_checkpoints("chkpts")

# ---------- train model -----------
x,y = getdata(train_set)
# multiprocessing?
history = model.fit(
    x=x,
    y=y,
    validation_data=getdata(val_set),
    verbose="auto",
    callbacks=model_callbacks,
    epochs=NUM_EPOCHS,
)


# ------------ viz results ------------
