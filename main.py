import os
import sys
import pathlib

import numpy as np
import tensorflow as tf
from nn import build_model
from pp import load_discrete_dataset, train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
train_set, test_set = load_discrete_dataset(path=path)
# samples are time segments from the CWT per LE-22_036.. etc.
# (label, sample) , (dir_name, npy)


# ------------ create train,val, test ------------
print("Prepping Dataset")                            
train_set = train_set.shuffle(buffer_size = PREFETCH_BUFFER_SIZE,reshuffle_each_iteration=False)
train_set = train_set.batch(GLOBAL_BATCH_SIZE)
train_set = train_set.prefetch(buffer_size=PREFETCH_BUFFER_SIZE)



test_set = test_set.shuffle(buffer_size = PREFETCH_BUFFER_SIZE,reshuffle_each_iteration=False)
test_set = test_set.batch(GLOBAL_BATCH_SIZE)
test_set = test_set.prefetch(buffer_size=PREFETCH_BUFFER_SIZE)



# ------------ build model ------------
model = build_model()


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # Check optimizers
    loss=tf.keras.losses.BinaryCrossentropy(),  # Check losses
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.FalseNegatives(),
    ],  # Check metrics
)


# ---------- train model -----------

# multiprocessing?
model.fit(
    x=train_set,
    verbose="auto",
    callbacks=None,
    epochs = NUM_EPOCHS,
)


# ------------ viz results ------------
