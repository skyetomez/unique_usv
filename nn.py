import tensorflow as tf
from tensorflow import keras

from keras.layers import Conv1D, Conv2D, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# Build model
NUM_FILTERS = 1  # how to pick num of filters
NUM_KERNELS = 1
STRIDES = 1
PADDING = "same"
SMALLEST_VEC_LEN = 21762020

CROP_DIMS = ((74, 68), (100, 100))  # px 74 off top and bottom, 102 off left and right

GOBAL_BATCH_SIZE = 64


def build_cnn1d_model(
    batch_size: int = GOBAL_BATCH_SIZE, seq_len: int = SMALLEST_VEC_LEN
):
    _specs_ = (10, seq_len)
    print(f"building 1D CNN of as {_specs_} ")
    _input_ = Input(shape=_specs_, batch_size=batch_size)
    _output_ = Conv1D(
        filters=NUM_FILTERS,
        kernel_size=NUM_KERNELS,
        strides=STRIDES,
        padding=PADDING,
    )(_input_)
    model = tf.keras.Model(inputs=_input_, outputs=_output_)
    print("model built!")
    return model


def build_cnn2d_model(batch_size: int = GOBAL_BATCH_SIZE, seq_len: int = 1):
    _specs_ = (10, seq_len)
    print(f"building 2D CNN of as {_specs_} ")
    _input_ = Input(shape=_specs_, batch_size=batch_size)
    x = tf.keras.layers.Cropping2D(cropping=CROP_DIMS)(_input_)
    x = tf.keras.layers.Rescaling(scale=1.0 / 255)(x)
    # x = tf.keras.layes.Resizing() may not be necessary
    _output_ = Conv2D(
        filters=NUM_FILTERS,
        kernel_size=NUM_KERNELS,
        strides=STRIDES,
        padding=PADDING,
    )(x)
    model = tf.keras.Model(inputs=_input_, outputs=_output_)
    print("model built!")
    return model


def get_checkpoints(model_save_path, monitor: str = "val_loss", best: bool = True):

    earlystop = EarlyStopping(
        monitor=monitor,
        min_delta=0,
        patience=13,
        mode="auto",
        verbose=0,
        restore_best_weights=best,
    )

    chkpt = ModelCheckpoint(
        filepath=model_save_path,
        monitor=monitor,
        save_best_only=best,
        save_weights_only=False,
    )

    tnsbrd = TensorBoard()

    return [earlystop, chkpt, tnsbrd]


if __name__ == "__main__":

    model = build_cnn1d_model()

    model.build(input_shape=(1000, 10, 2176202))
