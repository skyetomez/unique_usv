import tensorflow as tf
from tensorflow import keras
import keras_tuner as hp
from keras.layers import Conv1D, Conv2D, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# Build model
NUM_FILTERS = 64  # how to pick num of filters
NUM_KERNELS = 8
STRIDES = 1
PADDING = "same"
SMALLEST_VEC_LEN = 21762020
CROP_DIMS = ((74, 68), (100, 100))  # px 74 off top and bottom, 102 off left and right
GOBAL_BATCH_SIZE = 64


def build_cnn1d_model(
    batch_size: int = GOBAL_BATCH_SIZE, seq_len: int = SMALLEST_VEC_LEN
):
    _specs_ = (10, seq_len)
    print(f"building 1D CNN with input shape {_specs_} ")
    _input_ = Input(shape=_specs_, batch_size=batch_size)
    _output_ = Conv1D(
        filters=NUM_FILTERS,
        kernel_size=NUM_KERNELS,
        strides=STRIDES,
        padding=PADDING,
    )(
        _input_
    )  # add normalization layer
    model = tf.keras.Model(inputs=_input_, outputs=_output_)
    print("model built!")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # Check optimizers
        loss=tf.keras.losses.CategoricalCrossentropy(),  # Check losses
        metrics=[
            tf.keras.metrics.CategoricalCrossentropy(),
            tf.keras.metrics.Accuracy(),
            tf.keras.metrics.MeanSquaredError(),
        ],  # Check metrics
    )
    print("model compiled")
    return model


def build_cnn2d_model(input_shape: tuple, num_classes: int):
    """CANNOT BE PASSED BATCH SIZE IF BEING LOADED WITH


    Args:
        input_shape (tuple): SHAPE OF IMAGE BASED ON tf.keras.utils.image_from_dataset
        num_classes (int): NUM OF DIRECTORIES CONTAINING FILES

    Returns:
        _type_: RETURNS MODEL
    """

    _specs_ = input_shape
    print(f"building 2D CNN of with input shape {_specs_} ")
    _input_ = Input(shape=_specs_)
    x = tf.keras.layers.Cropping2D(cropping=CROP_DIMS)(_input_)
    x = tf.keras.layers.Rescaling(scale=1.0 / 255)(x)
    x = tf.keras.layers.Resizing(height=512, width=512)(x)  # make multiple of 2
    x = Conv2D(
        filters=NUM_FILTERS,  # How to choose number of filters
        kernel_size=NUM_KERNELS,  # How to choose kernel size
        strides=STRIDES,  # How to choose strides
        padding=PADDING,
    )(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs=_input_, outputs=x)
    hp_learning_rate = 1e-4
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp_learning_rate
        ),  # Check optimizers
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),  # Check losses
        metrics=[
            tf.keras.metrics.CategoricalCrossentropy(from_logits=True),
            "accuracy",
        ],  # Check metrics
    )
    print("model compiled")
    return model


def get_checkpoints(model_save_path, monitor: str = "val_accuracy", best: bool = True):

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
        mode="auto",
        save_weights_only=False,
    )

    return [earlystop, chkpt]


if __name__ == "__main__":
    pass
    # model = build_cnn1d_model()

    # model.build(input_shape=(1000, 10, 2176202))
