import tensorflow as tf

from keras.layers import Conv1D, Input

# Build model
NUM_FILTERS = 1  # how to pick num of filters
NUM_KERNELS = 1
STRIDES = 1
PADDING = "same"
SMALLEST_VEC_LEN = 21762020

CROP_DIM = ((74, 68), (100, 100))

GOBAL_BATCH_SIZE = 64


def build_model():
    _specs_ = (10, SMALLEST_VEC_LEN)
    print(f"building 1D CNN of as {_specs_} ")
    _input_ = Input(shape=_specs_, batch_size=GOBAL_BATCH_SIZE)
    x = tf.keras.layers.Cropping2D(cropping=CROP_DIM)(_input_)
    x = tf.keras.layers.Rescaling(scale=1.0 / 255)(x)
    _output_ = Conv1D(
        filters=NUM_FILTERS,
        kernel_size=NUM_KERNELS,
        strides=STRIDES,
        padding=PADDING,
    )(x)
    model = tf.keras.Model(inputs=_input_, outputs=_output_)
    print("model built!")
    return model


if __name__ == "__main__":

    model = build_model()

    model.build(input_shape=(1000, 10, 2176202))
