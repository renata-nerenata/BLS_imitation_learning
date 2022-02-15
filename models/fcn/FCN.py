import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
import pandas as pd
from keras import optimizers


def FCN2(nClasses, input_height=224, input_width=224):
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    IMAGE_ORDERING = "channels_last"

    input = get_data()

    ## Block 1
    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        name="block1_conv1",
        data_format=IMAGE_ORDERING,
    )(input)
    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        name="block1_conv2",
        data_format=IMAGE_ORDERING,
    )(x)
    x = MaxPooling2D(
        (2, 2), strides=(2, 2), name="block1_pool", data_format=IMAGE_ORDERING
    )(x)
    f1 = x

    # Block 2
    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
        name="block2_conv1",
        data_format=IMAGE_ORDERING,
    )(x)
    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
        name="block2_conv2",
        data_format=IMAGE_ORDERING,
    )(x)
    x = MaxPooling2D(
        (2, 2), strides=(2, 2), name="block2_pool", data_format=IMAGE_ORDERING
    )(x)
    f2 = x

    # Block 3
    x = Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same",
        name="block3_conv1",
        data_format=IMAGE_ORDERING,
    )(x)
    x = Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same",
        name="block3_conv2",
        data_format=IMAGE_ORDERING,
    )(x)
    x = Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same",
        name="block3_conv3",
        data_format=IMAGE_ORDERING,
    )(x)
    x = MaxPooling2D(
        (2, 2), strides=(2, 2), name="block3_pool", data_format=IMAGE_ORDERING
    )(x)
    pool3 = x

    # Block 4
    x = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        name="block4_conv1",
        data_format=IMAGE_ORDERING,
    )(x)
    x = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        name="block4_conv2",
        data_format=IMAGE_ORDERING,
    )(x)
    x = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        name="block4_conv3",
        data_format=IMAGE_ORDERING,
    )(x)
    pool4 = MaxPooling2D(
        (2, 2), strides=(2, 2), name="block4_pool", data_format=IMAGE_ORDERING
    )(
        x
    )  ## (None, 14, 14, 512)

    # Block 5
    x = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        name="block5_conv1",
        data_format=IMAGE_ORDERING,
    )(pool4)
    x = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        name="block5_conv2",
        data_format=IMAGE_ORDERING,
    )(x)
    x = Conv2D(
        512,
        (3, 3),
        activation="relu",
        padding="same",
        name="block5_conv3",
        data_format=IMAGE_ORDERING,
    )(x)
    pool5 = MaxPooling2D(
        (2, 2), strides=(2, 2), name="block5_pool", data_format=IMAGE_ORDERING
    )(x)
    vgg = Model(input, pool5)
    vgg.load_weights(
        VGG_Weights_path
    )  ## loading VGG weights for the encoder parts of FCN8

    n = 4096
    o = (
        Conv2D(
            n,
            (7, 7),
            activation="relu",
            padding="same",
            name="conv6",
            data_format=IMAGE_ORDERING,
        )
    )(pool5)
    conv7 = (
        Conv2D(
            n,
            (1, 1),
            activation="relu",
            padding="same",
            name="conv7",
            data_format=IMAGE_ORDERING,
        )
    )(o)

    conv7_4 = Conv2DTranspose(
        nClasses,
        kernel_size=(4, 4),
        strides=(4, 4),
        use_bias=False,
        data_format=IMAGE_ORDERING,
    )(conv7)
    pool411 = (
        Conv2D(
            nClasses,
            (1, 1),
            activation="relu",
            padding="same",
            name="pool4_11",
            data_format=IMAGE_ORDERING,
        )
    )(pool4)
    pool411_2 = (
        Conv2DTranspose(
            nClasses,
            kernel_size=(2, 2),
            strides=(2, 2),
            use_bias=False,
            data_format=IMAGE_ORDERING,
        )
    )(pool411)

    pool311 = (
        Conv2D(
            nClasses,
            (1, 1),
            activation="relu",
            padding="same",
            name="pool3_11",
            data_format=IMAGE_ORDERING,
        )
    )(pool3)

    o = Add(name="add")([pool411_2, pool311, conv7_4])
    o = Conv2DTranspose(
        nClasses,
        kernel_size=(8, 8),
        strides=(8, 8),
        use_bias=False,
        data_format=IMAGE_ORDERING,
    )(o)
    o = (Activation("softmax"))(o)

    model = Model(input, o)

    return model


if __name__ == "__main__":
    model = FCN2(nClasses=n_classes, input_height=224, input_width=224)
    # model.summary()

    sgd = optimizers.SGD(lr=1e-2, decay=5 ** (-4), momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=50,
        epochs=150,
        verbose=2,
    )
