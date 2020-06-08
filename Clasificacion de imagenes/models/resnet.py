# import the necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False,
        reg=0.0001, bnEps=2e-5, bnMom=0.9):
        # the shortcut branch of the ResNet module should be
        # initialize as the input (identity) data
        shortcut = data

        # the first block of the ResNet module are the 1x1 CONVs
        bn1 = layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
            momentum=bnMom)(data)
        act1 = layers.Activation("relu")(bn1)
        conv1 = layers.Conv2D(int(K * 0.25), (1, 1), use_bias=False,
            kernel_regularizer=keras.regularizers.l2(reg))(act1)

        # the second block of the ResNet module are the 3x3 CONVs
        bn2 = layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
            momentum=bnMom)(conv1)
        act2 = layers.Activation("relu")(bn2)
        conv2 = layers.Conv2D(int(K * 0.25), (3, 3), strides=stride,
            padding="same", use_bias=False,
            kernel_regularizer=keras.regularizers.l2(reg))(act2)

        # the third block of the ResNet module is another set of 1x1
        # CONVs
        bn3 = layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
            momentum=bnMom)(conv2)
        act3 = layers.Activation("relu")(bn3)
        conv3 = layers.Conv2D(K, (1, 1), use_bias=False,
            kernel_regularizer=keras.regularizers.l2(reg))(act3)

        # if we are to reduce the spatial size, apply a CONV layer to
        # the shortcut
        if red:
            shortcut = layers.Conv2D(K, (1, 1), strides=stride,
                use_bias=False, kernel_regularizer=keras.regularizers.l2(reg))(act1)

        # add together the shortcut and the final CONV
        x = layers.Add()([conv3, shortcut])

        # return the addition as the output of the ResNet module
        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters,
        reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset="cifar"):
        # initialize the input shape to be "channels last" and the
        # channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # set the input and apply a single BN + CONV layer
        inputs = layers.Input(shape=inputShape)
        x = layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
            momentum=bnMom)(inputs)
        x = layers.Conv2D(filters[0], (3, 3), use_bias=False,
            padding="same", kernel_regularizer=keras.regularizers.l2(reg))(x)

        # loop over the number of stages
        for i in range(0, len(stages)):
            # initialize the stride, then apply a residual module
            # used to reduce the spatial size of the input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride,
                chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a ResNet module
                x = ResNet.residual_module(x, filters[i + 1],
                    (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)

        # apply BN => ACT => POOL
        x = layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
            momentum=bnMom)(x)
        x = layers.Activation("relu")(x)
        x = layers.AveragePooling2D((8, 8))(x)

        # softmax classifier
        x = layers.Flatten()(x)
        x = layers.Dense(classes, kernel_regularizer=keras.regularizers.l2(reg))(x)
        x = layers.Activation("softmax")(x)

        # create the model
        model = keras.Model(inputs, x, name="resnet")

        # return the constructed network architecture
        return model