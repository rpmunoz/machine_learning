from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from models.resnext_block import build_ResNeXt_block


class ResNeXt(keras.Model):
    
    @staticmethod
    def build(width, height, depth, classes, repeat_num_list, cardinality,
              reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset="cifar",
              training=None, mask=None):
        
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
        
        x = keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")(inputs)
        
        x = keras.layers.BatchNormalization()(x, training=training)
        x = layers.Activation("relu")(x)
        x = keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")(x)

        x = build_ResNeXt_block(filters=128,
                                strides=1,
                                groups=cardinality,
                                repeat_num=repeat_num_list[0])(x, training=training)
        
        x = build_ResNeXt_block(filters=256,
                                strides=2,
                                groups=cardinality,
                                repeat_num=repeat_num_list[1])(x, training=training)
        
        x = build_ResNeXt_block(filters=512,
                                strides=2,
                                groups=cardinality,
                                repeat_num=repeat_num_list[2])(x, training=training)
        
        x = build_ResNeXt_block(filters=1024,
                                strides=2,
                                groups=cardinality,
                                repeat_num=repeat_num_list[3])(x, training=training)

        x = keras.layers.GlobalAveragePooling2D()(x)

        # softmax classifier
        x = layers.Dense(classes, kernel_regularizer=keras.regularizers.l2(reg))(x)
        x = layers.Activation("softmax")(x)

        # create the model
        model = keras.Model(inputs, x, name="ResNeXt")

        # return the constructed network architecture
        return model


def ResNeXt50():
    return ResNeXt(repeat_num_list=[3, 4, 6, 3],
                   cardinality=32)


def ResNeXt101():
    return ResNeXt(repeat_num_list=[3, 4, 23, 3],
                   cardinality=32, num_classes=100)