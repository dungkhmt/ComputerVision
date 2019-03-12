from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate, Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard 
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from skimage.transform import resize
import tensorflow as tf
import numpy as np
#Create embedding

inception = InceptionResNetV2(weights=None, include_top=True)
inception.load_weights('./trained_model/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
inception.graph = tf.get_default_graph()

def incpetion_predict(image):
    image = image.reshape((1, 256, 256, 3)).astype(float)
    image /= 255
    with inception.graph.as_default():
        emb = inception.predict(image)
        print(emb.shape)
    return emb

def create_inception_embedding(grayscaled_rgb):
    #Load weights
    inception = InceptionResNetV2(weights=None, include_top=True)
    inception.load_weights('./trained_model/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
    inception.graph = tf.get_default_graph()
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed


class ColorizationModel():
    def __init__(self):
        pass

    def define_model(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(128,128, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
        model.add(UpSampling2D((2, 2)))
        # Finish model
        model.compile(optimizer='rmsprop', loss='mse')
        return model

    def transfer_model(self):
        
        #Load weights
        embed_input = Input(shape=(1000,))

        #Encoder
        encoder_input = Input(shape=(200, 200, 1,))
        encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
        encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
        encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
        encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

        #Fusion
        fusion_output = RepeatVector(25 * 25)(embed_input) 
        fusion_output = Reshape(([25, 25, 1000]))(fusion_output)
        fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
        fusion_output = Conv2D(200, (1, 1), activation='relu', padding='same')(fusion_output) 

        #Decoder
        decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        # embed_input = create_inception_embedding(encoder_input)
        model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return model

