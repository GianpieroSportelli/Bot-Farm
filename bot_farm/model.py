import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow_text


def create_model(labels,epochs=1,init_learning_rate=1e-3,dropout=0.1,hidden_unit=256,url="https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3",fine_tune=True,is_training=True):
    hub_module=hub.load(url)
    layers=[
        hub.KerasLayer(hub_module,output_shape=(512,),input_shape=(),trainable=fine_tune,dtype=tf.string),
        Dropout(dropout if is_training else 0.0),
        Dense(hidden_unit, activation='relu'),
        Dense(len(labels), activation='softmax')
    ]
    model = Sequential(layers)
    opt = Adam(learning_rate=init_learning_rate,decay=init_learning_rate/epochs)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model







