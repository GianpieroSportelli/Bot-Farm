import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def create_model(labels,epochs=1,init_learning_rate=2e-5,dropout=0.1,max_seq_length = 64,fine_tune=True,is_training=True):
    bert_layer=get_bert_layer(fine_tune=fine_tune)

    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="segment_ids")
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    output=Dropout(dropout if is_training else 0.0)(pooled_output)
    logit=Dense(len(labels), activation='softmax')(output)
    model = Model(inputs=[input_word_ids,input_mask,segment_ids], outputs=logit)
    opt = Adam(learning_rate=init_learning_rate,decay=init_learning_rate/epochs)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def get_bert_layer(url="https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1",fine_tune=True):
    hub_module = hub.load(url)
    return hub.KerasLayer(hub_module,trainable=fine_tune)








