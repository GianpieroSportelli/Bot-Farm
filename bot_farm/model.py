import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from bot_farm.custom_layer import Bert
import tensorflow_model_optimization as tfmot


def create_model(labels,train_steps,epochs=1,init_learning_rate=2e-5,dropout=0.1,max_seq_length = 64,fine_tune=True,is_training=True):
    quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer

    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="segment_ids")
    output = Bert(fine_tuning=fine_tune)([input_word_ids, input_mask, segment_ids])
    output=Dropout(dropout if is_training else 0.0)(output)
    logit=Dense(len(labels), activation='softmax')(output)
    model = Model(inputs=[input_word_ids,input_mask,segment_ids], outputs=logit)
    # pruning_schedule = tfmot.sparsity.keras.ConstantSparsity(target_sparsity=0.4,begin_step=0,frequency=10)
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,final_sparsity=0.1,begin_step=0,end_step=train_steps,frequency=int(train_steps/epochs))
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)
    opt = Adam(learning_rate=init_learning_rate,decay=init_learning_rate/epochs)
    model_for_pruning.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model_for_pruning







