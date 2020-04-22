import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model
from bot_farm.custom_layer import Bert
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt
from bot_farm.optimization import WarmUp,create_optimizer

# ##https://www.tensorflow.org/lite/guide/ops_select
# converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                        tf.lite.OpsSet.SELECT_TF_OPS]
# tflite_model = converter.convert()
# open(os.path.join(model_path,"model.tflite"), "wb").write(tflite_model)

# https://www.dlology.com/blog/how-to-compress-your-keras-model-x5-smaller-with-tensorflow-model-optimization/
# pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
#     initial_sparsity=0.0, final_sparsity=0.6,
#     begin_step=0, end_step=4000)
#
# model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)
#
# model_for_pruning.fit(train_data, y_train, epochs=epochs, batch_size=batch_size, validation_data=(dev_data, y_dev))
#
# model_for_pruning.save(model_path,include_optimizer=False)


def build_and_train_model(labels, train_steps, train_data, y_train, dev_data, y_dev, batch_size=32, epochs=1,
                          init_learning_rate=2e-5, dropout=0.1, max_seq_length=64, fine_tune=True, is_training=True,compress=True):
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                        name="segment_ids")

    output = Bert(fine_tuning=fine_tune)([input_word_ids, input_mask, segment_ids])
    output = Dropout(dropout if is_training else 0.0)(output)
    logit = Dense(len(labels), activation='softmax')(output)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=logit)
    # learning_rate = CustomSchedule(init_lr=init_learning_rate, train_steps=train_steps,
    #                                warmup_steps=int(train_steps * 0.10))

    # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999,
    #                                      epsilon=1e-6, decay=0.01)
    #
    optimizer=create_optimizer(init_learning_rate,train_steps,int(train_steps * 0.10))
    if compress:
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0, final_sparsity=0.1, begin_step=0,
                                                                end_step=train_steps, frequency=int(train_steps / epochs))
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)

        model_for_pruning.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model_for_pruning.summary()
        history = model_for_pruning.fit(train_data, y_train, epochs=epochs, batch_size=batch_size,
                                        validation_data=(dev_data, y_dev),
                                        callbacks=[tfmot.sparsity.keras.UpdatePruningStep(),
                                                   tfmot.sparsity.keras.PruningSummaries("logdir")])
        final_model = tfmot.sparsity.keras.strip_pruning(model)
        final_model.summary()

        return final_model
    else:
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.summary()
        history = model.fit(train_data, y_train, epochs=epochs, batch_size=batch_size,
                                        validation_data=(dev_data, y_dev),
                                        callbacks=[tf.keras.callbacks.TensorBoard("logdir")])
        model.save("model.h5")
        return model


if __name__=="__main__":
    scheduler=WarmUp(0.1,100,10)
    plt.plot(scheduler(tf.range(100, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.show()