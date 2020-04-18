import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer
from tensorflow.keras.layers import Layer
from tensorflow_model_optimization.python.core.sparsity.keras.prunable_layer import PrunableLayer

url="https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1"
fine_tune=True
hub_module = hub.load(url)
bert_layer=hub.KerasLayer(hub_module,trainable=fine_tune)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = FullTokenizer(vocab_file, do_lower_case)
del bert_layer,vocab_file,do_lower_case

class Bert(PrunableLayer,Layer):
    def __init__(self,input_shape=(1),output_representation='cls_output',fine_tuning=True, url = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1",**kwargs):
        super(Bert, self).__init__(**kwargs)
        self.output_representation=output_representation
        self.bert = hub.KerasLayer(url, trainable=fine_tuning)
        super(Bert, self).build(input_shape)
        self.set_weights(self.bert.get_weights())

    def get_prunable_weights(self):
        return self.bert.variables[:-1]

    def call(self, x):
        outputs =  self.bert(x)[1
        if self.output_representation in ['sequence_output', 'cls_output'] else 0]

        if self.output_representation == 'cls_output':
            return tf.squeeze(outputs[:, 0:1, :], axis=1)
        else:
            return outputs

    def compute_output_shape(self, input_shape):
        if self.output_representation in ['pooled_output', 'cls_output']:
            return (None, 768)
        else:
            return (None, 512, 768)

    def get_config(self):
        return self.bert.get_config()