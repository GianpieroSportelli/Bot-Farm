import tensorflow as tf
import pandas as pd
from bot_farm.model import create_model
from bot_farm.label_util import save_labels, encode
from bot_farm.model import get_bert_layer
from bert.tokenization import FullTokenizer
import logging

logger =tf.get_logger()
logger.setLevel(logging.INFO)
logger.propagate=False

global tokenizer

bert_layer = get_bert_layer(fine_tune=False)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)


def create_bert_features(texts, max_seq_length):
    input_ids = []
    input_mask = []
    segment_ids = []

    for idx,text in enumerate(texts):
        tokens = ["[CLS]"]
        tokens.extend(tokenizer.tokenize(text))
        if len(tokens) > max_seq_length -1:
            tokens = tokens[:max_seq_length - 1]
        tokens.append("[SEP]")
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_val = [0] * max_seq_length
        for i, ids in enumerate(token_ids):
            input_val[i] = ids
        segment_val=[0] * max_seq_length
        mask_val=[int(i < len(token_ids)) for i in range(max_seq_length)]
        if(idx<10):
            logger.info("example idx: {}".format(idx))
            logger.info("tokens: {}".format(tokenizer.convert_ids_to_tokens(input_val[:len(token_ids)])))
            logger.info("mask: {}".format(mask_val))
            logger.info("segment: {}".format(segment_val))

        segment_ids.append(segment_val)
        input_mask.append(mask_val)
        input_ids.append(input_val)

    return input_ids, input_mask, segment_ids


def run(train: pd.DataFrame, dev: pd.DataFrame, label_path=None, model_path=None, epochs=5, batch_size=16,
        max_seq_length=64):
    data = train.append(dev)

    labels = list(set(data.label))
    if not label_path:
        save_labels(labels)
    else:
        save_labels(labels, label_path)

    y_train = list(train['label'])
    x_train = list(train['text'])
    train_input_ids, train_input_mask, train_segment_ids = create_bert_features(x_train, max_seq_length)

    y_dev = list(dev['label'])
    x_dev = list(dev['text'])
    dev_input_ids, dev_input_mask, dev_segment_ids = create_bert_features(x_dev, max_seq_length)

    y_train = encode(y_train, labels)
    y_dev = encode(y_dev, labels)

    train_data = (tf.constant(train_input_ids, shape=[len(x_train), max_seq_length], dtype=tf.float32),
                  tf.constant(train_input_mask, shape=[len(x_train), max_seq_length], dtype=tf.float32),
                  tf.constant(train_segment_ids, shape=[len(x_train), max_seq_length], dtype=tf.float32))

    dev_data = (tf.constant(dev_input_ids, shape=[len(x_dev), max_seq_length], dtype=tf.float32),
                  tf.constant(dev_input_mask, shape=[len(x_dev), max_seq_length], dtype=tf.float32),
                  tf.constant(dev_segment_ids, shape=[len(x_dev), max_seq_length], dtype=tf.float32))

    model = create_model(labels, epochs, fine_tune=True, is_training=True)

    history = model.fit(train_data, y_train, epochs=epochs, batch_size=batch_size, validation_data=(dev_data, y_dev))
    if not model_path:
        model.save("model")
    else:
        model.save(model_path)


if __name__ == "__main__":
    train = pd.read_csv('train.csv')
    train = train.sample(frac=1.0)
    dev = pd.read_csv('dev.csv')

    epochs = 5
    batch_size = 32
    run(train, dev)
