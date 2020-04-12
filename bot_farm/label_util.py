import pickle
import tensorflow as tf


def save_labels(labels, file_path="labels.pkl"):
    with open(file_path, "wb") as lf:
        pickle.dump(labels, lf)


def load_labels(file_path="labels.pkl"):
    with open(file_path, "rb") as lf:
        return pickle.load(lf)


def encode(y, labels):
    return tf.constant([[float(yi == l) for l in labels] for yi in y], shape=(len(y), len(labels)), dtype=tf.float32)

def decode(pred, labels):
    return [labels[int(p)] for p in pred]
