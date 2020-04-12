import tensorflow as tf
import pandas as pd
from bot_farm.model import create_model
from bot_farm.label_util import save_labels, encode


def run(train: pd.DataFrame, dev: pd.DataFrame,label_path=None,model_path=None, epochs=5, batch_size=32):
    data = train.append(dev)

    labels = list(set(data.label))
    if not label_path:
        save_labels(labels)
    else:
        save_labels(labels,label_path)

    y_train = list(train['label'])
    x_train = list(train['text'])

    y_dev = list(dev['label'])
    x_dev = list(dev['text'])

    y_train = encode(y_train, labels)
    y_dev = encode(y_dev, labels)

    train_data = tf.constant(x_train, shape=[len(x_train)], dtype=tf.string)

    dev_data = tf.constant(x_dev, shape=[len(x_dev)], dtype=tf.string)

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
