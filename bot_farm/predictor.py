import tensorflow as tf
from bot_farm.label_util import load_labels, decode


class model_predictor:
    def __init__(self, label_path: str = "labels.pkl", model_path: str = "model"):
        self.labels = load_labels(label_path)
        self.model = tf.saved_model.load(model_path)

    def predict(self, text: str):
        x = tf.constant([text], shape=(1), dtype=tf.string)
        y = self.model(x)
        pred = tf.reshape(tf.argmax(y, axis=1), [1]).numpy()
        confidence = tf.reshape(tf.reduce_max(y, axis=1), [1]).numpy().tolist()[0]
        output = decode(pred, self.labels)[0]
        return output, confidence

if __name__ =="__main__":
    import pandas as pd
    from datetime import datetime


    dev = pd.read_csv('dev.csv')
    acc = 0.0
    tot_time=None
    predictor = model_predictor()
    for idx, row in dev.iterrows():
        start=datetime.now()
        pred,conf = predictor.predict(row["text"])
        end = datetime.now()
        print("[{}/{}] {}".format(idx,len(dev),(end-start)))
        acc += float(row["label"] == pred)

    print("Acc: {}".format(acc / len(dev)))
