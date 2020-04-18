import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.flow as naf
import nltk

import pandas as pd
import logging

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw')


def train_eval_dataset(dataset: pd.DataFrame):
    flow = naf.Sometimes([naw.SynonymAug(lang="ita", aug_min=10),naw.RandomWordAug("swap"),naw.RandomWordAug("delete"),nac.OcrAug()])

    train_afert_exp=[]
    dev_after_exp=[]

    for idx, row in dataset.iterrows():
        logging.info("[{}/{}] {}".format(idx, len(dataset), row["question"]))
        new_text = [new for new in flow.augment(row["question"], n=20)]
        train_afert_exp.append({"label": row["question_id"], "text": row["question"]})
        th=int(len(new_text)*0.8)
        for text in new_text[:th]:
            train_afert_exp.append({"label": row["question_id"], "text": text})
        for text in new_text[th:]:
            dev_after_exp.append({"label": row["question_id"], "text": text})

    train=train_afert_exp
    dev=dev_after_exp

    train = pd.DataFrame(train)
    dev = pd.DataFrame(dev)

    return train, dev


if __name__ == "__main__":
    dataset = pd.read_csv("dataset.csv")
    train, dev = train_eval_dataset(dataset)
    train.to_csv("train.csv", index=False)
    dev.to_csv("dev.csv", index=False)
