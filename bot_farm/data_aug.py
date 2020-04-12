import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
import nltk

import pandas as pd
import logging

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw')


def train_eval_dataset(dataset: pd.DataFrame):
    flow = naf.Sometimes([naw.SynonymAug(lang="ita", aug_min=10), naw.RandomWordAug(action="swap"),
                          naw.AntonymAug(lang="ita", aug_min=5), naw.RandomWordAug()])

    id_set=list(set(dataset.question_id))
    examples = {id_question:[row for _,row in dataset[dataset.question_id == id_question].T.to_dict().items()] for id_question in id_set}
    train = []
    dev = []

    for id_question in id_set:
        sample = examples[id_question]
        if len(sample)>2:
            th=int(len(sample)*0.8)
            train.extend(sample[:th])
            dev.extend(sample[th:])
        else:
            train.append(sample[0])
            dev.append(sample[0])

    train_afert_exp=[]
    dev_after_exp=[]

    for idx, row in enumerate(train):
        logging.info("[{}/{}] {}".format(idx, len(train), row["question"]))
        new_text = [new for new in flow.augment(row["question"], n=20)]
        for text in new_text:
            train_afert_exp.append({"label": row["question_id"], "text": text})

    for idx, row in enumerate(dev):
        logging.info("[{}/{}] {}".format(idx, len(dev), row["question"]))
        new_text = [new for new in flow.augment(row["question"], n=20)]
        for text in new_text:
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
