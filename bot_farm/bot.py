from bot_farm.predictor import model_predictor
from bot_farm.data_aug import train_eval_dataset
from bot_farm.train_model import run
from telepot.delegate import pave_event_space, per_chat_id, create_open
from telepot.loop import MessageLoop

import pandas as pd

import random
import sys
import time
import logging
import telepot
import os
import argparse

global predictor,answers,dataset


parser = argparse.ArgumentParser(description='Bot farm')
parser.add_argument('--token', help='telegram token', required=True)
parser.add_argument('--dataset', help='dataset file path', required=True)
parser.add_argument('--answer', help='answer file path', required=True)
parser.add_argument("--base_dir",help="base directory",default="./")
parser.add_argument("--default_answer",help="the bot default answer",default="NO_ANSWER")
parser.add_argument("--welcome_message",help="the bot welcome message",default="Benvenuto, fammi una domanda.")
parser.add_argument("--threshold",help="the bot threshold",type=float,default=0.5)
parser.add_argument("--epochs",help="the bot epochs in training",type=int,default=5)
parser.add_argument("--language",help="the bot language",type=str,default="ita")
parser.add_argument("--expansion",help="the bot data aug expansion",type=int,default=5)
bot_args = parser.parse_args()

logs_file=os.path.join(bot_args.base_dir,"bot.log")

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
hdlr = logging.FileHandler(logs_file)
ch.setLevel(logging.INFO)
hdlr.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
hdlr.setFormatter(formatter)
root.addHandler(ch)
root.addHandler(hdlr)

class Bot(telepot.helper.ChatHandler):
    def __init__(self, *args, **kwargs):
        super(Bot, self).__init__(*args, **kwargs)

    def on_chat_message(self, msg):
        logging.info(msg)  # display del messaggio che manda l'utente
        content_type, chat_type, chat_id = telepot.glance(msg)
        if "text" in msg:
            command = msg['text']
            if '/start' in command:
                self.sender.sendMessage(bot_args.welcome_message)
            else:
                response = self.getAnswer(command,th=bot_args.threshold)
                if "text" in response:
                    alternatives=response["text"]
                    text=random.choice(alternatives)
                    tokens =text.split(" ")
                    text_message=[]
                    for t in tokens:
                        tentative=list(text_message)
                        tentative.append(t)
                        tentative_message=" ".join(tentative)
                        if len(tentative_message)>4096:
                            self.sender.sendMessage(" ".join(text_message))
                            text_message=[t]
                        else:
                            text_message=tentative
                    if len(text_message):
                        self.sender.sendMessage(" ".join(text_message))
                elif "photo" in response:
                    alternatives = response["photo"]
                    photo_url= random.choice(alternatives)
                    self.sender.sendPhoto(photo_url)

    def getAnswer(self,text, th=0.3):
        global predictor, answers
        label, conf = predictor.predict(text)
        logging.info("for question: {} prediction {} with confidence {}".format(text, label, conf))
        if conf > th:
            response=answers[label]
            return response
        else:
            return {"text":[bot_args.default_answer]} if not "NO_ANSWER" == bot_args.default_answer else {}


def setup():
    global predictor,answers,dataset
    model_path=os.path.join(bot_args.base_dir,"model")
    label_path=os.path.join(bot_args.base_dir,"labels.pkl")
    dataset = pd.read_csv(bot_args.dataset)
    answer = pd.read_csv(bot_args.answer)
    if not os.path.exists(model_path):
        train,dev = train_eval_dataset(dataset,lang=bot_args.language,expansion=bot_args.expansion)
        run(train,dev,label_path=label_path,model_path=model_path,epochs=bot_args.epochs)

    predictor=model_predictor(label_path=label_path,model_path=model_path)
    answers={row["question_id"]:{row["type"]:row["answer"].split("[SEP]") for _,row2 in answer.iterrows() if row2["question_id"]==row["question_id"]} for _,row in answer.iterrows()}
    return 0

setup()

token = bot_args.token
bot = telepot.DelegatorBot(token, [pave_event_space()(per_chat_id(), create_open, Bot, timeout=360),
])

MessageLoop(
    bot).run_as_thread();  # in caso di chat, esegui il metodo on_chat_message. Ci sono altre modalità che vedremo in futuro
logging.info('Listening ...')

while 1:
    time.sleep(10)