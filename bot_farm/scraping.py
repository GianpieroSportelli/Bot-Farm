import requests
from bs4 import BeautifulSoup,Tag
import pandas as pd


class AbstractScraper:
    def __init__(self,url):
        self.url=url

    def createDataset(self):
        raise NotImplemented


class COVID_Scraper(AbstractScraper):
    def __init__(self,url):
        super().__init__(url)

    def __convert(self,tag):
        if type(tag) == Tag:
            return " ".join([self.__convert(e) for e in tag])
        else:
            return tag.strip()

    def createDataset(self):
        page = requests.get(self.url)
        soup = BeautifulSoup(page.content, 'html.parser')
        job_elems = soup.find_all('dl')
        result=[]
        i=0
        for job_elem in job_elems:
            i+=1
            for child in job_elem.find_all("dt"):
                tags=child.find_all("strong")
                id_question=tags[0].string.strip()
                question = tags[1].string.strip()
                answer=list(job_elem.find_all("dd"))[int(id_question.replace(".",""))-1]
                answer_text=self.__convert(answer)
                result.append({"question_id":"{}.{}".format(i,id_question),"question":question,"answer":answer_text})

        return pd.DataFrame(result)

class Telegram_Scraper(AbstractScraper):
    def __init__(self, url):
        super().__init__(url)

    def __convert(self, tag):
        if type(tag) == Tag:
            return " ".join([self.__convert(e) for e in tag])
        else:
            return tag.strip()

    def createDataset(self):
        page = requests.get(self.url)
        print(page.text)
        soup = BeautifulSoup(page.content, 'html.parser')
        job_elems = soup.find_all('h4')
        result = []
        i = 0
        start_question=False
        for job_elem in soup:
            if type(job_elem) == Tag:
                if job_elem.name == "h3":
                    print("START section {}".format(job_elem.text))
                    start_question=True
                if job_elem.name == "h4":
                    print("     START question {}".format(job_elem.text))
                if job_elem.name == "p" and start_question:
                    print("         START answer {}".format(job_elem.text))

            # i += 1
            # for child in job_elem.find_all("dt"):
            #     tags = child.find_all("strong")
            #     id_question = tags[0].string.strip()
            #     question = tags[1].string.strip()
            #     answer = list(job_elem.find_all("dd"))[int(id_question.replace(".", "")) - 1]
            #     answer_text = self.__convert(answer)
            #     result.append(
            #         {"question_id": "{}.{}".format(i, id_question), "question": question, "answer": answer_text})

        # return pd.DataFrame(result)
class Film_Scraper(AbstractScraper):
    def __init__(self):
        super().__init__(url="https://www.cineblog.it/post/36923/citazioni-cinematografiche-quale-vi-tatuereste")

    def createDataset(self):
        page = requests.get(self.url)
        result=[]
        # print(page.text)
        soup = BeautifulSoup(page.content, 'html.parser')
        job_elems = soup.find_all('p')

        for elem in job_elems:
            text = elem.text
            try:
                num=int(text[0])
                for sub_elem in text.split("\n"):
                    i=0
                    while not sub_elem[i]==".":
                        i+=1

                    good=sub_elem[i+1:]
                    good=good.strip()
                    assert len(good.split("-"))==2
                    splitted=good.split("-")
                    example=splitted[0].strip()
                    film=splitted[1].strip()
                    result.append({"question":example,"question_id":film})
            except:
                continue

        return pd.DataFrame(result)

if __name__ == "__main__":
    dataset=Film_Scraper().createDataset()
    dataset.to_csv("/home/gianpiero/data/film_bot/new_dataset.csv",index=False)
    answer=dataset[["question_id"]]
    answer=answer.drop_duplicates()
    answer["type"]=answer.question_id.apply(lambda x: "photo")
    answer["answer"] = answer.question_id.apply(lambda x: "LINK")
    answer.to_csv("/home/gianpiero/data/film_bot/new_answer.csv", index=False)
