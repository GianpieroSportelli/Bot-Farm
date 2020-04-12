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
        job_elems = soup.find_all('hr')
        result = []
        i = 0
        for job_elem in job_elems:
            print(job_elem)
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

if __name__ == "__main__":
    # main_page = "http://www.salute.gov.it/portale/nuovocoronavirus/dettaglioFaqNuovoCoronavirus.jsp?lingua=italiano&id=228#11"
    # dataset=COVID_Scraper(main_page).createDataset()
    # dataset.to_csv("dataset.csv")
    base_url="https://telegram.org/faq/it"
    Telegram_Scraper(base_url).createDataset()

