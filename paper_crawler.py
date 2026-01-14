import requests
import sys
from loguru import logger
from datetime import datetime

class PaperCrawler:
    def __init__(self, date = None):
        self.date = date
        self.logger = logger

    def get_Huggingface_Daily_Paper_list(self):
        if self.date is None:
            date = datetime.now().strftime('%Y%m%d')
        else:
            date = self.date
        
        formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        
        url = f"https://huggingface.co/api/daily_papers?date={formatted_date}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            paper_list=[ f"https://arxiv.org/abs/{entity.get('paper').get('id')}" for entity in data]

            return paper_list

        except requests.RequestException as e:
            print(f"Error downloading daily papers: {e}")
            sys.exit(1)

def main():
    crawler = PaperCrawler()
    paper_list = crawler.get_Huggingface_Daily_Paper_list()
    print("paper_list: ", paper_list)

if __name__ == "__main__":
    main()