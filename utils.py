import os
import glob
import requests
from bs4 import BeautifulSoup
import random 
import time 

class web_scraper():

    def __init__(self, url, retries = 3):
        self.url = url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        self.retries = retries
        for i in range(self.retries + 1):
            time_interval = random.randint(1, 5)
            try:
                self.response = requests.get(self.url, headers = self.headers)
                self.response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if i < self.retries:
                    print(f"Retrying {i+1} time after {time_interval} seconds")
                    time.sleep(time_interval)
                    continue
                else:
                    print(f"Failed to get response after {i+1} retries")
                    raise e
        self.soup = BeautifulSoup(self.response.content, 'html.parser')