#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re
import json
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import socket
import datetime
from dateutil import parser
from time import sleep

import requests
import investpy as inv
from tqdm.auto import tqdm

import sys

sns.set(style='darkgrid', font_scale=1.3, palette='Set2')

with open('timezones.json') as timezones:
    tzinfos = json.loads(timezones.read())
    
    

INV_URL = 'https://www.investing.com'
COLUMNS = ['ticker', 'source', 'title', 'body', 'date', 'link']


def get_investing_link_by_ticker(ticker):
    r = Request(f'{INV_URL}/search/?q={ticker}', headers={"User-Agent": "Mozilla/5.0"})
    c = urlopen(r).read()
    soup = BeautifulSoup(c, "html.parser")
    return soup.findAll('a', attrs={'class': 'js-inner-all-results-quote-item row'})[0].attrs['href']


def download_news_from_investing(link, page, ticker):
    df = []
    
    try:
        r = Request(INV_URL + link + f'-news/{page}', headers={"User-Agent": "Mozilla/5.0"})
        c = urlopen(r).read()
        soup = BeautifulSoup(c, "html.parser")
    
        articles_section = soup.findAll('section', attrs={'id': 'leftColumn'})[0]
        articles_list = articles_section.findAll('div', attrs={'class': 'mediumTitle1'})[0]
    except Exception:
        print(f'Error in load page {page} by link {INV_URL + link}-news/{page}')
        return pd.DataFrame(df, columns=COLUMNS)

    for article in articles_list:
        if article == '\n':
            continue
            
        try:
            article_link = article.findAll('a')[0].attrs['href']
        except Exception:
            print(f"Can't parse {article}.")
            continue
        
        try:
            if article_link.startswith('/news/'):
                sleep(0.05)
                r_art = Request(INV_URL + article_link, headers={"User-Agent": "Mozilla/5.0"})
                c_art = urlopen(r_art, timeout=10).read()

                soup_art = BeautifulSoup(c_art, "html.parser")
                data_section = soup_art.findAll('section', attrs={'id': 'leftColumn'})[0]

                text_art = data_section.findAll('div', attrs={'class': 'WYSIWYG articlePage'})[0].text
                title_art = data_section.findAll('h1', attrs={'class': 'articleHeader'})[0].text
                info_art = data_section.findAll('div', attrs={'class': 'contentSectionDetails'})[0]

                src_art = info_art.findAll('img')[0].attrs['src']
                src_art = '.'.join(src_art.split('/')[-1].split('.')[:-1])
                date_art = info_art.findAll('span')[0].text.strip()
                date_art = parser.parse(date_art, tzinfos=tzinfos)

                df.append([ticker, src_art, title_art, text_art, date_art, INV_URL + article_link])
        except Exception:
            print(f'Error in {INV_URL + article_link}')
    
    return pd.DataFrame(df, columns=COLUMNS)


def download_ticker_news(ticker, start_page, end_page, filename=None, save_step=10):
    df = pd.DataFrame(columns=COLUMNS)
    
    link = get_investing_link_by_ticker(ticker)
    
    for i, page_num in enumerate(tqdm(range(start_page, end_page))):
        data = download_news_from_investing(link, page_num, ticker)
        df = pd.concat([df, data], ignore_index=True)
        
        if i % save_step == save_step - 1 and filename is not None:
            df.to_csv(filename)
    
    if filename is not None:
        df.to_csv(filename)
    else:
        return df


ticker = sys.argv[1]
start_page = int(sys.argv[2])
end_page = int(sys.argv[3])

save_step = 20
if len(sys.argv) > 4:
    save_step = int(sys.argv[4])

download_ticker_news(ticker, start_page, end_page + 1, filename=f'data/{ticker}_{start_page}_{end_page}.csv', save_step=save_step)

