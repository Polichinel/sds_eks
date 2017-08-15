
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
from bs4 import BeautifulSoup
#from selenium.webdriver.common.by import By
import re

#%% Header
session = requests.session()
session.headers['emails'] = "fdz242@alumni.ku.dk, tfj674@alumni.ku.dk, clj135@alumni.ku.dk og xpn381@alumni.ku.dk"
session.headers['names'] = "Elias, Niels, Julius og Simon"
session.headers['description'] = "Til brug for eksamen i Social Data Science, KU (http://kurser.ku.dk/course/a%C3%98kk08216u/2016-2017)"
#session.headers
#%% 
##############################################################
### Links til de enkelte artikler hentes fra oversigtsside ###
##############################################################

baseurl = 'https://www.information.dk'
debat_page_0 = '/debat?page='
debat_page_1 = '&lst_tag'

def get_art_links(startindex, slutindex):
    """
    Funktion der tager et sideindeks fra oversigtssiden og spytter en dataframe ud
    med links.
    Input: Start og Slutindeks
    Output: DataFrame
    """
    titles = []
    links =[]
    page_no = range(startindex, slutindex) #Sæt antal oversigtsartikler
    for pgno in page_no:
        time.sleep(1)
        link = baseurl + debat_page_0 + str(pgno) + debat_page_1
        text = requests.get(link).text
        soup = BeautifulSoup(text, 'lxml')
        articles = soup.findAll("h3", {'class':'node-title'})
        for art in articles:
            links.append(art.a['href'])
            titles.append(art.a.get_text())
    print('-')
    df = pd.DataFrame()
    df['links'] = pd.Series(links)
    df['titles'] = pd.Series(titles)
    df['names'] = np.nan
    df['text'] = np.nan
    df['comments_no'] = np.nan
    df['date'] = np.nan
    return df

def get_art_data(link):
    """
    Ud fra fuld link til artikel på Information.dk findes forfatter(e), manchet og antallet af kommentarer.
    Input: Link til artikel (str)
    Output: list (len=3) forfatter (str), tekst (str) og antal kommentarer (int)
    """
    time.sleep(1)
    baseurl = 'https://www.information.dk'
    text = requests.get(baseurl + link).text
    soup = BeautifulSoup(text, "lxml")
    content = soup.findAll("div", {'class':'byline-comma-list'})
    forfatter = content[0].get_text()
    content2 = soup.findAll('div', {'class':'field field-name-body'})
    if len(content2)>0:
        teksten = content2[0].get_text()
    else:
        teksten = ""
    commentNo = soup.findAll('div', {'class':'field field-name-comment-count-top'})
    txt = commentNo[0].get_text()
    no_comment = int(re.compile('\d+').findall(txt)[0])
    cont = soup.findAll('div', {'class':'field field-name-print-date-w-fallbacks field-type-ds field-label-hidden'})
    date = cont[0].get_text()
    return forfatter, teksten, no_comment, date


#%% 
errors = []
j = 1
trin = 100
startside = 300 # Nummer på første oversigtsside
slutside = 4101 # Nummer på den sidste (minus 'trin') oversigtsside
index_list = list(np.arange(startside,slutside, step=trin))

for i in index_list: 
    start = i
    slut = start + trin
    df = get_art_links(start, slut)
    for index, row in df.iterrows():
        if index % 100 == 0:
            print('Linje', index, 'fil nr.', str(j))
        try:
            art_data = get_art_data(row['links'])
            df.loc[index, 'names'] = art_data[0]
            df.loc[index, 'text'] = art_data[1]
            df.loc[index, 'comments_no'] = art_data[2]
            df.loc[index, 'date'] = art_data[3]
        except (RuntimeError, TypeError, NameError, ZeroDivisionError, IndexError) as err:
            fejlstr = 'Fejl: ' + str(err) + ' - Linje ' + str(index) + ' fil nr ' + str(j)          
            print(fejlstr)
            errors.append(fejlstr)
            pass
    
    filename = 'text'+str(j)+'.csv'
    df.to_csv(filename, encoding='utf-8', header=True)
    print('Fil nummer ' + str(j) + ' er gemt')
    j += 1
