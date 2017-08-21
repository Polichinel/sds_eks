# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 14:54:52 2017

@author: Gruppe 2 i SDS_exam
"""
import pandas as pd 
from itertools import compress
import numpy as np

def leder_laeserbreve(dataframe):
    """
    Funktioner der opretter leder og læserbrevevariabel
    Input: DataFrame (med kolonne, der hedder links)
    Ouput: DataFrame med to yderligere variable
    """
    dataframe['leder'] = [1 if '/debat/leder/' in li else 0 for li in df['links']]
    dataframe['lbreve'] = [1 if '/laeserbreve-' in li else 0 for li in df['links']]
    return dataframe

mon = {'januar':'Jan',
   'februar':'Feb',
   'marts':'Mar', 
   'april':'Apr',
   'maj':'May',
   'juni':'Jun',
   'juli':'Jul',
   'august':'Aug',
   'september':'Sep',
   'oktober':'Oct',
   'november':'Nov',
   'december':'Dec'}

def dato_omskriv(ds):
    """
    Funktion, der tager input som string sådan som det står på information.dk. Omdanner det til en timestamp.
    Input: Dato som String med kolonne
    Ouput: Dato som timestamp
    """
    try:
        ds = str(ds)
        #print(ds)
        dag = ds.replace('.','')
        #print(dag)
        mdr = dag.split(' ')[1]
        dag = dag.replace(mdr,mon[mdr])
        dat = datetime.strptime(dag, '%d %b %Y')
    except (IndexError, TypeError, UnboundLocalError, KeyError):
        dat = np.nan
    return dat

navne_p = list(pd.read_csv('Pige.csv',encoding='utf-8',  header=None)[0])
navne_d = list(pd.read_csv('Drenge.csv',encoding='utf-8',  header=None)[0])

## rettelser mht. køn og navne baseret på vores data.
Piger = ['Dominique','Nana','Lykke','Deniz','Kit','Sacha','Pil','Elisa',
         'Maxime','Linn','Mai','Justice',
         'Maria','Nikola','Nour','Nur','Jannie','Robin',
         'Maj','Andrea','Gunde','Gry','Michel','Anda','Misha',
         'Jo', 'Sandy','Rana', 'Anne','Gabi','Bjørk',
         'Jochen', 'Zelle', 'Nushin', 'Agi', 'Kanar', 'Boonyoung', 'Tawakkol', 
         'Lærke-Sofie', 'Vandana', 'Camilla-Dorthea', 'Fern', 'Trinelise', 'Priyamvada', 
         'Byung-Chul', 'Jayati', 'Navi', 'Maaza', 'Sadhbh', 'Bernardine', 'Saratu', 
         'Emer', 'Yanaba', 'Noreena', 'Leny', 'ELSEBETH', 'Joan','Jaleh',
         'Kat.', 'Sausan', 'Yechiela', 'Mairav', 'Mette-Line', 'Rose-Sofie', 
         'Eini', 'Aleqa']

Drenge = ['Sam', 'Lave', 'Ray', 'Elias', 'Bo', 'Manu', 'Dan','Tonny','Kim','Tonni','Nadeem',
          'Alex','Ronnie','Addis','Kai','Glenn','Joe','Hamdi','Chris', 'Saman',  'Alaa',   'Roman',   
          'Benny', 'Iman', 'Ryan',  'Mikka', 'Jean','Slavoj', 'Johnny','Evin', 'Sami', 'Dani',
          'Jens-André', 'Noralv', 'Miguel-Anxo', 'Srecko', 'Sven-Åge', 'Gwynne', 'Shadman', 
          'Evgeny', 'Wajahat', 'Ban', 'Slavoj Žižek', 'Johs.', 'Kehinde', 'Seumas', 
          'Joschka', 'Costas', 'Poyâ', 'Yuriy', 'Razmig', 'Manyar', 'Gérard', 'Jotam', 'Hilik', 
          'László', 'Nussaibah', 'Fiachra', 'Rafał', 'MARTIN', 'Yehuda', 
          'Arulanantharajah', 'Jin-Tae', 'Wadah', 'Hew', 'Ota', 'Tomaso','Sarfraz', 
          'Hallgrímur', 'Peter-Christian', 'Aqbal', 'Bengt-Åke', 'Simplice', 'Praveen', 'Fareed', 
          'Ed', 'Karl-Ludwig', 'Per-Olof', 'Jean-François', 'Torbjörn', 'Jan-Werner', 'Sik', 
          'Bernard-Henri', 'Niels-Anton', 'Ha-Joon', 'Rune-Christoffer', 'Thuraya', 'Senai', 
          'Wadah', 'Lindhart', 'Rens', 'Amartya', 'Sanou', 'Pankaj', 'Johnjoe', 'Paddy',
          'Prospéry', 'Oleh', 'Noa-Noelle','Masahiro', 'Jonatann', 'Ghayth', 'Brahma', 'Smike']

dr1 = [(txt not in Piger) for txt in navne_d]
pi1 = [(txt not in Drenge) for txt in navne_p]
drengeunik = list(compress(navne_d, dr1))
pigeunik = list(compress(navne_p, pi1))
drengenavne = list(set(drengeunik + Drenge))
pigenavne = list(set(pigeunik + Piger))
del navne_d, navne_p, dr1, pi1, Piger, Drenge, drengeunik, pigeunik

def fornavne(names):
    try:
        navne = names.replace(' og ',',')
        navne_mult = navne.split(',')
        navne_mult = list(map(str.strip, navne_mult))
        fornavne = []
        for name in navne_mult:
            fornavne.append(name.split(' ')[0])
    except:
        fornavne = ""
    return fornavne

def navne_gender(navne):
    first_name = fornavne(navne)
    #print(type(first_name))
    if len(first_name) == 1:
        first_name = first_name[0]
        if first_name in drengenavne:
            gender = 'male'
        elif first_name in pigenavne:
            gender = 'female'
        else:
            gender = 'fejl'
    else:
        n_container = []
        for n in first_name:
            if n in drengenavne:
                n_container.append(0)
            elif n in pigenavne:
                n_container.append(1)
            else:
                pass
        if sum(n_container) == 0:
            gender = 'all male'
        elif sum(n_container) == len(n_container):
            gender = 'all female'
        else:
            gender = 'mixed'
    return gender