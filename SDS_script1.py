# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:27:59 2017

@author: Niels
"""

import SDS_module as SDS
import time

t = time.time()



df = pd.read_csv('13300.csv',encoding='utf-8')

navne = df.loc[:,'names']
df['gender'] = navne.apply(SDS.navne_gender)
fejl = df[df['gender']=='fejl']

print(-t + time.time())