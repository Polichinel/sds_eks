#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 19:38:59 2017

@author: polichinel
"""
#%%
import nltk
import sklearn
from sklearn.cross_validation import train_test_split as tts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score
import re
import numpy as np
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt

#%%
df = pd.read_csv('/home/polichinel/Dropbox/KU/7.semester/SDS/Eks/nyt/TestAndTrainMatrix_inclOtherIgnore.csv')
df = df.drop('Unnamed: 0', axis = 1)




#%%
df.head()

#%%
stemmer = nltk.stem.snowball.DanishStemmer()
def prep(text):
    wordlist = nltk.word_tokenize(text)
    wordlist = [stemmer.stem (w) for w in wordlist]
    pattern = '^[,;:?Â«<>Â»]+$'
    text = re.sub(pattern,'', text)
    
    return wordlist

def custom_tokenize(text):
   
    text = re.sub('^[,;:?Â«<>Â»]+$','',text)
    
    wordlist = prep(text) # our old function
    wordlist = [word.strip('.,"?') for word in wordlist]
    return wordlist





#%%

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) # Count
X  = vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['int_pol'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)


nb = BernoulliNB(alpha=0.1)
# train the model using X_train
nb.fit(X_train, y_train)
y_pred_class = nb.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))





#%%
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=custom_tokenize) # Tfidf
X  = vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['int_pol'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

cl_weight = {0:1, 1:4.42857142857} #Rigtig god - god balance med relativt høj recall. se dit plot!

#cl_weight = {0:1, 1:3.0789473684210527} #Rigtig god - High f1

logreg = LogisticRegression(class_weight=cl_weight,solver='newton-cg')

# train the model using X_train_dtm
logreg.fit(X_train, y_train)

# make class predictions for X_test_dtm
y_pred_class = logreg.predict(X_test)

# calculate predicted probabilities for X_test_dtm (well calibrated)
#y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))

print('roc_curve: ',metrics.roc_curve(y_test, y_pred_class))
print('roc_auc_curve: ',metrics.roc_auc_score(y_test, y_pred_class))

#%%
# Det her skal vi havde til at virke!!! +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=custom_tokenize) # Tfidf
X  = vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['int_pol'])

print(cross_val_score(logreg, X, y, cv=5))
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#%% Loop over vÃ¦gte
start = 3
stop = 5
steps = 50
meassures = {'acc_score':[], 'prec_score':[], 'recall_score':[], 'f1_score':[], 'conf_matr':[],'acc_score_o':[], 'prec_score_o':[], 'recall_score_o':[], 'f1_score_o':[], 'weight':[]}
for i in np.linspace(start, stop, steps):
    cl_weight = {0:1, 1:i}
    logreg = LogisticRegression(class_weight=cl_weight,solver='newton-cg')
    logreg.fit(X_train, y_train)
    y_pred_class = logreg.predict(X_test)
    acc_score = metrics.accuracy_score(y_test, y_pred_class)
    prec_score = metrics.precision_score(y_test, y_pred_class)
    recall_score = metrics.recall_score(y_test, y_pred_class)
    f1_score =     metrics.f1_score(y_test, y_pred_class)
    confusion = list(metrics.confusion_matrix(y_test, y_pred_class))
    meassures['conf_matr'].append([confusion])
    meassures['acc_score'].append([i, acc_score])
    meassures['prec_score'].append([i, prec_score])
    meassures['recall_score'].append([i, recall_score])
    meassures['f1_score'].append([i, f1_score])
    meassures['acc_score_o'].append([acc_score]) # til plt
    meassures['prec_score_o'].append([prec_score])
    meassures['recall_score_o'].append([recall_score]) # til plt
    meassures['f1_score_o'].append([f1_score]) # til plt
    meassures['weight'].append([i]) # til plt

# du mangler så prec!

#%%

plt.plot(meassures['weight'],meassures['acc_score_o'])

plt.plot([4.42857142857, 4.42857142857], [0.8, 0.86], 'k-', lw=1, color='green') 
#Vertical: husk du har tallet (4.42857142857) fra overstående (acc) loop!
plt.plot([2.9, 5], [0.82452642073778659, 0.82452642073778659], 'k-', lw=1, color='green')
#Horisontel: husk du har tallet (0.637860082305) fra overstående loop (acc) hvor w = 4.42857142857!


plt.plot([3.0816326530612246, 3.0816326530612246], [0.8, 0.86], 'k-', lw=1, color='red')
 #Vertical: husk du har tallet (3.081632653061224) fra overstående (acc) loop 
plt.plot([2.9, 3.26], [0.85244267198404788, 0.85244267198404788], 'k-', lw=1, color='red')
 #Horisontel:husk du har tallet (0.85244267198404788) fra overstående loop (acc) hvor w =  3.0816326530612246!



#%%

plt.plot(meassures['weight'],meassures['prec_score_o'])

plt.plot([4.42857142857, 4.42857142857], [0.55, 0.7], 'k-', lw=1, color='green') 
#Vertical: husk du har tallet (4.42857142857) fra overstående (prec) loop!
plt.plot([2.9, 5], [0.58052434456928836, 0.58052434456928836], 'k-', lw=1, color='green')
#Horisontel: husk du har tallet (0.58052434456928836) fra overstående loop (prec) hvor w = 4.42857142857!


plt.plot([3.0816326530612246, 3.0816326530612246], [0.55, 0.7], 'k-', lw=1, color='red') 
#Vertical: husk du har tallet (3.081632653061224) fra overstående (prec) loop!
plt.plot([2.9, 3.26], [0.68205128205128207, 0.68205128205128207], 'k-', lw=1, color='red') 
#Horisontel:husk du har tallet (0.68205128205128207) fra overstående loop (prec) hvor w = 4.42857142857!




#%%
plt.plot(meassures['weight'],meassures['recall_score_o'])

plt.plot([4.42857142857, 4.42857142857], [0.6, 0.72], 'k-', lw=1, color='green')
 #Vertical: husk du har tallet (4.42857142857) fra overstående (recall) loop!
plt.plot([2.9, 5], [0.70776255707762559, 0.70776255707762559], 'k-', lw=1, color='green')
#Horisontel: husk du har tallet (0.637860082305) fra overstående loop (recall) hvor w = 4.42857142857!


plt.plot([3.0816326530612246, 3.0816326530612246], [0.6, 0.72], 'k-', lw=1, color='red') 
#Vertical: husk du har tallet (3.081632653061224) fra overstående loop!3.0816326530612246
plt.plot([2.9, 3.26], [0.60730593607305938, 0.60730593607305938], 'k-', lw=1, color='red') 
#Horisontel:husk du har tallet (0.64251207729468596) fra overstående loop hvor w =  3.0816326530612246!


#%%
plt.plot(meassures['weight'],meassures['f1_score_o'])
plt.plot([4.42857142857, 4.42857142857], [0.615, 0.65], 'k-', lw=1, color='green') #Vertical: husk du har tallet (4.42857142857) fra overstående loop (f1)!

plt.plot([2.9, 5], [0.637860082305, 0.637860082305], 'k-', lw=1, color='green')#Horisontel: husk du har tallet (0.637860082305) fra overstående loop (f1) hvor w = 4.42857142857!

plt.plot([3.0816326530612246, 3.0816326530612246], [0.615, 0.648], 'k-', lw=1, color='red') #Vertical: husk du har tallet (3.081632653061224) fra overstående loop loop (f1)
plt.plot([2.9, 3.25], [0.64251207729468596, 0.64251207729468596], 'k-', lw=1, color='red') #Horisontel:husk du har tallet (0.64251207729468596) fra overstående loop (f1) hvor w =  3.0816326530612246!


#%%
# FInt, men nu skal du også lave en for acc og prec!!!!
# og der skal text på!!!
#%%






















