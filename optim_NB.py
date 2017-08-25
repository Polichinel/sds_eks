# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:38:47 2017

@author: polichinel
"""

#%%
import nltk
import sklearn
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split as tts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import numpy as np
import pandas as pd
from sklearn import metrics

#%%
df = pd.read_csv('/home/polichinel/Dropbox/KU/7.semester/SDS/Eks/nyt/TestAndTrainMatrix_inclOtherIgnore.csv')
df = df.drop('Unnamed: 0', axis = 1)
df.iloc[3888, df.columns.get_loc('int_pol')] = 1
df.iloc[3518, df.columns.get_loc('int_pol')] = 1

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

df.keys()    
#%%
###############################################################################
########################## Optimerede par. int_pol ############################
###############################################################################

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) # Tfidf
X  =   vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['int_pol'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

nb = MultinomialNB()

# train the model using X_train_dtm
nb.fit(X_train, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test)

# calculate predicted probabilities for X_test_dtm (well calibrated)
#y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%%

# Setup the hyperparameter grid
alpha = np.linspace(0.001, 1)
param_grid = {'alpha': alpha, 'fit_prior' : [True, False], 'class_prior' : [None, [.1,.9],[.2, .8]]}  #, 'class_weight': weights

# Instantiate the GridSearchCV
nb_cv = GridSearchCV(nb, param_grid, cv=5, scoring='f1', n_jobs= -1)

# Fit it to the data
nb_cv.fit(X_train, y_train)


#%%
print(nb_cv.best_score_)
print(nb_cv.best_params_)
print(nb_cv.best_estimator_)
#%%
# Nu med optimerede parametre!
vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) # Tfidf
X  =   vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['int_pol'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

nb = MultinomialNB(alpha=0.16410204081632654, class_prior=None, fit_prior=True)

# train the model using X_train_dtm
nb.fit(X_train, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test)

# calculate predicted probabilities for X_test_dtm (well calibrated)
#y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


###############################################################################
###############################################################################
###############################################################################
#%%

###############################################################################
######################### Optimerede par. mil_kli #############################
###############################################################################

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) # Tfidf
X  =   vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['mil_kli'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

nb = MultinomialNB()

# train the model using X_train_dtm
nb.fit(X_train, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test)

# calculate predicted probabilities for X_test_dtm (well calibrated)
#y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%%

# Setup the hyperparameter grid
alpha = np.linspace(0.0001, 1)
param_grid = {'alpha': alpha, 'fit_prior' : [True, False], 'class_prior' : [None, [.1,.9],[.2, .8]]}  #, 'class_weight': weights

# Instantiate the GridSearchCV
nb_cv = GridSearchCV(nb, param_grid, cv=5, scoring='f1', n_jobs= -1)

# Fit it to the data
nb_cv.fit(X_train, y_train)


#%%
print(nb_cv.best_score_)
print(nb_cv.best_params_)
print(nb_cv.best_estimator_)
#%%
# Nu med optimerede parametre!
vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) # Tfidf
X  =   vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['mil_kli'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

nb = MultinomialNB(alpha=0.040912244897959187, class_prior=None, fit_prior=True)

# train the model using X_train_dtm
nb.fit(X_train, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test)

# calculate predicted probabilities for X_test_dtm (well calibrated)
#y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


###############################################################################
###############################################################################
###############################################################################
#%%
#%%
###############################################################################
########################## Optimerede par. kul_rel ############################
###############################################################################

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) # Tfidf
X  =   vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['kul_rel'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

nb = MultinomialNB()

# train the model using X_train_dtm
nb.fit(X_train, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test)

# calculate predicted probabilities for X_test_dtm (well calibrated)
#y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%%

# Setup the hyperparameter grid
alpha = np.linspace(0.001, 1)
param_grid = {'alpha': alpha, 'fit_prior' : [True, False], 'class_prior' : [None, [.1,.9],[.2, .8]]}  #, 'class_weight': weights

# Instantiate the GridSearchCV
nb_cv = GridSearchCV(nb, param_grid, cv=5, scoring='f1', n_jobs= -1)

# Fit it to the data
nb_cv.fit(X_train, y_train)


#%%
print(nb_cv.best_score_)
print(nb_cv.best_params_)
print(nb_cv.best_estimator_)
#%%
# Nu med optimerede parametre!
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=custom_tokenize) # Tfidf
X  =   vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['sta_tek'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

nb = MultinomialNB(alpha=0.12332653061224491, class_prior=None, fit_prior=True)

# train the model using X_train_dtm
nb.fit(X_train, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test)

# calculate predicted probabilities for X_test_dtm (well calibrated)
#y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


###############################################################################
###############################################################################
###############################################################################
#%%
#%%
###############################################################################
########################## Optimerede par. sta_tek ############################
##############################################################################

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=custom_tokenize) # Tfidf
X  =   vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['sta_tek'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

nb = MultinomialNB()

# train the model using X_train_dtm
nb.fit(X_train, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test)

# calculate predicted probabilities for X_test_dtm (well calibrated)
#y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%%

# Setup the hyperparameter grid
alpha = np.linspace(0.001, 1)
param_grid = {'alpha': alpha, 'fit_prior' : [True, False], 'class_prior' : [None, [.1,.9],[.2, .8]]}  #, 'class_weight': weights

# Instantiate the GridSearchCV
nb_cv = GridSearchCV(nb, param_grid, cv=5, scoring='f1', n_jobs= -1)

# Fit it to the data
nb_cv.fit(X_train, y_train)


#%%
print(nb_cv.best_score_)
print(nb_cv.best_params_)
print(nb_cv.best_estimator_)

#%%
# Nu med optimerede parametre!
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=custom_tokenize) # Tfidf
X  =   vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['sta_tek'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

nb = MultinomialNB(alpha=0.24565306122448982, class_prior=None, fit_prior=True)

# train the model using X_train_dtm
nb.fit(X_train, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test)

# calculate predicted probabilities for X_test_dtm (well calibrated)
#y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


###############################################################################
###############################################################################
###############################################################################
#%%
#%%
###############################################################################
########################## Optimerede par. kon_kon ############################
###############################################################################

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) # Tfidf
X  =   vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['kon_kon'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

nb = MultinomialNB()

# train the model using X_train_dtm
nb.fit(X_train, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test)

# calculate predicted probabilities for X_test_dtm (well calibrated)
#y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%%

# Setup the hyperparameter grid
alpha = np.linspace(0.001, 1)
param_grid = {'alpha': alpha, 'fit_prior' : [True, False], 'class_prior' : [None, [.1,.9],[.2, .8]]}  #, 'class_weight': weights

# Instantiate the GridSearchCV
nb_cv = GridSearchCV(nb, param_grid, cv=5, scoring='f1', n_jobs= -1)

# Fit it to the data
nb_cv.fit(X_train, y_train)


#%%
print(nb_cv.best_score_)
print(nb_cv.best_params_)
print(nb_cv.best_estimator_)
#%%
# Nu med optimerede parametre!
vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) # Tfidf
X  =   vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['kon_kon'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

nb = MultinomialNB(alpha=0.10293877551020408, class_prior=None, fit_prior=True)

# train the model using X_train_dtm
nb.fit(X_train, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test)

# calculate predicted probabilities for X_test_dtm (well calibrated)
#y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


###############################################################################
###############################################################################
###############################################################################
#%%

#%%
###############################################################################
########################## Optimerede par. fam_id ############################
###############################################################################

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) # Tfidf
X  =   vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['fam_id'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

nb = MultinomialNB()

# train the model using X_train_dtm
nb.fit(X_train, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test)

# calculate predicted probabilities for X_test_dtm (well calibrated)
#y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%%

# Setup the hyperparameter grid
alpha = np.linspace(0.001, 1)
param_grid = {'alpha': alpha, 'fit_prior' : [True, False], 'class_prior' : [None, [.1,.9],[.2, .8]]}  #, 'class_weight': weights

# Instantiate the GridSearchCV
nb_cv = GridSearchCV(nb, param_grid, cv=5, scoring='f1', n_jobs= -1)

# Fit it to the data
nb_cv.fit(X_train, y_train)


#%%
print(nb_cv.best_score_)
print(nb_cv.best_params_)
print(nb_cv.best_estimator_)
#%%
# Nu med optimerede parametre!
vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) # Tfidf
X  =   vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['fam_id'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

nb = MultinomialNB(alpha=0.95922448979591846, class_prior=[0.1, 0.9],fit_prior=True)

# train the model using X_train_dtm
nb.fit(X_train, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test)

# calculate predicted probabilities for X_test_dtm (well calibrated)
#y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


###############################################################################
###############################################################################
###############################################################################
#%%


#%%
###############################################################################
############################ Optimerede par. other ############################
###############################################################################

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) # Tfidf
X  =   vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['other'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

nb = MultinomialNB()

# train the model using X_train_dtm
nb.fit(X_train, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test)

# calculate predicted probabilities for X_test_dtm (well calibrated)
#y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%%

# Setup the hyperparameter grid
alpha = np.linspace(0.001, 1)
param_grid = {'alpha': alpha, 'fit_prior' : [True, False], 'class_prior' : [None, [.1,.9],[.2, .8]]}  #, 'class_weight': weights

# Instantiate the GridSearchCV
nb_cv = GridSearchCV(nb, param_grid, cv=5, scoring='f1', n_jobs= -1)

# Fit it to the data
nb_cv.fit(X_train, y_train)


#%%
print(nb_cv.best_score_)
print(nb_cv.best_params_)
print(nb_cv.best_estimator_)
#%%
# Nu med optimerede parametre!
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=custom_tokenize) # Tfidf
X  =   vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['other'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

nb = MultinomialNB(alpha=0.28642857142857142, class_prior=[0.1, 0.9],fit_prior=True)

# train the model using X_train_dtm
nb.fit(X_train, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test)

# calculate predicted probabilities for X_test_dtm (well calibrated)
#y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


###############################################################################
###############################################################################
###############################################################################
#%%


#%%
###############################################################################
########################### Optimerede par. igno ##############################
###############################################################################

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) # Tfidf
X  =   vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['igno'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

nb = MultinomialNB()

# train the model using X_train_dtm
nb.fit(X_train, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test)

# calculate predicted probabilities for X_test_dtm (well calibrated)
#y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%%

# Setup the hyperparameter grid
alpha = np.linspace(0.001, 1)
param_grid = {'alpha': alpha, 'fit_prior' : [True, False], 'class_prior' : [None, [.1,.9],[.2, .8]]}  #, 'class_weight': weights

# Instantiate the GridSearchCV
nb_cv = GridSearchCV(nb, param_grid, cv=5, scoring='f1', n_jobs= -1)

# Fit it to the data
nb_cv.fit(X_train, y_train)


#%%
print(nb_cv.best_score_)
print(nb_cv.best_params_)
print(nb_cv.best_estimator_)
#%%
# Nu med optimerede parametre!
vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) # Tfidf
X  =   vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['igno'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

nb = MultinomialNB(alpha=0.18448979591836737, class_prior=None, fit_prior=True)

# train the model using X_train_dtm
nb.fit(X_train, y_train)

# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test)

# calculate predicted probabilities for X_test_dtm (well calibrated)
#y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


###############################################################################
###############################################################################
###############################################################################
#%%