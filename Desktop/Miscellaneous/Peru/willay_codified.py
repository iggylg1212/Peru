#Based on replication files for Wu (2018) https://www.aeaweb.org/articles?id=10.1257/pandp.20181101
#TO RECOVER numpy
#pip install --user --force-reinstall --ignore-installed --no-binary :all: numpy
from collections             import Counter
from nltk.corpus             import stopwords
from nltk.stem               import SnowballStemmer
from nltk.util               import ngrams
from sklearn                 import metrics
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import sklearn.preprocessing, sklearn.model_selection, sklearn.decomposition, sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn.model_selection import GridSearchCV
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model    import LogisticRegressionCV
from sklearn.ensemble    import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.neural_network import MLPClassifier

import re
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import time
import pickle
random.seed(1)

def to_ascii(x):
    x = str(x)
    original    = ['á','é','í','ó','ú','Á','É','Í','Ó','Ú','°','º','ñ','Ñ','ü','ÿ','Ÿ','¿','“','”','–','"','’','‘','‰','&','.','?',';','!',',','-',':']
    modified    = ['a','e','i','o','u','A','E','I','O','U','','','nh','NH','u','y','Y','' ,'' ,'' ,'' ,'' ,'' ,'' ,'%','y','','','','','','','','']
    for i, j in zip(original, modified):
        x = x.replace(i, j)
    x = x.upper()
    return x

#Instantiates Stemmer and cleans stopwords
stemmer = SnowballStemmer('spanish')
stopwords = [x.upper() for x in stopwords.words('spanish')]
stopwords = [to_ascii(word) for word in stopwords]

#Takes in both stata files and merges them on unique identifier; produces willay_merged csv and dataframe
worked = pd.read_stata('willay_worked.dta')
worked.drop_duplicates()
codified = pd.read_stata('willay_codified.dta')
codified['NUMERO_HECHO']=codified['NUMERO_HECHO'].astype(int)
codified.drop_duplicates()

denuncias = pd.merge(codified, worked, how="left", on=["NUMERO_RAC","NUMERO_HECHO"])

denuncias.to_csv('willay_merged.dta')

#Cleans dataframe and creates word tokens that will be used in NLP
denuncias = denuncias[['alarm','TIPO_DENUNCIANTE_x','CANAL_x','UO_ARA_x','RECONS_CAMBIOS_x','ALERTA_CIUDADANA_x','CODENTIDADSUJETA_x','SUMILLA_HECHO_x','DESCRIPCION_HECHO_x','TIPO_SISTEMA_x','SISTEMA_x','JUSTIFICACION_x','PUEDEIDENTIFICARSE_x','COMPETENCIACGR_x','TITULOHE_x','RESUMENHE_x','TIPIFICACION_x','RECURRENCIA_x','FINANCIAMIENTO_x','PRODUCTOPROPUESTO_x','PRODUCTOAPROBADO_x','ENTIDAD_DENUNCIADA','TIPO_DE_ENTIDAD','DEPARTAMENTO','PROVINCIA','DISTRITO','TIPODEDENUNCIA_PRIMARIO','TIPODEDENUNCIA_SECUNDARIO','TIPODEDENUNCIA_TERCIARIO','NOMBRES_PERSONAS_DENUNCIADAS','NOMBRES_PERSONAS_OTRAS','NOMBRES_EMPRESAS_INVOLUCRADAS','ENTIDADES_PUBLICAS_OTRAS','ENTIDADES_PRIVADAS_OTRAS','CONTRATACIONES_MENCIONADAS','PROYECTOS_ESTUDIOSMENCIONADOS','OBRAS_MENCIONADAS','DOCUMENTOS_MENCIONADOS','day','id_yy','N_yy','N_yy_denuncias','N_UO_claims','N_UO_denuncias','N_UO_evals','N_UO_alarms','dep','electricity17','pop17','N_dep_claims','N_dep_denuncias','N_dep_evals','N_dep_alarms','N_dep_claims100Kpop','N_dep_denuncias100Kpop','N_dep_evals100kpop','N_dep_alarms100kpop']]
denuncias = denuncias.dropna()
denuncias = denuncias.reset_index(drop=True)
denuncias['sort_key'] = [random.random() for i in range(len(denuncias))]
denuncias.sort_values(by='sort_key')

def nlpprocessor(variable):
    for i in variable:
        denuncias[i] = denuncias[[i]].applymap(to_ascii)
        denuncias[i] = [text.split() for text in denuncias[i]]
        denuncias[i] = [[stemmer.stem(s).upper() for s in split] for split in denuncias[i]]
        denuncias[i] = [[re.sub(r'[^a-zA-Z0-9]+', '', s) for s in split] for split in denuncias[i]]
        denuncias[i] = [[s for s in split if s != "" and s not in stopwords] for split in denuncias[i]]

nlpvariables=['SUMILLA_HECHO_x','DESCRIPCION_HECHO_x','JUSTIFICACION_x','TITULOHE_x','RESUMENHE_x','NOMBRES_PERSONAS_DENUNCIADAS','NOMBRES_PERSONAS_OTRAS','NOMBRES_EMPRESAS_INVOLUCRADAS','ENTIDADES_PUBLICAS_OTRAS','ENTIDADES_PRIVADAS_OTRAS','CONTRATACIONES_MENCIONADAS','PROYECTOS_ESTUDIOSMENCIONADOS','OBRAS_MENCIONADAS','DOCUMENTOS_MENCIONADOS']
nlpprocessor(nlpvariables)

#Transforms data, applying standard scaler, label binarizer, and tfidf vectorizer to select columns, producing numpy array
def dummy(doc):
    return doc

mapper = DataFrameMapper([('TIPO_DENUNCIANTE_x', sklearn.preprocessing.LabelBinarizer()),('CANAL_x', sklearn.preprocessing.LabelBinarizer()),('UO_ARA_x', sklearn.preprocessing.LabelBinarizer()),('RECONS_CAMBIOS_x', sklearn.preprocessing.LabelBinarizer()),('ALERTA_CIUDADANA_x', sklearn.preprocessing.LabelBinarizer()),('CODENTIDADSUJETA_x', sklearn.preprocessing.LabelBinarizer()),('SUMILLA_HECHO_x', TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, ngram_range=(1,2))),('DESCRIPCION_HECHO_x', TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, ngram_range=(1,2))),('TIPO_SISTEMA_x', sklearn.preprocessing.LabelBinarizer()),('SISTEMA_x', sklearn.preprocessing.LabelBinarizer()),('JUSTIFICACION_x', TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, ngram_range=(1,2))),('PUEDEIDENTIFICARSE_x', sklearn.preprocessing.LabelBinarizer()),('COMPETENCIACGR_x', sklearn.preprocessing.LabelBinarizer()),('TITULOHE_x', TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, ngram_range=(1,2))),('RESUMENHE_x', TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, ngram_range=(1,2))),('TIPIFICACION_x', sklearn.preprocessing.LabelBinarizer()),('RECURRENCIA_x', sklearn.preprocessing.LabelBinarizer()),('FINANCIAMIENTO_x', sklearn.preprocessing.LabelBinarizer()),('PRODUCTOPROPUESTO_x', sklearn.preprocessing.LabelBinarizer()),('PRODUCTOAPROBADO_x', sklearn.preprocessing.LabelBinarizer()),('ENTIDAD_DENUNCIADA', sklearn.preprocessing.LabelBinarizer()),('TIPO_DE_ENTIDAD', sklearn.preprocessing.LabelBinarizer()),('DEPARTAMENTO', sklearn.preprocessing.LabelBinarizer()),('PROVINCIA', sklearn.preprocessing.LabelBinarizer()),('DISTRITO', sklearn.preprocessing.LabelBinarizer()),('TIPODEDENUNCIA_PRIMARIO', sklearn.preprocessing.LabelBinarizer()),('TIPODEDENUNCIA_SECUNDARIO', sklearn.preprocessing.LabelBinarizer()),('NOMBRES_PERSONAS_OTRAS',TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, ngram_range=(1,2))),('NOMBRES_PERSONAS_DENUNCIADAS', TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, ngram_range=(1,2))),('NOMBRES_PERSONAS_DENUNCIADAS',TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, ngram_range=(1,2))),('ENTIDADES_PUBLICAS_OTRAS',TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, ngram_range=(1,2))),('ENTIDADES_PRIVADAS_OTRAS',TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, ngram_range=(1,2))),('PROYECTOS_ESTUDIOSMENCIONADOS',TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, ngram_range=(1,2))),('CONTRATACIONES_MENCIONADAS',TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, ngram_range=(1,2))),('OBRAS_MENCIONADAS',TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, ngram_range=(1,2))),('DOCUMENTOS_MENCIONADOS',TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, ngram_range=(1,2))),(['day'], sklearn.preprocessing.StandardScaler()),(['id_yy'], sklearn.preprocessing.StandardScaler()),(['N_yy'], sklearn.preprocessing.StandardScaler()),(['N_yy_denuncias'], sklearn.preprocessing.StandardScaler()),(['N_UO_claims'], sklearn.preprocessing.StandardScaler()),(['N_UO_denuncias'], sklearn.preprocessing.StandardScaler()),(['N_UO_evals'], sklearn.preprocessing.StandardScaler()),(['N_UO_alarms'], sklearn.preprocessing.StandardScaler()),('dep', sklearn.preprocessing.LabelBinarizer()),(['electricity17'], sklearn.preprocessing.StandardScaler()),(['pop17'], sklearn.preprocessing.StandardScaler()),(['N_dep_claims'], sklearn.preprocessing.StandardScaler()),(['N_dep_denuncias'], sklearn.preprocessing.StandardScaler()),(['N_dep_evals'], sklearn.preprocessing.StandardScaler()),(['N_dep_alarms'], sklearn.preprocessing.StandardScaler()),(['N_dep_claims100Kpop'], sklearn.preprocessing.StandardScaler()),(['N_dep_denuncias100Kpop'], sklearn.preprocessing.StandardScaler()),(['N_dep_evals100kpop'], sklearn.preprocessing.StandardScaler()),(['N_dep_alarms100kpop'], sklearn.preprocessing.StandardScaler())], sparse=True)
data_proc=mapper.fit_transform(denuncias)

# #Reduce dimensions using TruncatedSVD
# svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
# data_proc=svd.fit_transform(data)

#Creates numpy array from denuncias column for variable of interest
alarm=denuncias['alarm'].to_numpy()

#Saves processed data and alarm as pickles
pickle.dump(data_proc, open( "data_proc.p", "wb" ))
pickle.dump(alarm, open( "alarm.p", "wb" ))

#Opens processed data and alarm pickle
data_proc = pickle.load(open( "data_proc.p", "rb" ))
alarm = pickle.load(open( "alarm.p", "rb" ))
#Splits data into train and test groups
X_train, X_test, Y_train, Y_test = train_test_split(data_proc, alarm, test_size=0.33, random_state=42)
##################Model####################
# # Neural Network Grid Search
# #param_grid = {'activation':('identity','logistic', 'tanh', 'relu'), 'solver':('adam','lbfgs','sgd'), 'hidden_layer_sizes':((2,2),(4,4),(10,10),(25,25),(100,100),(25,25,25)), 'alpha':[1e-06,1e-05,0.001,0.1,1], 'learning_rate':('constant', 'invscaling', 'adaptive'), 'random_state':[42]}
# param_grid = {'solver':('adam','lbfgs','sgd')}
# mlp=MLPClassifier(activation='logistic', hidden_layer_sizes= (4,4), alpha = 1e-06, learning_rate = 'constant', random_state=42)
# #Performs exhaustive search over specified hyperparameters
# clf=GridSearchCV(mlp, param_grid, scoring='recall', n_jobs=6, cv=10, verbose=20)
# #Fits model with best parameters to test data
# best=clf.fit(X_train,Y_train)
# #Saves fit model with best parameters to pickle
# #pickle.dump(best, open( "NN_best.p", "wb" ))

# # Logistic Regression Grid Search
# #param_grid = {'penalty':('l1','l2','elasticnet'), 'Cs':[1e-4,1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3,1e4], 'solver':('newton-cg','lbfgs','liblinear','sag','saga'), 'random_state':[42]}
# param_grid = {'solver':('newton-cg','lbfgs','liblinear','sag','saga')}
# lr=LogisticRegressionCV()
# #Performs exhaustive search over specified hyperparameters
# clf=GridSearchCV(lr, param_grid, scoring='recall', cv=10, verbose=20, n_jobs=-1)
# #Fits model with best parameters to test data
# best=clf.fit(X_train,Y_train)
# #Saves fit model with best parameters to pickle
# pickle.dump(best, open( "logit_best.p", "wb" ))

# Random Forest Grid Search
param_grid = {'n_estimators':list(range(10,101,10)),'criterion':('gini','entropy'), 'max_depth':list(range(6,32,5)), 'bootstrap':('False','True'), 'oob_score':('False','True'), 'random_state':[42]}
rf=RandomForestClassifier()
#Performs exhaustive search over specified hyperparameters
clf=GridSearchCV(rf, param_grid, scoring='recall', cv=10, verbose=20, n_jobs=6, pre_dispatch=12)
#Fits model with best parameters to test data
best=clf.fit(X_train,Y_train)
#Saves fit model with best parameters to pickle
pickle.dump(best, open( "randomforest_best.p", "wb" ))

Y_pred_proba=best.predict_proba(X_test)
Y_pred=best.predict(X_test)

#Prints best parameters
print(best.best_params_)
#Finds the precision of the best model
print(precision_score(Y_test,Y_pred))
#Finds the accuracy of the best model
print(recall_score(Y_test,Y_pred))
#Finds the F1 of the model
print(f1_score(Y_test,Y_pred))
#Finds the AUC of the best model
fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred_proba[:, 1])
print(metrics.auc(fpr, tpr))
