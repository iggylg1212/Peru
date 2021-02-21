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

# def to_ascii(x):
#     x = str(x)
#     original    = ['á','é','í','ó','ú','Á','É','Í','Ó','Ú','°','º','ñ','Ñ','ü','ÿ','Ÿ','¿','“','”','–','"','’','‘','‰','&','.','?',';','!',',','-',':']
#     modified    = ['a','e','i','o','u','A','E','I','O','U','','','nh','NH','u','y','Y','' ,'' ,'' ,'' ,'' ,'' ,'' ,'%','y','','','','','','','','']
#     for i, j in zip(original, modified):
#         x = x.replace(i, j)
#     x = x.upper()
#     return x
#
# #Instantiates Stemmer and cleans stopwords
# stemmer = SnowballStemmer('spanish')
# stopwords = [x.upper() for x in stopwords.words('spanish')]
# stopwords = [to_ascii(word) for word in stopwords]
#
# #Takes in both stata files and merges them on unique identifier; produces willay_merged csv and dataframe
# worked = pd.read_stata('willay_worked.dta')
# worked.drop_duplicates()
# codified = pd.read_stata('willay_all_new.dta')
# codified['NUMERO_HECHO']=codified['NUMERO_HECHO'].astype(int)
# codified.drop_duplicates()
#
# denuncias = pd.merge(codified, worked, how="left", on=["NUMERO_RAC","NUMERO_HECHO"])
#
# denuncias.to_csv('willay_merged.dta')
#
# #Cleans dataframe and creates word tokens that will be used in NLP
# denuncias = denuncias[['TIPO_DENUNCIANTE_x','CANAL_x','UO_ARA', "RECONS_CAMBIOS",	"ALERTA_CIUDADANA", 'NOMENTIDAD',"SUMILLA_HECHO",'DESCRIPCION_HECHO_x',"TIPO_SISTEMA","SISTEMA_x","ENTIDAD_DENUNCIADA",'TIPO_DE_ENTIDAD',"DEPARTAMENTO","PROVINCIA","DISTRITO","TIPODEDENUNCIA_PRIMARIO","TIPODEDENUNCIA_SECUNDARIO","TIPODEDENUNCIA_TERCIARIO", "NOMBRES_PERSONAS_DENUNCIADAS", "NOMBRES_EMPRESAS_INVOLUCRADAS", "OBRAS_MENCIONADAS", "DOCUMENTOS_MENCIONADOS",'CODENTIDADSUJETA', 'CIERRE_DESGLOSE',	'JUSTIFICACION', 'COMPETENCIACGR', 'TITULOHE', 'RESUMENHE','RECURRENCIA','FINANCIAMIENTO', 'day','id_yy','N_yy', 'N_yy_denuncias', 'alarm', 'dep', 'electricity17', 'pop17','N_dep_claims',	'N_dep_denuncias', 'N_dep_evals', 'N_dep_alarms', 'N_dep_claims100Kpop', 'N_dep_denuncias100Kpop', 'N_dep_evals100kpop','N_dep_alarms100kpop']]
# denuncias = denuncias.dropna()
# denuncias = denuncias.reset_index(drop=True)
# denuncias['sort_key'] = [random.random() for i in range(len(denuncias))]
# denuncias.sort_values(by='sort_key')
#
# def nlpprocessor(variable):
#     for i in variable:
#         denuncias[i] = denuncias[[i]].applymap(to_ascii)
#         denuncias[i] = [text.split() for text in denuncias[i]]
#         denuncias[i] = [[stemmer.stem(s).upper() for s in split] for split in denuncias[i]]
#         denuncias[i] = [[re.sub(r'[^a-zA-Z0-9]+', '', s) for s in split] for split in denuncias[i]]
#         denuncias[i] = [[s for s in split if s != "" and s not in stopwords] for split in denuncias[i]]
#
# nlpvariables=['SUMILLA_HECHO','DESCRIPCION_HECHO_x',"NOMBRES_PERSONAS_DENUNCIADAS","NOMBRES_EMPRESAS_INVOLUCRADAS","OBRAS_MENCIONADAS","DOCUMENTOS_MENCIONADOS",'JUSTIFICACION','TITULOHE', 'RESUMENHE']
# labelvariables=['TIPO_DENUNCIANTE_x','CANAL_x','UO_ARA', 'RECONS_CAMBIOS',"ALERTA_CIUDADANA", 'NOMENTIDAD', "TIPO_SISTEMA","SISTEMA_x","ENTIDAD_DENUNCIADA",'TIPO_DE_ENTIDAD',"DEPARTAMENTO","PROVINCIA","DISTRITO","TIPODEDENUNCIA_PRIMARIO","TIPODEDENUNCIA_SECUNDARIO","TIPODEDENUNCIA_TERCIARIO",'CODENTIDADSUJETA','CIERRE_DESGLOSE','COMPETENCIACGR','RECURRENCIA','FINANCIAMIENTO','id_yy','dep']
# scalervariables=['day','N_yy', 'N_yy_denuncias','electricity17', 'pop17','N_dep_claims', 'N_dep_denuncias', 'N_dep_evals', 'N_dep_alarms', 'N_dep_claims100Kpop', 'N_dep_denuncias100Kpop', 'N_dep_evals100kpop','N_dep_alarms100kpop']
#
# #Transforms data, applying standard scaler, label binarizer, and tfidf vectorizer to select columns, producing numpy array
# def dummy(doc):
#     return doc
#
# nlpprocessor(nlpvariables)
#
# def construct(nlp,label,scaler):
#     obj=[]
#     for i in nlp:
#         obj.append((i, TfidfVectorizer(tokenizer=dummy, preprocessor=dummy, ngram_range=(1,2))))
#     for e in label:
#         obj.append((e, sklearn.preprocessing.LabelBinarizer()))
#     for a in scaler:
#         obj.append(([a], sklearn.preprocessing.StandardScaler()))
#     return obj
#
# mapper=DataFrameMapper(construct(nlpvariables, labelvariables, scalervariables) ,sparse=True)
#
# data_proc=mapper.fit_transform(denuncias)
#
# # #Reduce dimensions using TruncatedSVD
# # svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
# # data_proc=svd.fit_transform(data)
#
# #Creates numpy array from denuncias column for variable of interest
# alarm=denuncias['alarm'].to_numpy()
#
# #Saves processed data and alarm as pickles
# pickle.dump(data_proc, open( "data_proc.p", "wb" ))
# pickle.dump(alarm, open( "alarm.p", "wb" ))

#Opens processed data and alarm pickle
data_proc = pickle.load(open( "data_proc.p", "rb" ))
alarm = pickle.load(open( "alarm.p", "rb" ))
#Splits data into train and test groups
X_train, X_test, Y_train, Y_test = train_test_split(data_proc, alarm, test_size=0.33, random_state=42)
##################Model####################
# Neural Network Grid Search
#param_grid = {'activation':('identity','logistic', 'tanh', 'relu'), 'solver':('adam','lbfgs','sgd'), 'hidden_layer_sizes':((2,2),(4,4),(10,10),(25,25),(100,100),(25,25,25)), 'alpha':[1e-06,1e-05,0.001,0.1,1], 'learning_rate':('constant', 'invscaling', 'adaptive'), 'random_state':[42]}
param_grid = {}
mlp=MLPClassifier(activation='logistic',solver='adam', hidden_layer_sizes= (2,2), alpha =1e-05, learning_rate = 'constant', random_state=42)
#Performs exhaustive search over specified hyperparameters
clf=GridSearchCV(mlp, param_grid, scoring='recall', n_jobs=9, cv=10, verbose=20)
#Fits model with best parameters to test data
best=clf.fit(X_train,Y_train)
#Saves fit model with best parameters to pickle
#pickle.dump(best, open( "NN_best.p", "wb" ))

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

# # Random Forest Grid Search
# param_grid = {'n_estimators':list(range(10,101,10)),'criterion':('gini','entropy'), 'max_depth':list(range(6,32,5)), 'bootstrap':('False','True'), 'oob_score':('False','True'), 'random_state':[42]}
# rf=RandomForestClassifier()
# #Performs exhaustive search over specified hyperparameters
# clf=GridSearchCV(rf, param_grid, scoring='recall', cv=10, verbose=20, n_jobs=6, pre_dispatch=12)
# #Fits model with best parameters to test data
# best=clf.fit(X_train,Y_train)
# #Saves fit model with best parameters to pickle
# pickle.dump(best, open( "randomforest_best.p", "wb" ))

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
