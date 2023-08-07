# -*- coding: utf-8 -*-
"""
Spyder Editor
Created on Thu Aug  3 08:06:14 2023

@author: Luan Felix
"""

import numpy as np  # cálculo numérico
import pandas as pd  # preparação dos dados
import matplotlib.pyplot as plt  # pacote de plotagem
import matplotlib as mpl  # melhorar a plotagem
from sklearn.linear_model import LogisticRegression  # regressão lógica, modelo de classificação
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


mpl.rcParams['figure.dpi'] = 150

df = pd.read_csv('dados_explorados_e_limpos.csv')

# Contas que ficarão inadimplentes pertencem à classe positiva (=1) enquanto as que não ficarão pertencem à classe negativa
# Verificando qual a proporção de classe positiva:
df['default payment next month'].mean()  # demonstra que 22% das contas ficarão inadimplentes
df.groupby('default payment next month')['ID'].count()

# OBS: dados desbalanceados, modelos simples de ML trabalham com 50/50, aqui temos 22/78

my_lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001,
                           C=1.0, fit_intercept=True, intercept_scaling=1,
                           class_weight=None, random_state=None, solver='lbfgs',
                           max_iter=100, multi_class='auto', verbose=0,
                           warm_start=False, n_jobs=None, l1_ratio=None)
# opções padrão, para ter um bom modelo, temos que especificar algumas opções


""" 1º modelo para testarmos os codigos """
my_lr.C = 0.1
my_lr.solver = 'liblinear'

x = df['EDUCATION'][0:10].values.reshape(-1, 1)  # reshape -> redimensionar as caracteristicas.
# -1 -> linhas flexiveis, devem ir preechendo conferme for entrando dados.
# 1 -> uma coluna apenas.

y = df['default payment next month'][0:10].values

my_lr.fit(x, y)  # treinando o modelo

new_x = df['EDUCATION'][10:20].values.reshape(-1, 1)  # dados para teste

print("Valores previstos: ", my_lr.predict(new_x))
print("Valores reais: ", df['default payment next month'][10:20].values)


