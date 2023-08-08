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
from sklearn.linear_model import LogisticRegression, LinearRegression  # regressão lógica, modelo de classificação
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html



""" CONFIGURANDO E CARREGANDO DADOS """
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



""" 1º modelo para testarmos os codigos - Introdução ao Scikit-Learn"""
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



""" Gerar dados sinteticos """
np.random.seed(seed=1)  # seed -> faz uma pseudorandomizacao
x = np.random.uniform(low=0, high=10, size=(1000,))
x[0:10]



""" Dados para uma regressão linear """
# criando dados lineares aleatorios com ruidos gaussianos
# y = ax + b + N (µ, σ)

slope = 0.25
intercept = -1.25

y = slope * x + np.random.normal(loc=0.0, scale=1.0, size=(1000,)) + intercept # loc -> media do ruido | scale -> desvio padrao

plt.scatter(x, y, s=1)  # s - tamanho dos pontos



""" Regressão Linear com o Scikit-Learn """
# fazendo um modelo de regressão em cima dos dados sinteticos gerados
lin_reg = LinearRegression()
lin_reg.fit(x.reshape(-1, 1), y)

print(lin_reg.intercept_)  # interceptação 
print(lin_reg.coef_)  # inclinação

y_pred = lin_reg.predict(x.reshape(-1, 1))

# plotando a linha a de melhor a ajuste para os dados
plt.scatter(x, y, s=1)
plt.plot(x, y_pred, 'r')  # 'r' da a linha a cor vermelha


