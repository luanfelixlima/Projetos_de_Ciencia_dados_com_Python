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

mpl.rcParams['figure.dpi'] = 150

df = pd.read_csv('dados_explorados_e_limpos.csv')

# Contas que ficarão inadimplentes pertencem à classe positiva (=1) enquanto as que não ficarão pertencem à classe negativa
# Verificando qual a proporção de classe positiva:
df['default payment next month'].mean()  # demonstra que 22% das contas ficarão inadimplentes
df.groupby('default payment next month')['ID'].count()

# OBS: dados desbalanceados, modelos simples de ML trabalham com 50/50, aqui temos 22/78