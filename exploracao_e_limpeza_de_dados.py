import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

""" Obtendo os dados """
# https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients -> dicionario dos dados


df = pd.read_excel("default_of_credit_card_clients__courseware_version_1_21_19.xls")


""" Explorando os dados """


print("Quantidade de linhas e colunas: ", df.shape)
print()
# print(df.head())  # Ver as 5 primeiras linhas do DataFrame

print("Quantidade de linhas:", df.shape[0])
print("Quantidade de dados exclusivos de cada linha da coluna \"ID\":", len(df['ID'].unique()))

qnt_id = df['ID'].value_counts()  # value_counts() / groupby|count do SQL
print("")
print("1 - ids unicos\n2 - ids repetidos", qnt_id.value_counts())  # retornando quantos ids repetidos há
print("")

# Mascara Booleana
mascara_ids = qnt_id == 2  # armazendo em bool os ids são repetidos
print("True - Id repetido", mascara_ids[:5], end="\n\n")

print("Index dos ids duplicados:\n", qnt_id.index[mascara_ids][:5], end="\n\n")  # retornando o index dos ids duplicados
ids_duplicados = qnt_id.index[mascara_ids]
ids_duplicados = list(ids_duplicados)
print("IDs duplicados: ", len(ids_duplicados))
print("Dados do ids duplicados:")
print(df.loc[df['ID'].isin(ids_duplicados[0:3]), :].head(3))  # .loc e .isin permitem uma melhor visualização do DataFrame a partir do index

# Removendo registro inválidos (só há "0" nos dados)
mascara_df_zero = df == 0  # retorna True onde houver "0" dentro do dataframe
print("True - 0")
print(mascara_df_zero[0:10])
print("")

# verificando se há linhas inteirissas de 0.
linhas_invalidas = mascara_df_zero.iloc[:, 1:].all(axis=1)
print("Linha inválida - True\n", linhas_invalidas)
# .iloc metodo de indexacao de inteiros
# : <-> todas as linhas  1: <-> apartir da coluna 1
# all() retorna True se todas as colunas == True
print("\nTotal de linhas inválidas:", sum(linhas_invalidas))  # True = 1

# Limpando as linhas inválidas do DataFrame
df_limpo_1 = df.loc[~linhas_invalidas, :].copy()  # ~ -> not | Copiamos todas linhas que não são True, não são inválidas

print("\n\nLimpeza de DataFrame realizada:")
print("Linhas totais no novo DataFrame - ", df_limpo_1.shape[0], "|", "IDs Únicos - ", df_limpo_1['ID'].nunique())

# Verificando mais colunas
print("\nID  PAY_1\n", df_limpo_1['PAY_1'].head(3))
print("")
print("Quantidade por tipos de dados na coluna:", df_limpo_1['PAY_1'].value_counts())
# Not Available: Pandas add simbolizando um valor ausente

# Mascara Booleana
print("")
pagamentos_validos_mascara = df_limpo_1['PAY_1'] != "Not available"
print("Valore válidos (sem \"Not available\"):", sum(pagamentos_validos_mascara))

df_limpo_2 = df_limpo_1.loc[pagamentos_validos_mascara, :].copy()
print("Novo DataFrame:", df_limpo_2.shape)

# Convertendo o tipo da coluna de "Object" para "Int64"
print("")
df_limpo_2['PAY_1'] = df_limpo_2['PAY_1'].astype("int64")  # astype converter o tipo de dados da coluna
print("Infos PAY_1")
print(df_limpo_2['PAY_1'].info())

