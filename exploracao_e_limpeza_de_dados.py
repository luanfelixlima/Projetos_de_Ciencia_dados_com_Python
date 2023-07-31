import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

""" Obtendo os dados """
# https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients -> dicionario dos dados
df = pd.read_excel("default_of_credit_card_clients__courseware_version_1_21_19.xls")


""" Explorando e Limpando os dados """


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
print("")

# histogramas
mpl.rcParams["figure.dpi"] = 400  # alta resolução de imagem
df_limpo_2[['LIMIT_BAL', 'AGE']].hist()  # gerando um histograma do limite de credito e da idade
print(df_limpo_2[['LIMIT_BAL', 'AGE']].describe())  # relatorio tabular

# limpando coluna education
print("")
print("De acordo com o dicionario de dados 0, 5 e 6 não correspondem a nenhum grau de educação\n"
      "vamos mover eles para o grau \"outros\" [4]")
print(df_limpo_2['EDUCATION'].value_counts())
df_limpo_2['EDUCATION'].replace(to_replace=[0, 5, 6], value=4, inplace=True)  # inplace -> não cria um dataFrame novo, apenas altera o existente
print(df_limpo_2['EDUCATION'].value_counts())  # df limpo

# limpando coluna marriage
print("")
print("De acordo com o dicionario de dados 0 não correspondem a nenhum valor de estado civil\n"
      "vamos mover eles para o estado de \"outros\" [3]")
print(df_limpo_2['MARRIAGE'].value_counts())
df_limpo_2['MARRIAGE'].replace(to_replace=0, value=3, inplace=True)
print(df_limpo_2['MARRIAGE'].value_counts())  # df limpo
print("")

# inverteremos as classificações para aplicar um one-hot encoding melhor para o modelo que iremos criar
print("Criando uma coluna com string ao invés de números para classificar o grau estudo:")
df_limpo_2['EDUCATION_CAT'] = 'none'

cat_mapping = {  # relação
      1: "graduate school",
      2: "university",
      3: "high school",
      4: "others"
}
df_limpo_2['EDUCATION_CAT'] = df_limpo_2['EDUCATION'].map(cat_mapping)  # .map -> mapeia os valores antigos e atribui aos novos os valores de acordo com o dicionario
print(df_limpo_2[['EDUCATION_CAT', 'EDUCATION']].head(10))

print("\nCodificao one-hot encoding:")
education_ohe = pd.get_dummies(df_limpo_2['EDUCATION_CAT'])  # get_dummies -> recebe uma coluna de DF e retorna um novo DF com um num igual de colunas e níveis de variável categórica. Variáveis dummy.
print(education_ohe.head(10))

print("\nConcatenando o DF de codificação one-hot com o original")
df_ohe = pd.concat(objs=[df_limpo_2, education_ohe], axis=1)  # axis=1 -> para que sejam concatenados horizontalmente, eixo da coluna.
print(df_ohe[['EDUCATION_CAT', 'graduate school', 'high school', 'university', 'others']].head(10))

df_ohe.to_csv('dados_explorados_e_limpos.csv')  # salvando o DF limpo e codificado em um arquivo CSV
