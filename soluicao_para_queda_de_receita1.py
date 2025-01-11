import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Carregar os dados
ds_dados_clientes = pd.read_csv("C:/Users/Pichau/Desktop/estudo de caso ciencia de dados/Estudo de Caso ATDBank/ds_dados_clientes.csv", encoding='iso-8859-1')
print(ds_dados_clientes.info())

# Tratamento dos dados faltantes
print((1 - (ds_dados_clientes.isnull().sum() / len(ds_dados_clientes['POSSE']))) * 100)
ds_dados_clientes = ds_dados_clientes.dropna(subset=['LIMITE_CREDITO'])
ds_dados_clientes = ds_dados_clientes.dropna(subset=['PAGAMENTO_MINIMO'])
print(ds_dados_clientes.info())
ds_dados_clientes = ds_dados_clientes.drop(['CLI_ID'], axis=1)

# Verificando a existência de NaNs
print(ds_dados_clientes.isnull().sum())

# Se houver NaNs, podemos imputar ou remover
# Vamos usar imputação com a média para os valores ausentes:
imputer = SimpleImputer(strategy='mean')
ds_dados_clientes = pd.DataFrame(imputer.fit_transform(ds_dados_clientes), columns=ds_dados_clientes.columns)

# Normalização
normalizador = StandardScaler()
ds_dados_clientes_normalizados = normalizador.fit_transform(ds_dados_clientes)

# Aplicando PCA para reduzir para 1 componente
saldo_e_compra_unica = PCA(n_components=1)
pca_result = saldo_e_compra_unica.fit_transform(ds_dados_clientes[['COMPRAS', 'COMPRA_UNICA_VEZ']].dropna())

# Converter o resultado do PCA em DataFrame
pca_result = pd.DataFrame(pca_result, columns=['PCA_COMPRAS_COMPRA_UNICA'])

# Remover as colunas originais
dataset = ds_dados_clientes.drop(['COMPRAS', 'COMPRA_UNICA_VEZ'], axis=1)

# Concatenar os resultados transformados
dataset = pd.concat([dataset, pca_result], axis=1)

# Normalizando o dataset
dataset = pd.DataFrame(normalizador.fit_transform(dataset), columns=dataset.columns)

# Verifique o tipo de dataset após a normalização
print(type(dataset))  # Verifica se é um DataFrame ou ndarray

# Agora, você pode usar dataset.info() sem problemas
print(dataset.info())

# Clusterização do dataset
modelo = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=10, random_state=45)
modelo.fit(dataset)
y = modelo.predict(dataset)
y = pd.Series(y)
y = y.astype(str)
y = y.replace({'0': 'A', '1': 'B'})  # Tipo de clientes A, B e C

# Concatenando a coluna 'Categoria_de_cliente' de volta ao dataset
dataset['Categoria_de_cliente'] = y

print(dataset.head(20))
print(dataset.info())
print(dataset['Categoria_de_cliente'].value_counts())

# Calcular o Silhouette Score
score = silhouette_score(dataset.drop('Categoria_de_cliente', axis=1), y)
print(f'Silhouette Score: {score}')  # Pontuação que mede a qualidade da separação entre os clusters

# Identificando correlação entre as variáveis
plt.figure(figsize=(20, 20))
sns.heatmap(dataset.drop('Categoria_de_cliente', axis=1).corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()

# ANOVA para verificar a relação entre a variável PCA_COMPRAS_COMPRA_UNICA e a Categoria de Cliente
f_value, p_value = stats.f_oneway(
    dataset[dataset['Categoria_de_cliente'] == 'A']['PCA_COMPRAS_COMPRA_UNICA'].dropna(),
    dataset[dataset['Categoria_de_cliente'] == 'B']['PCA_COMPRAS_COMPRA_UNICA'].dropna(),
  )

print(f"f_value: {f_value}\np_value: {p_value}")
# Como p_value é muito menor que 0.05, isso significa que os clusters A, B e C apresentam comportamentos muito diferentes em relação ao PCA_COMPRAS_COMPRA_UNICA

# Boxplot para verificar a relação entre 'SALDO' e 'Categoria_de_cliente'
sns.boxplot(x='Categoria_de_cliente', y='SALDO', data=dataset)
plt.show()

# Dividindo o dataset em treino e teste
dataset.dropna(inplace=True)
print(dataset.info())

x = dataset.drop(['Categoria_de_cliente'], axis=1)
y = dataset['Categoria_de_cliente']

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=42)

# Definindo parâmetros para GridSearchCV (caso queira usar)
parametros = {'max_depth': np.array([5, 8]), 'min_samples_split': np.array([1, 2]),
              'min_samples_leaf': np.array([2, 4]), 'max_features': ['sqrt', 'log2']}

# Usando ExtraTreesClassifier para prever as categorias de clientes
modelo = ExtraTreesClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=2, n_estimators=100)
modelo.fit(x_treino, y_treino)
acuracia = modelo.score(x_teste, y_teste)

# Caso queira utilizar GridSearchCV para otimização de parâmetros:
# gridSearchCV = GridSearchCV(estimator=modelo, param_grid=parametros, cv=5, n_jobs=-1, scoring='accuracy')
# gridSearchCV.fit(x_treino, y_treino)

print(f"Acurácia do modelo: {acuracia}")  # Acurácia do modelo ExtraTreesClassifier de 0.95

# Calcular médias de cada variável por cluster
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
cluster_means = dataset.groupby('Categoria_de_cliente').mean()
print(cluster_means)

'''Estratégia para o Cluster A (Clientes mais ativos/risco mais elevado):
Ofertas de crédito personalizado: Como o cluster A tem um comportamento de compras mais frequentes e parceladas, pode ser interessante oferecer produtos financeiros, como cartões de crédito com limites mais altos, financiamento de compras ou parcelamento sem juros, com taxas atrativas.


Programas de fidelidade e descontos: Focar em programas que incentivem a frequência de compras. Oferecer descontos e vantagens exclusivas pode incentivar esse grupo a continuar comprando e aumentando seu envolvimento com o banco.


Monitoramento de risco: Dado o uso intenso de crédito, seria importante oferecer serviços de acompanhamento do saldo e do limite de crédito, além de estratégias de educação financeira para prevenir inadimplência.


Aprimoramento do relacionamento: Enviar ofertas de produtos financeiros como empréstimos ou seguros personalizados, já que este grupo pode estar em busca de mais serviços de crédito.


Estratégia para o Cluster B (Clientes mais conservadores/baixo risco):
Ofertas de produtos de baixo risco: Como este grupo tem menor frequência de compras e usa menos crédito, seria interessante oferecer opções mais conservadoras de produtos bancários, como contas de poupança, investimentos seguros ou cartões de crédito com limites mais baixos.


Ações de educação financeira: Como o saldo é maior e há menos uso de crédito, a educação financeira pode ser importante. Estratégias que promovam o entendimento sobre o uso do crédito e a promoção de economias seriam bem-vindas.


Promoções para aumentar o uso do crédito: Embora o grupo seja mais conservador, eles podem ser incentivados a usar o crédito de maneira mais estratégica. Estratégias de marketing focadas em consumo inteligente, como "Pague em até 3x sem juros", podem ser um bom estímulo.


Incentivar o uso digital: Oferecer incentivos para o uso de canais digitais como aplicativo de banco e internet banking para facilitar o dia a dia desse público, especialmente com foco em ações que envolvem saques e transferências.
'''




