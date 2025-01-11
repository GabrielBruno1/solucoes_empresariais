import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


ds_dados_clientes = pd.read_csv("C:/Users/Pichau/Desktop/estudo de caso ciencia de dados/Estudo de Caso ATDBank/ds_dados_clientes.csv", encoding='iso-8859-1')
print(ds_dados_clientes.info())

#tratamento dos dados faltantes
print((1 - (ds_dados_clientes.isnull().sum() / len(ds_dados_clientes['POSSE'])))*100)
ds_dados_clientes = ds_dados_clientes.dropna(subset=['LIMITE_CREDITO'])
ds_dados_clientes = ds_dados_clientes.dropna(subset=['PAGAMENTO_MINIMO'])
print(ds_dados_clientes.info())

#redimensionando dataset
normalizador = StandardScaler()
ds_dados_clientes = ds_dados_clientes.drop(['CLI_ID'], axis=1)
ds_dados_clientes_normalizados = normalizador.fit_transform(ds_dados_clientes)

#clusterização do dataset
modelo = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=10, random_state=45)
modelo.fit(ds_dados_clientes_normalizados)
y = modelo.predict(ds_dados_clientes_normalizados)
y = pd.Series(y)
y = y.astype(str)
y = y.replace({'0': 'A', '1': 'B', '2': 'C'}) # tipo de clientes A, B e C


dataset = pd.concat([ds_dados_clientes, y], axis=1)
dataset = dataset.rename(columns={0: 'Categoria_de_cliente'})
print(dataset.head(20))
print(dataset.info())
print(dataset['Categoria_de_cliente'].value_counts())

score = silhouette_score(ds_dados_clientes_normalizados, y)
print(f'Silhouette Score: {score}') #pontuação que mede a qualidade da separação entre os clusters OUTPUT: o maior valor é de 0.24 para n_clusters=3
'''============================================================================================='''

#dataset.to_csv('dados_cliente.csv', index=False)

#identificando correlação entre as variáveis
plt.figure(figsize=(20,20))
# Aplicar PCA para reduzir para 1 componente
saldo_e_compra_unica = PCA(n_components=1)
pca_result = saldo_e_compra_unica.fit_transform(dataset[['COMPRAS', 'COMPRA_UNICA_VEZ']].dropna())

# Converter o resultado do PCA em DataFrame
pca_result = pd.DataFrame(pca_result, columns=['PCA_COMPRAS_COMPRA_UNICA'])

# Remover as colunas originais
dataset = dataset.drop(['COMPRAS', 'COMPRA_UNICA_VEZ'], axis=1)

# Concatenar os resultados transformados
dataset = pd.concat([dataset.reset_index(drop=True), pca_result], axis=1)
sns.heatmap(dataset.drop('Categoria_de_cliente', axis=1).corr(), annot=True, cmap='coolwarm', linewidths=0.5)

plt.show()
#como quero saber da correlação de cada variavel em relação a minha variavel categorica então usarei a a analise de variança anova:
f_value, p_value = stats.f_oneway(
    dataset[dataset['Categoria_de_cliente'] == 'A']['PCA_COMPRAS_COMPRA_UNICA'].dropna(),
    dataset[dataset['Categoria_de_cliente'] == 'B']['PCA_COMPRAS_COMPRA_UNICA'].dropna(),
    dataset[dataset['Categoria_de_cliente'] == 'C']['PCA_COMPRAS_COMPRA_UNICA'].dropna())

print(f"f_value: {f_value}\np_value: {p_value}")
'''OUTPUT: f_value: 7.826354032419173 p_value: 0.0004020173339950085
p_value sendo menor que 0.05 isso significa que os cluster A,B e C apresentam comportamnetos muito diferentes em rerlação ao PCA_COMPRAS_COMPRA_UNICA
'''

''' #como p_value é muito menor que 0.05, isso significa que a coluna SALDO tem uma relação significativa com 'Categoria_de_cliente' '''
sns.boxplot(x='Categoria_de_cliente', y='SALDO', data=dataset)
plt.show()



dataset.dropna(inplace=True)
print(dataset.info())
x = dataset.drop(['Categoria_de_cliente'], axis=1)
y = dataset['Categoria_de_cliente']
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.3, random_state=42)


parametros = {'max_depth': np.array([5,8]), 'min_samples_split': np.array([1,2]),
            'min_samples_leaf':np.array([2,4]), 'max_features':['sqrt', 'log2'],}

modelo = ExtraTreesClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=2, n_estimators=100)
modelo.fit(x_treino, y_treino)
acuracia = modelo.score(x_teste, y_teste)
'''gridSearchCV = GridSearchCV(estimator=modelo, param_grid=parametros, cv=5, n_jobs=-1, scoring='accuracy')
gridSearchCV.fit(x_treino, y_treino)'''

print(acuracia) #output: 0.67
dataset.to_csv('cl1.csv', index=False)


