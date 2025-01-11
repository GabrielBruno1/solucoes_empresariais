import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats  # par a estrategian de analise de variação das variáveis ANOVA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.decomposition import PCA


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
ds_analise_credito = pd.read_csv('ds_analise_credito.csv',  encoding='iso-8859-1')
print(ds_analise_credito.head(20))
print(ds_analise_credito.info())

#tratando das variaveis
'''1.removi algumas variáveis pois acho que não está relacionado com o problema, removi o Score pois pelo que eu entendi
 ele está errado de acordo com o estudo de caso'''
ds_analise_credito = ds_analise_credito.drop(['TRABALHADOR_ESTRANGEIRO', 'IDADE', 'TELEFONE', 'GENERO_ESTADO_CIVIL', 'EMPREGADO_DESDE'
                                              ,'ANOS_RESIDENCIA', 'BENS', 'HABITACAO', 'TRABALHO', 'OUTROS_EMPRESTIMOS'
                                              ,'DEVEDOR_FIADOR', 'Number of people being liable to provide maintenance for', 'Score'], axis=1)

'''2.algumas variaveis preditivas apresentam valores do tipo string, logo utilizarei o hot code para eu conseguir utilizar 
na clusterização e depois no modelo de previsão. obs: algumas variaveis eu decidi não utilizar hot code e sim apenas excluir
como é o caso de POUPANCA e PROPOSITO'''
ds_analise_credito_1 = ds_analise_credito.copy()
unicode =  pd.get_dummies(ds_analise_credito['SITUACAO_CC'])
ds_analise_credito = pd.concat([ds_analise_credito, unicode], axis=1)
ds_analise_credito = ds_analise_credito.drop('SITUACAO_CC', axis=1)
ds_analise_credito = ds_analise_credito.rename(columns={'situacao cc 0 - 200':' 0 - 200', 'Acima de 200':'situacao cc  Acima de 200', 'Negativa': 'situacao cc  Negativa'})

unicode = pd.get_dummies(ds_analise_credito['HISTORICO_CREDITO'])
ds_analise_credito = pd.concat([ds_analise_credito, unicode], axis=1)
ds_analise_credito = ds_analise_credito.drop('HISTORICO_CREDITO', axis=1)

'''unicode = pd.get_dummies(ds_analise_credito['PROPOSITO'])
ds_analise_credito = pd.concat([ds_analise_credito, unicode], axis=1)'''
ds_analise_credito = ds_analise_credito.drop('PROPOSITO', axis=1)

'''unicode = pd.get_dummies(ds_analise_credito['POUPANCA'])
ds_analise_credito = pd.concat([ds_analise_credito, unicode], axis=1)'''
ds_analise_credito = ds_analise_credito.drop('POUPANCA', axis=1)
ds_analise_credito = ds_analise_credito.astype(int)



'''3.normalizei os dados para eu conseguir utilizar de maneira adequada o método de clusterização Kmeans
não normalizei algumas variaveis, pois tais variaveis preiditvas são classificatórios do tipo 0 e 1'''
normalizador = StandardScaler()
ds_analise_credito_continuas = pd.DataFrame(normalizador.fit_transform(ds_analise_credito[['TAXA_PAGTO', 'CREDITOS_EXISTENTES', 'TOTAL_CREDITO', 'DURACAO_EMPRESTIMO']]), columns=['TAXA_PAGTO', 'CREDITOS_EXISTENTES', 'TOTAL_CREDITO', 'DURACAO_EMPRESTIMO'])
ds_analise_credito = pd.concat([ds_analise_credito.drop(['TAXA_PAGTO', 'CREDITOS_EXISTENTES', 'TOTAL_CREDITO', 'DURACAO_EMPRESTIMO'], axis=1), ds_analise_credito_continuas], axis=1)
print(ds_analise_credito.info())



"redimensionando preditivas com alto relacionamento."
pca = PCA(n_components=1)
total_credito_e_duracao_emprestimo = pca.fit_transform(ds_analise_credito[['TOTAL_CREDITO', 'DURACAO_EMPRESTIMO']])
pca_result = pd.DataFrame(total_credito_e_duracao_emprestimo, columns=['TOTAL_CREDITO_Duracao_emprestimo_pca'])
ds_analise_credito = ds_analise_credito.drop(['TOTAL_CREDITO', 'DURACAO_EMPRESTIMO'], axis=1)
ds_analise_credito = pd.concat([ds_analise_credito, pca_result], axis=1)

print(ds_analise_credito.corr())
plt.figure(figsize=(20,20))
sns.heatmap(ds_analise_credito.corr(), annot=True,  cmap='coolwarm', linewidths=0.5)
plt.show()

'''5.clusterizei os dados com Kmeans paar separar eles entre Bom pagador e pagador ruim'''
kmeans = KMeans(n_clusters=2, random_state=10)
kmeans.fit(ds_analise_credito)
score = kmeans.predict(ds_analise_credito)
score = pd.Series(score)
score = score.astype(str)
acuracia = silhouette_score(ds_analise_credito, score)
score = score.replace({'0': 'Ruim', '1':'Bom'})

print(acuracia)
ds_analise_credito['Score'] = score
print(ds_analise_credito['Score'].value_counts())




'''6. aplicando a técnica de analise de variaça ANOVA para verificar se há diferença significativa entre as médias,
ou seja, verificar se eles estão ou não bem separados um do outro'''
f_value, p_value = stats.f_oneway(ds_analise_credito[ds_analise_credito["Score"]=='Ruim']["TOTAL_CREDITO_Duracao_emprestimo_pca"],
                                  ds_analise_credito[ds_analise_credito["Score"]=='Bom']["TOTAL_CREDITO_Duracao_emprestimo_pca"])

print(f"f_value: {f_value}\np_value: {p_value}")

x = ds_analise_credito.drop('Score', axis=1)
y = ds_analise_credito['Score']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

'''7. fiz testes de combinações de parametros e seus valores com, GridSearchCV para melhorar a acuracia do modelo
ficando os melhores como: class_weight= 'balanced', max_depth = None, min_samples_leaf= 1, min_samples_split= 2, n_estimators=50'''
modelo = ExtraTreesClassifier(class_weight= 'balanced', max_depth = None, min_samples_leaf= 1, min_samples_split= 2, n_estimators=50)
parametros = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]  # Para lidar com desbalanceamento
}


'''8. foi utilizado validação cruzada para verificação da acuracia'''
kfold = KFold(n_splits=20, random_state=42, shuffle=True)
resultado = cross_val_score(modelo, x,y, cv=kfold)
print('resultdo da validação cruzada com 20 kfolds : ', resultado.mean())
modelo.fit(x_train, y_train)
resultado = modelo.score(x_test, y_test)
print(f"Acurácia dos dados de treino: {modelo.score(x_train, y_train)}\nAcurácia dos dados de teste: {modelo.score(x_test, y_test)}")
from sklearn.metrics import classification_report


'''9. relatório de métricas para confirmar as avaliações de acuracia anteriores'''
# Previsões
y_pred = modelo.predict(x_test)

# Relatório de métricas
print('Relatório de métricas: ',classification_report(y_test, y_pred))




importances = modelo.feature_importances_
feature_names = x_train.columns

# Visualizar as importâncias
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)
print(ds_analise_credito.groupby('Score').mean())


# Exibir os resultados corrigidos

ds_analise_credito_1['Score'] = score
ds_analise_credito.to_csv('ds_analise_credito_1.csv', index=False, na_rep='NaN')

"""output final:
                                              Feature  Importance
11               TOTAL_CREDITO_Duracao_emprestimo_pca    0.800025
9                                          TAXA_PAGTO    0.066609
10                                CREDITOS_EXISTENTES    0.036264
6     créditos existentes pagos devidamente até agora    0.017733
4                      atraso no pagamento no passado    0.016233
0                                             0 - 200    0.012716
5   conta crítica/outros créditos existentes (não ...    0.010018
1                           situacao cc  Acima de 200    0.009691
2                               situacao cc  Negativa    0.009386
7   nenhum crédito recebido / todos os créditos pa...    0.008533
3                                       Não Possui CC    0.007254
8   todos os créditos neste banco foram devidament...    0.005539
        0 - 200  situacao cc  Acima de 200  situacao cc  Negativa  \
Score                                                               
Bom    0.240157                   0.074803               0.275591   
Ruim   0.361345                   0.025210               0.268908   

       Não Possui CC  atraso no pagamento no passado  \
Score                                                  
Bom         0.409449                        0.062992   
Ruim        0.344538                        0.168067   

       conta crítica/outros créditos existentes (não neste banco)  \
Score                                                               
Bom                                             0.299213            
Ruim                                            0.273109            

       créditos existentes pagos devidamente até agora  \
Score                                                    
Bom                                           0.566929   
Ruim                                          0.411765   

       nenhum crédito recebido / todos os créditos pagos devidamente  \
Score                                                                  
Bom                                             0.023622               
Ruim                                            0.092437               

       todos os créditos neste banco foram devidamente pagos  TAXA_PAGTO  \
Score                                                                      
Bom                                             0.047244        0.086351   
Ruim                                            0.054622       -0.276468   

       CREDITOS_EXISTENTES  TOTAL_CREDITO_Duracao_emprestimo_pca  
Score                                                             
Bom              -0.054856                             -0.584864  
Ruim              0.175631                              1.872547  """






