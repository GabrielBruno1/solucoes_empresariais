import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.feature_selection import chi2, SelectKBest
import numpy as np
import seaborn as sns

#1. formatação dos tipos de dados
ds_gestao_pessoas = pd.read_csv("C:/Users/Pichau/Desktop/estudo de caso ciencia de dados/Estudo de Caso ATDBank/ds_gestao_pessoas.csv", encoding='iso-8859-1')

#2. tratamento dos dados outliers
"""ds_gestao_pessoas.boxplot(column=[''])
"""
#3. exclusão de variaveis

ds_gestao_pessoas = ds_gestao_pessoas.drop(['NumeroEmpregado','HorarioPadrao', 'Acima18', 'ContagemEmpregados'], axis=1)

print(ds_gestao_pessoas[ds_gestao_pessoas['Cargo'] == 'Research Scientist']['SaiuDaEmpresa'].value_counts())


print(ds_gestao_pessoas["SaiuDaEmpresa"].value_counts())


# passando 'ViagensNegocio' para unicode
'''unicode = pd.get_dummies(ds_gestao_pessoas['ViagensNegocio']).astype(int)
ds_gestao_pessoas = pd.concat([ds_gestao_pessoas, unicode], axis=1)'''
ds_gestao_pessoas = ds_gestao_pessoas.drop(['ViagensNegocio'], axis=1)

# passando 'Departamento' para unicode
'''unicode = pd.get_dummies(ds_gestao_pessoas['Departamento']).astype(int)
ds_gestao_pessoas = pd.concat([ds_gestao_pessoas, unicode], axis=1)'''

# passando 'CampoEducacao' para unicode
'''unicode = pd.get_dummies(ds_gestao_pessoas['CampoEducacao']).astype(int)
ds_gestao_pessoas = pd.concat([ds_gestao_pessoas, unicode], axis=1)'''
ds_gestao_pessoas = ds_gestao_pessoas.drop(['CampoEducacao'], axis=1)



# passando 'StatusCivil' para unicode
'''unicode = pd.get_dummies(ds_gestao_pessoas['StatusCivil']).astype(int)
ds_gestao_pessoas = pd.concat([ds_gestao_pessoas, unicode], axis=1)'''
ds_gestao_pessoas = ds_gestao_pessoas.drop(['StatusCivil'], axis=1)

ds_gestao_pessoas['Genero'] = ds_gestao_pessoas['Genero'].replace({'Female': 0, 'Male': 1})
ds_gestao_pessoas['Genero'] = ds_gestao_pessoas['Genero'].astype(int)

ds_gestao_pessoas['HoraExtra'] = ds_gestao_pessoas['HoraExtra'].replace({'No': 0, 'Yes': 1})
ds_gestao_pessoas['HoraExtra'] = ds_gestao_pessoas['HoraExtra'].astype(int)

ds_gestao_pessoas['SaiuDaEmpresa'] = ds_gestao_pessoas['SaiuDaEmpresa'].replace({'No': 0, 'Yes': 1})
ds_gestao_pessoas['SaiuDaEmpresa'] = ds_gestao_pessoas['SaiuDaEmpresa'].astype(int)



# passando 'Cargo' para unicode
'''unicode = pd.get_dummies(ds_gestao_pessoas['Cargo']).astype(int)
ds_gestao_pessoas = pd.concat([ds_gestao_pessoas, unicode], axis=1)'''
ds_LaboratoryTechnician = ds_gestao_pessoas[ds_gestao_pessoas['Departamento']=='Research & Development']
ds_LaboratoryTechnician  = ds_LaboratoryTechnician.drop(['Departamento', 'Cargo'], axis=1)
ds_gestao_pessoas = ds_gestao_pessoas.drop(['Departamento', 'Cargo'], axis=1)
print(ds_LaboratoryTechnician.info())

'''ds_LaboratoryTechnician = ds_LaboratoryTechnician.drop(['Cargo'], axis=1)
ds_gestao_pessoas = ds_gestao_pessoas.drop(['Cargo'], axis=1)'''
correlacao = ds_gestao_pessoas.corr()

correlacao1 = ds_LaboratoryTechnician.corr()
print(correlacao1['SaiuDaEmpresa'])
sns.heatmap(correlacao1)
plt.show()





print(ds_gestao_pessoas.info())

#definição das preditivas e target
x = ds_gestao_pessoas.drop(['SaiuDaEmpresa', 'AumentoSalarialPercentual', 'Idade','ValorDia',
                            'NumEmpresaTrabalhou', 'ValorHora', 'Educacao', 'DistanciaDeCasa', 'NivelTrabalho', 'AnosNaEmpresa',
                            'AnosFuncaoAtual', 'AnosDesdeUltimaPromocao', 'TreinamentosRealizadosUltimoAno', 'TotalAnosTrabalho',
                            'NivelOpcaoAcoes', 'ClassificacaoDesempenho', 'ValoeMes', 'Genero',
                            'EnvolvimentoTrabalho', 'SatisfacaoTrabalho', 'SatisfacaoRelacionamento'
                          , 'RendaMensal', 'AnosComAtualGerente', 'SatisfacaoComAmbiente', 'EquilibrioVidaProfissional'
                               ], axis=1)
y = ds_gestao_pessoas['SaiuDaEmpresa']
#escolhendo as k melhores varivaveis preditivas
algoritmo = SelectKBest(chi2, k=34)
selectedKBest = algoritmo.fit_transform(x, y)

indices = algoritmo.get_support(2)
lista_de_var_exclud = []
for indice in indices:
    lista_de_var_exclud.append(x.columns[indice])


print(lista_de_var_exclud)


#x = ds_gestao_pessoas[['AumentoSalarialPercentual', 'Research Scientist']]



# O AumentoSalarialPercentual é o principal fator que resulta na alta rotatividade
#x = ds_gestao_pessoas[['Married']]
# qual seria o AumentoSalarialPercentualIdeal para diminuir a rotatividade?



parametros = {'max_depth': np.array([4,5,7,10]), 'min_samples_split': np.array([2,4,6]),
'criterion': ['gini', 'entropy', 'log_loss'], 'min_samples_leaf': np.array([1,2,3,4])}


#otimização dos parametros do modelo
modelo = ExtraTreesClassifier(max_depth=10, min_samples_split=2,criterion='log_loss', min_samples_leaf=3, class_weight='balanced')
"""gridSearchCV = GridSearchCV(estimator=modelo, param_grid=parametros, cv=5, n_jobs=-1)
gridSearchCV.fit(x, y)

print(f'"max_depth":{gridSearchCV.best_estimator_.max_depth}, "min_samples_split":{gridSearchCV.best_estimator_.min_samples_split},'
      f'"criterion":{gridSearchCV.best_estimator_.criterion}, "min_samples_leaf":{gridSearchCV.best_estimator_.min_samples_leaf}')"""
#"max_depth":10, "min_samples_split":2,"criterion":log_loss, "min_samples_leaf":3

print(x.info())
'''#validação cruzada
stratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True)
acuracia = cross_val_score(modelo, x, y, cv=stratifiedKFold, scoring='accuracy', n_jobs=-1)'''

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.3, random_state=42)
modelo.fit(x_treino, y_treino)


#print(acuracia.mean())

#0.8605442176870748 verificar qual variavel q   ue agrega mais peso na previsão para resultar nessa probabilidade de 86%
print(correlacao['SaiuDaEmpresa'])
print(modelo.score(x_teste, y_teste))
#print(modelo.predict([[10500]])) # a rotatividade tende a cair quando a RendaMensal é igual ou superior a 10500

'''Os fatores que parecem estar mais associados a uma alta rotatividade na empresa incluem:

Horas extras: Funcionários sobrecarregados com horas extras podem buscar alternativas de emprego.
Satisfação com o ambiente de trabalho e com o trabalho em si: A falta de satisfação com o ambiente e o trabalho são fatores importantes para a saída.
Equilíbrio entre vida pessoal e profissional: Um equilíbrio insatisfatório pode também estar contribuindo para o aumento da rotatividade.'''

'''A Solução para a rotatividade é diminuir as horas extras dando mais satisfação com o ambiente de trabalho e equilibrio entre vida profissional
e vida pessoal'''

