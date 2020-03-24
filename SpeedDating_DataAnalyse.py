# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:31:27 2020

@author: valde
"""
from DataLoad import load_data
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
raw_data = load_data()
data = raw_data.drop(['iid', 'id', 'idg', 'partner', 'pid', 'position', 'positin1', 'career', 'field', 'undergra', 'tuition', 'from', 'zipcode', 'income', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing','reading', 'tv', 'theater', 'movies','concerts', 'music', 'shopping', 'yoga', 'income', 'mn_sat' ], axis=1)
#data = data[data.columns.drop(list(data.filter(regex='_3')))]
#%%One hot encoding
data = data[data.columns.drop(list(data.filter(regex="_3")))]

data.drop(["gender", "race_o"], axis=1)

field_1hot = pd.get_dummies(data['field_cd'], prefix= 'field') #Encode fields
data = data.drop('field_cd', axis=1)
data = pd.concat([data, field_1hot], axis=1)

race_1hot = pd.get_dummies(data['race'], prefix='race')
data = data.drop('race', axis=1)
data = pd.concat([data, race_1hot], axis=1)

goal_1hot = pd.get_dummies(data['goal'], prefix='goal')
data = data.drop('goal', axis=1)
data = pd.concat([data, goal_1hot], axis=1)

career_1hot = pd.get_dummies(data['career_c'], prefix='career')
data = data.drop('career_c', axis=1)
data = pd.concat([data, career_1hot], axis=1)

date = data['date']
date = np.abs(8 - date)
data = data.drop('date', axis=1)
data = pd.concat([data, date], axis=1)

go_out = data['go_out']
go_out = np.abs(8-go_out)
data = data.drop('go_out', axis=1)
data = pd.concat([data, go_out], axis=1)




#%%Scale values from wave 1-5 and 10-21 to fit values to 1-10 scale for given metrics
def scaleGroup(group, upperBound, scalar):
    maxValue = 100/len(group.columns)
    scaledGroup = group.divide(maxValue) * scalar
    return scaledGroup.clip(upper=upperBound)
#Get all waves
wavesToScale = data[((data['wave'] >= 1) & (data['wave'] <= 5)) | ((data['wave'] >= 10) & (data['wave'] <= 21))]
allScaledColumns = pd.DataFrame(wavesToScale['wave'])

round_1_1 = ['attr1_1', "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]
columnsToScale = wavesToScale[round_1_1]
allScaledColumns = pd.concat([allScaledColumns, scaleGroup(columnsToScale, 10, 5)], axis=1)

round_4_1 = ['attr4_1', "sinc4_1", "intel4_1", "fun4_1", "amb4_1", "shar4_1"]
columnsToScale = wavesToScale[round_4_1]
allScaledColumns = pd.concat([allScaledColumns, scaleGroup(columnsToScale, 10, 5)], axis=1)

round_2_1 = ['attr2_1', "sinc2_1", "intel2_1", "fun2_1", "amb2_1", "shar2_1"]
columnsToScale = wavesToScale[round_2_1]
allScaledColumns = pd.concat([allScaledColumns, scaleGroup(columnsToScale, 10, 5)], axis=1)

round_1_2 = ['attr1_2', "sinc1_2", "intel1_2", "fun1_2", "amb1_2", "shar1_2"]
columnsToScale = wavesToScale[round_1_2]
allScaledColumns = pd.concat([allScaledColumns, scaleGroup(columnsToScale, 10, 5)], axis=1)

#data = pd.concat([allScaledColumns['wave'] >= 1 & (allScaledColumns['wave'] <= 5), [data['wave'] >= 6 & data['wave'] >= 9], allScaledColumns['wave'] >= 10 & allScaledColumns['wave'] <= 21], axis=0)


round_4_2 = ["attr4_2", "sinc4_2", "intel4_2", "fun4_2", "amb4_2", "shar4_2"]
columnsToScale = data[round_4_2]
scaledColumns = scaleGroup(columnsToScale, 10, 5)
data = data[data.columns.drop(round_4_2)]
data = pd.concat([data, scaledColumns], axis=1)

round_2_2 = ["attr2_2", "sinc2_2", "intel2_2", "fun2_2", "amb2_2", "shar2_2"]
columnsToScale = data[round_2_2]
scaledColumns = scaleGroup(columnsToScale, 10, 5)
data = data[data.columns.drop(round_2_2)]
data = pd.concat([data, scaledColumns], axis=1)

round_7_2 = ['attr7_2', "sinc7_2", "intel7_2", "fun7_2", "amb7_2", "shar7_2"]
columnsToScale = data[round_7_2]
scaledColumns = scaleGroup(columnsToScale, 10, 5)
data = data[data.columns.drop(round_7_2)]
data = pd.concat([data, scaledColumns], axis=1)



#columnsToScale = wavesToScale[['attr1_1', "sinc1_1",]]

#%% Set NaN to median values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(data)
data = pd.DataFrame(imputer.transform(data), columns=data.columns, index=data.index)




#%%Correlation bewteen what you see as important vs how you rate the other person and if this correlates to a match
self_look_for_before = data[['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']]
self_look_for_during_date = data[["attr1_s", "sinc1_s", "intel1_s", "fun1_s", "amb1_s", "shar1_s"]]
self_look_for_after_date_1 = data[["attr1_2", "sinc1_2", "intel1_2", "fun1_2", "amb1_2", "shar1_2"]]

date_score = data[['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']]

diff_before =  self_look_for_before.values - date_score
diff_during = self_look_for_during_date.values - date_score
diff_after_1 = self_look_for_after_date_1.values - date_score

def calcRowRMSE(row):
    return (row.values ** 2).mean() ** .5

rmse_before = diff_before.apply(calcRowRMSE, axis=1)
rmse_during = diff_during.apply(calcRowRMSE, axis=1)
rmse_after_1 = diff_after_1.apply(calcRowRMSE, axis=1)

rmse_before = 100 - rmse_before
rmse_during = 100 - rmse_during
rmse_after_1 = 100 - rmse_after_1

rmse_before.name = "rmse_before"
rmse_during.name = "rmse_during"
rmse_after_1.name = "rmse_after_1"

rmse = pd.concat([rmse_before, rmse_during, rmse_after_1], axis=1)
data = pd.concat([data, pd.DataFrame(rmse)], axis=1)

corr = data.corr()
corr_dec = corr['dec'].sort_values(ascending=False)




#%%
#y = data['match']
#data = data.drop('match', axis=1)

#%%
#n_components = 0.80; #Preserve 95% variance
#pca = PCA(n_components=n_components)
#data_transformed = pca.fit_transform(data)



#%%







