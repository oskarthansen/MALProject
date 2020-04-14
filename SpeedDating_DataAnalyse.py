# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:31:27 2020

@author: valde
"""
from lib.DataLoad import load_data
from lib.dataCleanUp import scaleGroup, replaceGroup
import pandas as pd
import numpy as np
raw_data = load_data()
data = raw_data.drop(['id', 'idg', 'partner', 'position', 'positin1', 'career', 'field', 'undergra', 'tuition', 'from', 'zipcode', 'income', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing','reading', 'tv', 'theater', 'movies','concerts', 'music', 'shopping', 'yoga', 'income', 'mn_sat' ], axis=1)
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
#Get all waves
wavesToScale = data[((data['wave'] >= 6) & (data['wave'] <= 9))]
allScaledColumns = pd.DataFrame(wavesToScale['wave'])

round_1_1 = ['attr1_1', "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]
columnsToScale = wavesToScale[round_1_1]
allScaledColumns = pd.concat([allScaledColumns, scaleGroup(columnsToScale, 100)], axis=1)

round_4_1 = ['attr4_1', "sinc4_1", "intel4_1", "fun4_1", "amb4_1", "shar4_1"]
columnsToScale = wavesToScale[round_4_1]
allScaledColumns = pd.concat([allScaledColumns, scaleGroup(columnsToScale, 100)], axis=1)

round_2_1 = ['attr2_1', "sinc2_1", "intel2_1", "fun2_1", "amb2_1", "shar2_1"]
columnsToScale = wavesToScale[round_2_1]
allScaledColumns = pd.concat([allScaledColumns, scaleGroup(columnsToScale, 100)], axis=1)

round_1_2 = ['attr1_2', "sinc1_2", "intel1_2", "fun1_2", "amb1_2", "shar1_2"]
columnsToScale = wavesToScale[round_1_2]
allScaledColumns = pd.concat([allScaledColumns, scaleGroup(columnsToScale, 100)], axis=1)

#data = pd.concat([allScaledColumns['wave'] >= 1 & (allScaledColumns['wave'] <= 5), [data['wave'] >= 6 & data['wave'] >= 9], allScaledColumns['wave'] >= 10 & allScaledColumns['wave'] <= 21], axis=0)


round_4_2 = ["attr4_2", "sinc4_2", "intel4_2", "fun4_2", "amb4_2", "shar4_2"]
columnsToScale = data[round_4_2]
scaledColumns = scaleGroup(columnsToScale, 100)
data = data[data.columns.drop(round_4_2)]
data = pd.concat([data, scaledColumns], axis=1)

round_2_2 = ["attr2_2", "sinc2_2", "intel2_2", "fun2_2", "amb2_2", "shar2_2"]
columnsToScale = data[round_2_2]
scaledColumns = scaleGroup(columnsToScale, 100)
data = data[data.columns.drop(round_2_2)]
data = pd.concat([data, scaledColumns], axis=1)

round_7_2 = ['attr7_2', "sinc7_2", "intel7_2", "fun7_2", "amb7_2", "shar7_2"]
columnsToScale = data[round_7_2]
scaledColumns = scaleGroup(columnsToScale, 100)
data = data[data.columns.drop(round_7_2)]
data = pd.concat([data, scaledColumns], axis=1)

attr_other = ["attr_o", "sinc_o", "intel_o", "fun_o", "amb_o", "shar_o"]

#%%Scale to 100 point scale 
round_3_1 = data[list(data.filter(regex="3_1"))]
round_3_1 = scaleGroup(round_3_1, 100)
data= replaceGroup(data, round_3_1)

round_5_1 = data[list(data.filter(regex="5_1"))]
round_5_1 = scaleGroup(round_5_1, 100)
data = replaceGroup(data, round_5_1)

round_3_s = data[list(data.filter(regex="3_s"))]
round_3_s = scaleGroup(round_3_s, 100)
data = replaceGroup(data, round_3_s)

score = data[["attr", "sinc", "intel", "fun", "amb", "shar"]]
score = scaleGroup(score, 100)
data = replaceGroup(data, score)

score_o = data[["attr_o", "sinc_o", "intel_o", "fun_o", "amb_o", "shar_o"]]
score_o = scaleGroup(score_o, 100)
data = replaceGroup(data, score_o)

#%%


#%% Set NaN to median values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(data)
data = pd.DataFrame(imputer.transform(data), columns=data.columns, index=data.index)






#%%Correlation bewteen what you see as important vs how you rate the other person and if this correlates to a match
self_look_for_before = data[['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']]
self_look_for_during_date = data[["attr1_s", "sinc1_s", "intel1_s", "fun1_s", "amb1_s", "shar1_s"]]
#self_look_for_after_date_1 = data[["attr1_2", "sinc1_2", "intel1_2", "fun1_2", "amb1_2", "shar1_2"]]

date_score = data[['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']]

diff_before =  self_look_for_before.values - date_score
diff_during = self_look_for_during_date.values - date_score
#diff_after_1 = self_look_for_after_date_1.values - date_score

def calcLength(row):
    return (row.values ** 2).mean() ** .5

lookfor_before_vs_datescore_diff = diff_before.apply(calcLength, axis=1)
lookfor_during_vs_datescore_diff = diff_during.apply(calcLength, axis=1)
#lookfor_after1_vs_datescore_diff = diff_after_1.apply(calcLength, axis=1)

#Invert scaling
lookfor_before_vs_datescore_diff = 100 - lookfor_before_vs_datescore_diff
lookfor_during_vs_datescore_diff = 100 - lookfor_during_vs_datescore_diff
#lookfor_after1_vs_datescore_diff = 100 - lookfor_after1_vs_datescore_diff

lookfor_before_vs_datescore_diff.name = "lookfor_before_vs_datescore_diff"
lookfor_during_vs_datescore_diff.name = "lookfor_during_vs_datescore_diff"
#lookfor_after1_vs_datescore_diff.name = "lookfor_after1_vs_datescore_diff"

lookfor_vs_datescore_diffs = pd.concat([lookfor_before_vs_datescore_diff, lookfor_during_vs_datescore_diff], axis=1)
data = pd.concat([data, pd.DataFrame(lookfor_vs_datescore_diffs)], axis=1)

corr = data.corr()
corr_dec = corr['dec'].sort_values(ascending=False)




#%%
#y = data['match']
#data = data.drop('match', axis=1)

#%%
#n_components = 0.80; #Preserve 95% variance
#pca = PCA(n_components=n_components)
#data_transformed = pca.fit_transform(data)



#%% For later use



round_3_2 = data[list(data.filter(regex="3_2"))]
data = replaceGroup(data, round_3_2)

round_5_2 = data[list(data.filter(regex="5_2"))]
data = replaceGroup(data, round_5_2)






