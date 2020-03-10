# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:31:27 2020

@author: valde
"""
from DataLoad import load_data
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
raw_data = load_data("Speed_Dating_Data.csv")
data = raw_data.drop(['iid', 'id', 'idg', 'partner', 'pid', 'position', 'positin1', 'career', 'field', 'undergra', 'tuition', 'from', 'zipcode', 'income', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing','reading', 'tv', 'theater', 'movies','concerts', 'music', 'shopping', 'yoga', 'income', 'mn_sat' ], axis=1)
#data = data[data.columns.drop(list(data.filter(regex='_3')))]
#%%One hot encoding
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

#%% Set NaN to median values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(data)
data = pd.DataFrame(imputer.transform(data), columns=data.columns, index=data.index)

#%%


#%%
#y = data['match']
#data = data.drop('match', axis=1)

#%%
n_components = 0.80; #Preserve 95% variance
pca = PCA(n_components=n_components)
data_transformed = pca.fit_transform(data)





#%%
corr = data.corr()
corr_match = corr['match'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
attributes = corr_match[20:23].index
scatter_matrix(data[attributes], figsize=(12,8))





