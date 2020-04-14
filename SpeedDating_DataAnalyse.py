# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:31:27 2020

@author: valde
"""
from lib.DataLoad import load_data
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


#%%Scale values from wave 1-5 and 10-21 to fit values to 1-10 scale for given metrics
def scaleGroup(group, upperBound, scalar):
    maxValue = 100/len(group.columns)
    scaledGroup = group.divide(maxValue) * scalar
    return scaledGroup.clip(upper=upperBound)
#Get all waves
wavesToScale = data[((data['wave'] >= 1) & (data['wave'] <= 5)) | ((data['wave'] >= 10) & (data['wave'] <= 21))]
allScaledColumns = pd.DataFrame()

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



round_4_2 = ["attr4_2", "sinc4_2", "intel4_2", "fun4_2", "amb4_2", "shar4_2"]
columnsToScale = data[round_4_2]
scaledColumns = scaleGroup(columnsToScale, 10, 5)
data.drop(round_4_2)
data = pd.concat([data, scaledColumns], axis=1)

round_2_2 = ["attr2_2", "sinc2_2", "intel2_2", "fun2_2", "amb2_2", "shar2_2"]
columnsToScale = data[round_2_2]
scaledColumns = scaleGroup(columnsToScale, 10, 5)
data.drop(round_2_2)
data = pd.concat([data, scaledColumns], axis=1)

round_7_2 = ['attr7_2', "sinc7_2", "intel7_2", "fun7_2", "amb7_2", "shar7_2"]
columnsToScale = data[round_7_2]
scaledColumns = scaleGroup(columnsToScale, 10, 5)
data.drop(round_7_2)
data = pd.concat([data, scaledColumns], axis=1)

round_7_3 = ["attr7_3", "sinc7_3", "intel7_3", "fun7_3", "amb7_3", "shar7_3"]
columnsToScale = data[round_7_3]
scaledColumns = scaleGroup(columnsToScale, 10, 5)
data.drop(round_7_3)
data = pd.concat([data, scaledColumns], axis=1)

round_4_3 = ["attr4_3", "sinc4_3", "intel4_3", "fun4_3", "amb4_3", "shar4_3"]
columnsToScale = data[round_4_3]
scaledColumns = scaleGroup(columnsToScale, 10, 5)
data.drop(round_4_3)
data = pd.concat([data, scaledColumns], axis=1)



#columnsToScale = wavesToScale[['attr1_1', "sinc1_1",]]

#%% Set NaN to median values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(data)
data = pd.DataFrame(imputer.transform(data), columns=data.columns, index=data.index)




#%%Correlation bewteen what you see as important vs how you rate the other person and if this correlates to a match
self_look_for_before = data[['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']]
date_score = data[['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']]
diff =  self_look_for_before.values - date_score



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



#%% Handle null/NaN's 
import re 


def removeByPrefix(prefix, data):
    column_names = data.columns;
    for column_name in column_names :
        if prefix in column_name:
            data = data.drop([ column_name ], axis=1)
    return data



# Get shared columns
column_names = data.columns;
first_round_columns = [];
second_round_columns = [];
attr_names = [];

for column_name in column_names :
    if column_name.endswith("_1",0) or column_name.endswith("_2",0):
        # Using re.findall() 
        # Splitting text and number in string  
        attr_splitted = re.findall('\d*\D+', column_name)
        attr_splitted = attr_splitted[0].split("_")
        attr_name = attr_splitted[0];
        attr_names.append(attr_name);
        
        if column_name.endswith("_1",0):
            first_round_columns.append(column_name)
        elif column_name.endswith("_2",0):
            second_round_columns.append(column_name)
        



# Selecting distinct columns
second_round_columns_distinct = list(dict.fromkeys(second_round_columns))
first_round_columns_distinct = list(dict.fromkeys(first_round_columns))


list_difference = []
for item in second_round_columns_distinct:
  if item not in first_round_columns_distinct:
    list_difference.append(item)



# removing data by a prefix
#data = removeByPrefix("satis", data)
#data = removeByPrefix("numdat", data)
#data = removeByPrefix("7", data)



# Go throgh all rows (each person)
second_round_columns.sort();
first_round_columns.sort();
columns_where_both_is_nan = [];

first_round_new = [];
second_round_new = [];


#for person_data in data: 
#    # get data for each of the sorted columns
#
#
#
#
#
#for i in range(len(second_round_columns)):
#    first_round_column_data = data[first_round_columns[i]]
#    second_round_column_data = data[second_round_columns[i]]
#    
#    
#    for index in range(len(first_round_column_data)):
#        first_round_column_person_data = first_round_column_data[index];
#        second_round_column_person_data = second_round_column_data[index];
#        
#        # Check for NaN
#        if np.isnan(first_round_column_person_data) and np.isnan(second_round_column_person_data):
#            # Remove when both are NaN
#            columns_where_both_is_nan.append(first_round_columns[i])
#        elif np.isnan(first_round_column_person_data):
#            first_round_column_data[index] = second_round_column_person_data;
#        elif np.isnan(second_round_column_person_data):
#            second_round_column_person_data[index] = first_round_column_person_data;
#        
#        
#        
        

# If 








    #matching = [s for s in column_names if "abc" in s]
    #diff_mising_columns = (first_round_columns != second_round_columns)


