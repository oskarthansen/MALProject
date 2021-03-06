# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:31:27 2020

@author: valde
"""
from DataLoad import load_data
from dataCleanUp import scaleGroup, replaceGroup
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
raw_data = load_data()
data = raw_data.drop(['id', 'idg', 'partner', 'position', 'positin1', 'career', "career_c", 'field', 'undergra', 'tuition', 'from', 'zipcode', 'income', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing','reading', 'tv', 'theater', 'movies','concerts', 'music', 'shopping', 'yoga', 'income', 'mn_sat' ], axis=1)
#data = data[data.columns.drop(list(data.filter(regex='_3')))]

#%%
# For fun, see how many found a match
pd.crosstab(index=raw_data['match'],columns="count")

shar = raw_data[["shar","iid"]]

#%% Removing Nans
## Summerizing nans in every feature/coloum 
#
## Chosing which features to drop. Based on number of NaNs, and interest for us.
#
#data1 = raw_data.iloc[:, 11:28]
#data2 = raw_data.iloc[:, 30:35]
#data3 = raw_data.iloc[:, 39:43]
#data4 = raw_data.iloc[:, 45:67]
#data5 = raw_data.iloc[:, 69:74]
#data6 = raw_data.iloc[:, 87:91]
#data7 = raw_data.iloc[:, 97:102]
#data8 = raw_data.iloc[:, 104:107]
#
#data = pd.concat([raw_data.iloc[:, 0],raw_data.iloc[:, 2],data1,data2,data3,data4,data5,data6,data7,data8], axis=1)
## Summerizing null values again
#data.isnull().sum()
## removing rows with a nan values. Okay, to do because the NaNs in the features are more likely 100 than 1000
#data2 = data.dropna()
#
## See if it works
#data2.isnull().sum()
#
## Looking on data types
#data2.dtypes
## Removing the object features. Maybe, we will onehot-encode them later
#data3 = data2.drop(['field', 'from', 'career'], axis=1)
## Make a heatmap
#plt.subplots(figsize=(20,15))
#ax = plt.axes()
#ax.set_title("Correlation Heatmap")
#corr = data3.corr()
#sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)
#%%
#Alternative way to filter away columns with to many NaN values.
#Preserve shar value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
imputer.fit(raw_data[["shar", "shar_o"]])
shar = pd.DataFrame(imputer.transform(raw_data[["shar", "shar_o"]]), columns=["shar", "shar_o"], index=raw_data.index)
raw_data = replaceGroup(raw_data, shar)


null_sum = raw_data.isnull().sum()
too_many_nans = null_sum[null_sum < 750].index.values
too_many_nans = [str(index) for index in too_many_nans]
data = raw_data[too_many_nans]
data = data.dropna()
data = data.drop(["field", "from", "career"], axis=1)

#%%One hot encoding
data = data[data.columns.drop(list(data.filter(regex="_3")))]

data.drop(["gender", "race_o", "race", "field_cd"], axis=1)

field_1hot = pd.get_dummies(data['field_cd'], prefix= 'field') #Encode fields
data = data.drop('field_cd', axis=1)
data = pd.concat([data, field_1hot], axis=1)

race_1hot = pd.get_dummies(data['race'], prefix='race')
data = data.drop('race', axis=1)
data = pd.concat([data, race_1hot], axis=1)

goal_1hot = pd.get_dummies(data['goal'], prefix='goal')
data = data.drop('goal', axis=1)
data = pd.concat([data, goal_1hot], axis=1)

date = data['date']
date = np.abs(8 - date)
data = data.drop('date', axis=1)
data = pd.concat([data, date], axis=1)

go_out = data['go_out']
go_out = np.abs(8-go_out)
data = data.drop('go_out', axis=1)
data = pd.concat([data, go_out], axis=1)

#%%
round_1_1 = ['attr1_1', "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]
columnsToScale = data[round_1_1]
scaledColumns = scaleGroup(columnsToScale, 100)
data = replaceGroup(data, scaledColumns)

round_2_1 = ['attr2_1', "sinc2_1", "intel2_1", "fun2_1", "amb2_1", "shar2_1"]
columnsToScale = data[round_2_1]
scaledColumns = scaleGroup(columnsToScale, 100)
data = replaceGroup(data, scaledColumns)

#%%Correlation bewteen what you see as important vs how you rate the other person and if this correlates to a match
self_look_for_before = data[['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']]


date_score = data[['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']]

diff_before =  self_look_for_before.values - date_score
#diff_after_1 = self_look_for_after_date_1.values - date_score

def calcLength(row):
    return (row.values ** 2).mean() ** .5

lookfor_before_vs_datescore_diff = diff_before.apply(calcLength, axis=1)
#lookfor_after1_vs_datescore_diff = diff_after_1.apply(calcLength, axis=1)

#Invert scaling
lookfor_before_vs_datescore_diff = 100 - lookfor_before_vs_datescore_diff
lookfor_before_vs_datescore_diff.name = "lookfor_before_vs_datescore_diff"

data = pd.concat([data, pd.DataFrame(lookfor_before_vs_datescore_diff)], axis=1)


#%%
self_look_for_before = data[['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']]
date_score = data[['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']]

scaled_date_score = date_score * self_look_for_before.values
scaled_date_score.columns = [ s + '_s' for s in ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']]
data = pd.concat([data, scaled_date_score], axis=1)

data_algorithm = data.drop(['match', 'iid', "id","idg", "condtn", "wave", "round", "position", "partner", "pid", "career_c", "sports", "tvsports", 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing','reading', 'tv', 'theater', 'movies','concerts', 'music', 'shopping', 'yoga'], axis=1)
data1 = data_algorithm.drop(list(data_algorithm.filter(regex="field")), axis=1)
data1 = data1.drop(list(data1.filter(regex="goal")), axis=1)
data1 = data1.drop(list(data1.filter(regex="_o")), axis=1)
data1 = data1.drop(list(data1.filter(regex="race")), axis=1)

corr = data1.corr()
corr_dec = corr['dec'].sort_values(ascending=False)

plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Correlation Heatmap")
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)
#%% Check to see what the different genders value most on paper
from Plots import PlotBarSeries
male_rows = data[data['gender'] == 1]
female_rows = data[data["gender"] == 0]
male_avg = male_rows.mean()
female_avg = female_rows.mean()

self_look_for_before_average_male = male_avg[['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']]
self_look_for_before_average_female = female_avg[['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']]
dataframe = pd.concat([self_look_for_before_average_male, self_look_for_before_average_female],axis=1).T
dataframe.index = ["male", "female"]
PlotBarSeries(dataframe, "Mean value","Attribute value mean by gender (round 1_1)", [0,30])


#%% Mean values by attribute for dec = 1
all_dec_1_male = data[(data["dec"] == 1) & (data["gender"] == 1)]
all_dec_1_female = data[(data["dec"] == 1) & (data["gender"] == 0)]

attrs = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like']

male_attrs_dec_avg = all_dec_1_male[attrs]
female_attrs_dec_avg = all_dec_1_female[attrs]
dataframe = pd.concat([male_attrs_dec_avg.mean(), female_attrs_dec_avg.mean()], axis=1).T
dataframe.index = ["male", "female"]
PlotBarSeries(dataframe, "Score mean value", "Date score mean value for dec=1", [0,10])

all_dec_0_male = data[(data["dec"] == 0) & (data["gender"] == 1)]
all_dec_0_female = data[(data["dec"] == 0) & (data["gender"] == 0)]

male_attrs_dec_avg = all_dec_0_male[attrs]
female_attrs_dec_avg = all_dec_0_female[attrs]
dataframe = pd.concat([male_attrs_dec_avg.mean(), female_attrs_dec_avg.mean()], axis=1).T
dataframe.index = ["male", "female"]
PlotBarSeries(dataframe, "Score mean value", "Date score mean value for dec=0", [0,10])

#%%Yes vs no scores
from Plots import PlotHeatmap
dec_yes = data[data["dec"] == 1]
dec_no = data[data["dec"] == 0]
dec_yes_attr = dec_yes[attrs]
dec_no_attr = dec_no[attrs]
dec_yes_mean = pd.DataFrame(dec_yes_attr.mean()).T
dec_yes_mean.index = ["Yes"]
dec_no_mean = pd.DataFrame(dec_no_attr.mean()).T
dec_no_mean.index = ["No"]
df = pd.concat([dec_yes_mean, dec_no_mean], axis=0)
PlotBarSeries(df, "Mean rating", "Mean rating of partner for yes and no", [0,10])

corr_yes = dec_yes_attr.corr()
corr_no = dec_no_attr.corr()
PlotHeatmap(corr_yes, "Yes", 0, 1)
PlotHeatmap(corr_no, "No", 0, 1)


#%% Check to see if you can predict your own score accurately. Which score predicts better? Prior or during the speed dating event?
attrs = ['attr', 'sinc', 'intel', 'fun', 'amb']
diff_scores = pd.DataFrame()
for i in np.arange(552):
    #Get rows for current participant
    rows = data[data["iid"] == i]
    self_score_1 = pd.DataFrame(rows[list(rows.filter(regex="3_1"))].mean()).T
    my_score = pd.DataFrame(rows[[attr + "_o" for attr in attrs]].mean()).T
    self_score_diff = self_score_1 - my_score.values
    result = pd.concat([self_score_diff], axis=1)
    diff_scores = pd.concat([diff_scores, result], axis=0)
    
diff_scores_mean = diff_scores.mean()
df = pd.DataFrame([diff_scores_mean])

df.index = ["3_1"]
df.columns = attrs

PlotBarSeries(df, "Mean difference", "Attribute score prediction for 3_1", [0,2.5])


#%% Does int_corr correlate with shar? or are we not able to evaluate shared interests in 4 minutes?
int_corr = data[["int_corr"]]
shar = data[["shar", "shar_o"]]
df = pd.concat([int_corr, shar], axis=1)
corr = df.corr()
PlotHeatmap(corr, "Shared interests", 0, 1)

#%%How good are we at predicting the other persons answer
dec_yes_other = data[data["dec_o"] == 1]
dec_no_other = data[data["dec_o"] == 0]
dec_yes = data[data["dec"] == 1]
dec_no = data[data["dec"] == 0]

dec_yes_other_prob_mean = dec_yes_other["prob"].mean()
dec_no_other_prob_mean = dec_no_other["prob"].mean()

dec_yes_prob_mean = dec_yes["prob"].mean()
dec_no_prob_mean = dec_no["prob"].mean()

df = pd.DataFrame([[dec_yes_prob_mean, dec_yes_other_prob_mean], [dec_no_prob_mean, dec_no_other_prob_mean]], columns=["Own decision", "Other decision"], index=["Yes", "No"])
PlotBarSeries(df,"prob mean-value" ,"Average probability for other say yes for dec=yes and dec=no")
    