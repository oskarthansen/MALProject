# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:31:27 2020

@author: valde
"""
from lib.DataLoad import load_data
from lib.dataCleanUp import scaleGroup, replaceGroup
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
raw_data = load_data()
#data = raw_data.drop(['id', 'idg', 'partner', 'position', 'positin1', 'career', "career_c", 'field', 'undergra', 'tuition', 'from', 'zipcode', 'income', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing','reading', 'tv', 'theater', 'movies','concerts', 'music', 'shopping', 'yoga', 'income', 'mn_sat' ], axis=1)
#data = data[data.columns.drop(list(data.filter(regex='_3')))]

#%%
# For fun, see how many found a match
pd.crosstab(index=raw_data['match'],columns="count")

#%% Removing Nans
# Summerizing nans in every feature/coloum 
raw_data.isnull().sum()
# Chosing which features to drop. Based on number of NaNs, and interest for us. 
data1 = raw_data.iloc[:, 11:28]
data2 = raw_data.iloc[:, 30:35]
data3 = raw_data.iloc[:, 39:43]
data4 = raw_data.iloc[:, 45:67]
data5 = raw_data.iloc[:, 69:74]
data6 = raw_data.iloc[:, 87:91]
data7 = raw_data.iloc[:, 97:102]
data8 = raw_data.iloc[:, 104:107]

data = pd.concat([raw_data.iloc[:, 0],raw_data.iloc[:, 2],data1,data2,data3,data4,data5,data6,data7,data8], axis=1)
# Summerizing null values again
data.isnull().sum()
# removing rows with a nan values. Okay, to do because the NaNs in the features are more likely 100 than 1000
data2 = data.dropna()
# See if it works
data2.isnull().sum()

# Looking on data types
data2.dtypes
# Removing the object features. Maybe, we will onehot-encode them later
data3 = data2.drop(['field', 'from', 'career'], axis=1)
# Make a heatmap
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Correlation Heatmap")
corr = data3.corr()
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)

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
#Check to see if similar goal
same_goal_column = pd.DataFrame(0, index=np.arange(len(data.index)), columns=["same_goal"])

for i, row in data.iterrows():
    goal = row.filter(regex="goal")
    pid = row["pid"]
    pid_rows = data["iid"] == pid
    pid_rows = data[pid_rows]
    if len(pid_rows) > 0:
        pid_rows = pid_rows.iloc[1,:] #get first row only
        goal_o = pid_rows.filter(regex="goal")
        same_goal = sum(((goal == goal_o) & (goal == 1)))
        same_goal_column.at[i, "same_goal"] = same_goal
data = pd.concat([data, same_goal_column], axis=1)


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

round_4_1 = ['attr4_1', "sinc4_1", "intel4_1", "fun4_1", "amb4_1", "shar4_1"]
columnsToScale = data[round_4_1]
scaledColumns = scaleGroup(columnsToScale, 100)
data = replaceGroup(data, scaledColumns)

round_2_1 = ['attr2_1', "sinc2_1", "intel2_1", "fun2_1", "amb2_1", "shar2_1"]
columnsToScale = data[round_2_1]
scaledColumns = scaleGroup(columnsToScale, 100)
data = replaceGroup(data, scaledColumns)

round_1_2 = ['attr1_2', "sinc1_2", "intel1_2", "fun1_2", "amb1_2", "shar1_2"]
columnsToScale = data[round_1_2]
scaledColumns = scaleGroup(columnsToScale, 100)
data = replaceGroup(data, scaledColumns)

round_4_2 = ["attr4_2", "sinc4_2", "intel4_2", "fun4_2", "amb4_2", "shar4_2"]
columnsToScale = data[round_4_2]
scaledColumns = scaleGroup(columnsToScale, 100)
data = replaceGroup(data, scaledColumns)

round_2_2 = ["attr2_2", "sinc2_2", "intel2_2", "fun2_2", "amb2_2", "shar2_2"]
columnsToScale = data[round_2_2]
scaledColumns = scaleGroup(columnsToScale, 100)
data = replaceGroup(data, scaledColumns)

round_7_2 = ['attr7_2', "sinc7_2", "intel7_2", "fun7_2", "amb7_2", "shar7_2"]
columnsToScale = data[round_7_2]
scaledColumns = scaleGroup(columnsToScale, 100)
data = replaceGroup(data, scaledColumns)


#%%Scale to 100 point scale 
#round_3_1 = data[list(data.filter(regex="3_1"))]
#round_3_1 = scaleGroup(round_3_1, 100)
#data= replaceGroup(data, round_3_1)

round_3_s = data[list(data.filter(regex="3_s"))]
round_3_s = scaleGroup(round_3_s, 100)
data = replaceGroup(data, round_3_s)

score = data[["attr", "sinc", "intel", "fun", "amb", "shar"]]
#score = scaleGroup(score, 100)
data = replaceGroup(data, score)

score_o = data[["attr_o", "sinc_o", "intel_o", "fun_o", "amb_o", "shar_o"]]
score_o = scaleGroup(score_o, 100)
data = replaceGroup(data, score_o)

#%%
#round_5_1 = data[list(data.filter(regex="5_1"))]
#round_5_1 = scaleGroup(round_5_1, 100)
#data = replaceGroup(data, round_5_1)

#round_3_s = data[list(data.filter(regex="3_s"))]
#round_3_s = scaleGroup(round_3_s, 100)
#data = replaceGroup(data, round_3_s)

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
corr_match = corr["match"].sort_values(ascending=False)


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
PlotBarSeries(dataframe, "Mean value","Attribute value mean by gender (round 1_1)")


#%%
self_look_for_during_average_male = male_avg[['attr1_s', 'sinc1_s', 'intel1_s', 'fun1_s', 'amb1_s', 'shar1_s']]
self_look_for_during_average_female = female_avg[['attr1_s', 'sinc1_s', 'intel1_s', 'fun1_s', 'amb1_s', 'shar1_s']]
labels = ['attr1_s', 'sinc1_s', 'intel1_s', 'fun1_s', 'amb1_s', 'shar1_s']
dataframe = pd.concat([self_look_for_during_average_male, self_look_for_during_average_female], axis=1).T
dataframe.index =  ["male", "female"]
PlotBarSeries(dataframe, "Mean value","Attribute value mean by gender (round 1_s)")

#%% Mean values by attribute for dec = 1
all_dec_1_male = data[(data["dec"] == 1) & (data["gender"] == 1)]
all_dec_1_female = data[(data["dec"] == 1) & (data["gender"] == 0)]

attrs = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']

male_attrs_dec_avg = all_dec_1_male[attrs]
female_attrs_dec_avg = all_dec_1_female[attrs]
dataframe = pd.concat([male_attrs_dec_avg.mean(), female_attrs_dec_avg.mean()], axis=1).T
dataframe.index = ["male", "female"]
PlotBarSeries(dataframe, "Score mean value", "Date score mean value for dec=1")

all_dec_0_male = data[(data["dec"] == 0) & (data["gender"] == 1)]
all_dec_0_female = data[(data["dec"] == 0) & (data["gender"] == 0)]

male_attrs_dec_avg = all_dec_0_male[attrs]
female_attrs_dec_avg = all_dec_0_female[attrs]
dataframe = pd.concat([male_attrs_dec_avg.mean(), female_attrs_dec_avg.mean()], axis=1).T
dataframe.index = ["male", "female"]
PlotBarSeries(dataframe, "Score mean value", "Date score mean value for dec=0")

#%% Mean values by attribute for dec = 1 compared to what men/women say they want



#%%Yes vs no scores
from Plots import PlotHeatmap
dec_yes = data[data["dec"] == 1]
dec_no = data[data["dec"] == 0]
dec_yes_attr = dec_yes[attrs + ["like"]]
dec_no_attr = dec_no[attrs + ["like"]]
dec_yes_mean = pd.DataFrame(dec_yes_attr.mean()).T
dec_yes_mean.index = ["Yes"]
dec_no_mean = pd.DataFrame(dec_no_attr.mean()).T
dec_no_mean.index = ["No"]
df = pd.concat([dec_yes_mean, dec_no_mean], axis=0)
PlotBarSeries(df, "Mean rating", "Mean rating of partner for yes and no")

corr_yes = dec_yes_attr.corr()
corr_no = dec_no_attr.corr()
PlotHeatmap(corr_yes, "Yes", 0, 1)
PlotHeatmap(corr_no, "No", 0, 1)

#%%Check for difference in how you think measure up vs how you think others perceive you
self_score_1 = data[list(data.filter(regex="3_1"))]
think_other_score = data[list(data.filter(regex="5_1"))]
diff = self_score_1.values - think_other_score
diff_mean = pd.DataFrame(diff.mean()).T
diff_mean.index = ["Diff"]
PlotBarSeries(diff_mean, "Mean diff", "Self score - think others percieve you")


#%% Check to see if you can predict your own score accurately. Which score predicts better? Prior or during the speed dating event?
attrs = ['attr', 'sinc', 'intel', 'fun', 'amb']
diff_scores = pd.DataFrame()
for i in np.arange(552):
    #Get rows for current participant
    rows = data[data["iid"] == i]
    self_score_1 = pd.DataFrame(rows[list(rows.filter(regex="3_1"))].mean()).T
    think_other_score = pd.DataFrame(rows[list(rows.filter(regex="5_1"))].mean()).T
    my_score = pd.DataFrame(rows[[attr + "_o" for attr in attrs]].mean()).T
    self_score_diff = self_score_1 - my_score.values
    think_other_score_diff = think_other_score - my_score.values
    result = pd.concat([self_score_diff, think_other_score_diff], axis=1)
    diff_scores = pd.concat([diff_scores, result], axis=0)
    
diff_scores_mean = diff_scores.mean()
df = pd.DataFrame([diff_scores_mean])

first_row = df.drop(list(df.filter(regex="3_1")), axis=1)
first_row.index = ["3_1"]
first_row.columns = attrs

second_row = df.drop(list(df.filter(regex="5_1")), axis=1)
second_row.index = ["5_1"]
second_row.columns = attrs

df = pd.concat([first_row, second_row], axis=0)
PlotBarSeries(df, "Mean difference", "Attribute score prediction for 3_1 and 5_1")

#%%How does the way that you view yourself change before and during the speed date
attr_3_1 = data[list(data.filter(regex="3_1"))]
attr_3_s = data[list(data.filter(regex="3_s"))]
attr_3_1 = pd.DataFrame(attr_3_1.mean()).T
attr_3_s = pd.DataFrame(attr_3_s.mean()).T
attr_3_1.columns = attrs
attr_3_s.columns = attrs
attr_3_1.index = ["Before"]
attr_3_s.index = ["During"]
df = pd.concat([attr_3_1, attr_3_s], axis=0)
PlotBarSeries(df, "Mean score", "How du you rate yourself before vs during the speedate?")


#%%Does the number of dec=1 impact the way you feel about yourself?


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


#%%
    
#%% For later use



round_3_2 = data[list(data.filter(regex="3_2"))]
data = replaceGroup(data, round_3_2)

round_5_2 = data[list(data.filter(regex="5_2"))]
data = replaceGroup(data, round_5_2)






