# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:31:27 2020

@author: valde
"""
import os
import sys
sys.path.insert(0, os.getcwd())
print(sys.path)
from DataLoad import load_data
data = load_data("Speed_Dating_data.csv")
data.drop(['iid', 'id', 'idg', 'partner', 'pid', 'position', 'positin1', 'field_cd', 'undergra', 'tuition', 'from', 'zipcode', 'income', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing','reading', 'tv', 'theater', 'movies','concerts', 'music', 'shopping', 'yoga'], axis=1)


corr = data.corr()
from sklearn.metrics import confusion_matrix
#conf_matrix = 
match_corr = corr["match"].sort_values(ascending=False)





