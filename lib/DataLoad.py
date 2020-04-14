# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:19:31 2020

@author: valde
"""
def load_data():
    import pandas as pd
    csv_path = "Speed_Dating_Data.csv"
    return pd.read_csv(csv_path, encoding = "ISO-8859-1")


