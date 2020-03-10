# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:19:31 2020

@author: valde
"""
def load_data(filename):
    import os 
    import pandas as pd
    csv_path = os.path.join(os.getcwd(), filename)
    return pd.read_csv(csv_path, encoding = "ISO-8859-1")


