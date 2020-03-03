# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:31:27 2020

@author: valde
"""
import os
import sys
sys.path.insert(0, os.getcwd())
print(sys.path)
from lib.DataLoad import load_data
data = load_data("")
