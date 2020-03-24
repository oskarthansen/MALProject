# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:12:06 2020

@author: jonas
"""
def scaleGroup(group, scalar):
    #group["sum"] = group.sum(axis=1)
    summ = group.sum(axis=1)
    for i in group.columns:
        group[i] = (group[i]/summ)*100
    return group