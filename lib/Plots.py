# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:11:00 2020

@author: valde
"""

def PlotBarSeries(dataframe, yaxis_title, title):
    import matplotlib.pyplot as plt
    import numpy as np
    
    x = np.arange(len(dataframe.columns))
    bar_width = 0.35
    labels = dataframe.index
    fig, ax = plt.subplots()
    for index, label  in enumerate(labels):
        rects = ax.bar(x - ((bar_width/2) * ((-1)**index) * (index+1)/2) , dataframe.loc[label, :], bar_width, label=label)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(yaxis_title)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(dataframe.columns)
    ax.legend()
    
    fig.tight_layout()
    
    plt.show()
    
def PlotHeatmap(dataframe, title, vmin, vmax):
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax= plt.subplots()
    sns.heatmap(dataframe,cmap="Greys", ax=ax, vmin=vmin, vmax=vmax, square=True)
    ax.set_title(title, fontsize=15)
    fig.tight_layout()