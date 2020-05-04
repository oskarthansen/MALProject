# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:11:00 2020

@author: valde
"""

def PlotBarSeries(dataframe, yaxis_title, title, ylim):
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
    ax.set_ylim(ylim)
    fig.tight_layout()
    
    plt.show()
    
def PlotHeatmap(dataframe, title, vmin, vmax):
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax= plt.subplots()
    sns.heatmap(dataframe,cmap="Greys", ax=ax, vmin=vmin, vmax=vmax, square=True)
    ax.set_title(title, fontsize=15)
    fig.tight_layout()

def ClassificationReport(model, X_test, y_test, target_names=None):
    from sklearn.metrics import classification_report
    assert X_test.shape[0]==y_test.shape[0]
    print("\nDetailed classification report:")
    print("\tThe model is trained on the full development set.")
    print("\tThe scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, model.predict(X_test)                 
    print(classification_report(y_true, y_pred, target_names))
    print()
    
def SearchReport(model): 
    def GetBestModelCTOR(model, best_params):
        def GetParams(best_params):
            ret_str=""          
            for key in sorted(best_params):
                value = best_params[key]
                temp_str = "'" if str(type(value))=="<class 'str'>" else ""
                if len(ret_str)>0:
                    ret_str += ','
                ret_str += f'{key}={temp_str}{value}{temp_str}'  
            return ret_str          
        try:
            param_str = GetParams(best_params)
            return type(model).__name__ + '(' + param_str + ')' 
        except:
            return "N/A(1)"
        
    print("\nBest model set found on train set:")
    print()
    print(f"\tbest parameters={model.best_params_}")
    print(f"\tbest '{model.scoring}' score={model.best_score_}")
    print(f"\tbest index={model.best_index_}")
    print()
    print(f"Best estimator CTOR:")
    print(f"\t{model.best_estimator_}")
    print()
    try:
        print(f"Grid scores ('{model.scoring}') on development set:")
        means = model.cv_results_['mean_test_score']
        stds  = model.cv_results_['std_test_score']
        i=0
        for mean, std, params in zip(means, stds, model.cv_results_['params']):
            print("\t[%2d]: %0.3f (+/-%0.03f) for %r" % (i, mean, std * 2, params))
            i += 1
    except:
        print("WARNING: the random search do not provide means/stds")
                  
    assert "f1_micro"==str(model.scoring), f"come on, we need to fix the scoring to be able to compare model-fits! Your scoreing={str(model.scoring)}...remember to add scoring='f1_micro' to the search"   
    return f"best: score={model.best_score_:0.5f}, model={GetBestModelCTOR(model.estimator,model.best_params_)}", model.best_estimator_ 

def FullReport(model, X_test, y_test, t):
    print(f"SEARCH TIME: {t:0.2f} sec")
    beststr, bestmodel = SearchReport(model)
    ClassificationReport(model, X_test, y_test)    
    print(f"CTOR for best model: {bestmodel}\n")
    print(f"{beststr}\n")
    return beststr, bestmodel
def PlotPerformanceMatrix(y_pred, y_true):
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt
    tp = sum(y_true[(y_pred == y_true) & (y_true == 1)])
    fp = sum(y_pred[(y_pred == 1) & (y_true == 0)])
    tn = sum(sum([(y_pred == 0) & (y_true == 0)]))
    fn = sum(sum([(y_pred == 0) & (y_true == 1)]))
    matrix = [[tp, fp], [fn, tn]]
    df = pd.DataFrame(matrix, index=[1,0], columns=[1,0])
    plt.figure(figsize = (10,7))
    ax = plt.axes()
    sn.heatmap(df, annot=True, cmap="gray", fmt='.0f')
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    return matrix
def getPerformanceMetrics(y_pred, y_true): 
    tp = sum(y_true[(abs(y_pred) == y_true) & (y_true == 1)])
    fp = sum(y_pred[(y_pred == 1) & (y_true == 0)])
    tn = sum(sum([(y_pred == 0) & (y_true == 0)]))
    fn = sum(sum([(y_pred == 0) & (y_true == 1)]))
    return tp, fp, tn, fn
    
def PlotConfusionMatrix(df, title):
    import seaborn as sn 
    import matplotlib.pyplot as plt
    ax = plt.axes()
    sn.heatmap(df, annot=True, cmap="gray", ax=ax, fmt='.4f')
    ax.set_title(title)
    plt.show()
    
def PlotClasswiseSeries(y_pred, y_true):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame({
        'y_pred': y_pred,
        'y_true': y_true
        })
    df.sort_values(by=['y_pred'])
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    z = df[['y_true']].to_numpy().astype(int)
    colors = ["#0000ff", "#ff0000"]
    c = [colors[i] for i in z[:,0]]
    x = df[['y_pred']].to_numpy()
    y = df[['y_true']].to_numpy()
    ax.scatter(x, y, c=c, marker='x', alpha=0.2)
    ax.set_xlabel('Predicted value')
    ax.set_ylabel('True value')
    ax.set_title('Predicted values for dec')
    plt.show()

    
