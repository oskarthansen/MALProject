# -*- coding: utf-8 -*-

from DataLoad import load_data
from dataCleanUp import scaleGroup, replaceGroup
from Plots import FullReport

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
raw_data = load_data()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='constant', fill_value=0)
imputer.fit(raw_data[["shar", "shar_o"]])
shar = pd.DataFrame(imputer.transform(raw_data[["shar", "shar_o"]]), columns=["shar", "shar_o"], index=raw_data.index)
raw_data = replaceGroup(raw_data, shar)


null_sum = raw_data.isnull().sum()
too_many_nans = null_sum[null_sum < 750].index.values
too_many_nans = [str(index) for index in too_many_nans]
data = raw_data[too_many_nans]
data = data.dropna()
data = data.drop(["field", "from", "career"], axis=1)

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

round_1_1 = ['attr1_1', "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]
columnsToScale = data[round_1_1]
scaledColumns = scaleGroup(columnsToScale, 100)
data = replaceGroup(data, scaledColumns)

round_2_1 = ['attr2_1', "sinc2_1", "intel2_1", "fun2_1", "amb2_1", "shar2_1"]
columnsToScale = data[round_2_1]
scaledColumns = scaleGroup(columnsToScale, 100)
data = replaceGroup(data, scaledColumns)

#Correlation bewteen what you see as important vs how you rate the other person and if this correlates to a match
self_look_for_before = data[['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']]

date_score = data[['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']]

diff_before =  self_look_for_before.values - date_score
#diff_after_1 = self_look_for_after_date_1.values - date_score

def calcLength(row):
    return (row.values ** 2).mean() ** .5

lookfor_before_vs_datescore_diff = diff_before.apply(calcLength, axis=1)

#Invert scaling
lookfor_before_vs_datescore_diff = 100 - lookfor_before_vs_datescore_diff
lookfor_before_vs_datescore_diff.name = "lookfor_before_vs_datescore_diff"

data = pd.concat([data, pd.DataFrame(lookfor_before_vs_datescore_diff)], axis=1)

corr = data.corr()
corr_dec = corr['dec'].sort_values(ascending=False)
corr_match = corr["match"].sort_values(ascending=False)
    

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data_algorithm = data.drop(['match', 'iid', "id","idg", "condtn", "wave", "round", "position", "partner", "pid", "career_c", "sports", "tvsports", 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing','reading', 'tv', 'theater', 'movies','concerts', 'music', 'shopping', 'yoga'], axis=1)
def remove_by_contains(searchString, inputData):
    matching = [s for s in inputData.columns if searchString in s]
    inputData = inputData.drop(matching, axis=1)
    return inputData

# Remove data from "other" person. (The other persons opinion)
data_algorithm = remove_by_contains('_o', data_algorithm)

#Generate full dataset with labels and features
features_all = data_algorithm.drop('dec', axis=1)
labels_all = data_algorithm.dec
# Scaling the data
scaler = MinMaxScaler()
features_all = scaler.fit_transform(features_all)
#
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(features_all, labels_all, test_size=0.15, random_state=42)
X_train_all, X_valid_all, y_train_all, y_valid_all = train_test_split(X_train_all, y_train_all, test_size=0.15, random_state=42)


#%% Find optimale learning rate
from ParamTuning import build_keras_model
from keras.wrappers.scikit_learn import KerasClassifier
from time import time
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler
import math

input_shape = X_train_all.shape[1:]
models = []
model_keras = KerasClassifier(
            build_fn = build_keras_model, 
            hidden_layers=[300,100], 
            dropout_rate=[0.2,0.2],
            n_input=input_shape[0],
            n_class=1, 
            optimizer=Adam(learning_rate=10**-5),
            default_dropout=0.2)

epochs = 100
def my_learning_rate(epoch, lr):
    return lr * np.exp(np.log(10**6)/epochs)
lrs = LearningRateScheduler(my_learning_rate)

results = []

start = time()
early_history = model_keras.fit(X_train_all, y_train_all, epochs=epochs, callbacks=[lrs], validation_data=(X_valid_all, y_valid_all), verbose=0, batch_size=500)
t = time()-start
results.append(early_history)
history = pd.DataFrame(early_history.history)

optimal_lr = history.loc[history.idxmin()["loss"], :]["lr"]

for result in results:
    loss = pd.DataFrame(result.history)[["val_loss", "loss"]]
    lr = pd.DataFrame(result.history)[["lr"]]
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Loss")
    ax1.plot(np.arange(0,100), loss)
    ax1.set_ylim(0,10)
    ax2 = ax1.twinx()
    
    ax2.set_ylabel("Learning rate")
    ax2.set_yscale('log')
    ax2.plot(np.arange(0,100), lr, color="red")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
#%%
from keras import backend as K
K.clear_session()
import tensorflow as tf
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.compat.v1.Session(config=config) 
K.set_session(sess)

from ParamTuning import build_keras_model
from keras.wrappers.scikit_learn import KerasClassifier
from time import time
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, LearningRateScheduler
import math

input_shape = X_train_all.shape[1:]

from ParamTuning import build_keras_model
models = []
hidden_layers = [(200,200),(300,300)]
# hidden_layers = [(1000,500,300,100,50)]
hidden_layers = [(1000,500,300,100,50)]
dropout_rates = [0.25]
for layer in hidden_layers:
    for dropout_rate in dropout_rates:
        model_keras = build_keras_model(
        hidden_layers=layer, 
        n_input=input_shape[0],
        n_class=1, 
        default_dropout=dropout_rate,
        optimizer=Nadam(learning_rate=0.0015),
        metrics=["accuracy"])
        models.append(model_keras)

# number of epochs is set to a large number, we'l
# let early stopping terminate the training process
epochs = 200
def step_decay(epoch): 
   initial_lrate = 0.0015
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   print(lrate)
   return lrate


lrate = LearningRateScheduler(step_decay)

early_stop = EarlyStopping(monitor='val_loss', patience=40, min_delta=0.001, restore_best_weights=True)
results = []

for model in models: 
    start = time()
    early_history = model.fit(X_train_all, y_train_all, epochs=epochs, callbacks=[early_stop, lrate], validation_data=(X_valid_all, y_valid_all), verbose=1, batch_size=10)
    t = time()-start
    results.append(early_history)
    
# y_pred = models[0].predict(X_test_all_scaled)
# y_pred = y_pred[:,0]
# from lib.Plots import PlotPerformanceMatrix
# conf_matrix = PlotPerformanceMatrix(y_pred, y_test_all)

#%%
for index, result in enumerate(results):
    pd.DataFrame(result.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
    plt.show()
    print("Hidden layers")
    print(hidden_layers[index // len(dropout_rates)])
    print("Dropout rate")
    print(dropout_rates[index % len(dropout_rates)])
    
for index, hidden_layer in enumerate(hidden_layers):
    hidden_layer_score = pd.DataFrame()
    for i in np.arange(len(dropout_rates)):
        scores = pd.DataFrame(results[(index * len(dropout_rates)) + i].history).tail()
        hidden_layer_score = pd.concat([hidden_layer_score, scores], axis=0)
    hidden_layer_score_mean = hidden_layer_score.mean()
    print("Mean values for layer: " + str(hidden_layer))
    print(hidden_layer_score_mean)

for index, dropout_rate in enumerate(dropout_rates):
    dropout_rate_score = pd.DataFrame()
    for i in np.arange(len(hidden_layers)):
        scores = pd.DataFrame(results[(i * len(hidden_layers)) + index].history).tail()
        dropout_rate_score = pd.concat([dropout_rate_score, scores], axis=0)
    dropout_rate_score_mean = dropout_rate_score.mean()
    print("Mean values for dropout_rate: " + str(dropout_rate))
    print(dropout_rate_score_mean)

#%%
from sklearn.metrics import accuracy_score, mean_squared_error
#Make confusion matrix for model performance
cf_matrix_val_acc = pd.DataFrame(index=[str(dr) for dr in dropout_rates], columns=[str(l) for l in hidden_layers])
cf_matrix_val_loss = pd.DataFrame(index=[str(dr) for dr in dropout_rates], columns=[str(l) for l in hidden_layers])
cf_matrix_val_loss_vs_loss = pd.DataFrame(index=[str(dr) for dr in dropout_rates], columns=[str(l) for l in hidden_layers])
cf_matrix_test_acc = pd.DataFrame(index=[str(dr) for dr in dropout_rates], columns=[str(l) for l in hidden_layers])
cf_matrix_test_loss = pd.DataFrame(index=[str(dr) for dr in dropout_rates], columns=[str(l) for l in hidden_layers])

for index, hidden_layer in enumerate(hidden_layers):
    for i in np.arange(len(dropout_rates)):
        cf_matrix_val_acc[str(hidden_layer)][str(dropout_rates[i])] = pd.DataFrame(results[(index * len(dropout_rates)) + i].history)["val_accuracy"].max()
        cf_matrix_val_loss[str(hidden_layer)][str(dropout_rates[i])] = pd.DataFrame(results[(index * len(dropout_rates)) + i].history)["val_loss"].min()
        cf_matrix_val_loss_vs_loss[str(hidden_layer)][str(dropout_rates[i])] = pd.DataFrame(results[(index * len(dropout_rates)) + i].history)["val_loss"].tail(1).values[0] - pd.DataFrame(results[(index * len(dropout_rates)) + i].history)["loss"].tail(1).values[0]
        y_pred = models[(index * len(dropout_rates)) + i].predict(X_test_all);
        y_pred = y_pred[:,0]
        y_pred = y_pred.round()
        cf_matrix_test_acc[str(hidden_layer)][str(dropout_rates[i])] = accuracy_score(y_test_all, y_pred)
        cf_matrix_test_loss[str(hidden_layer)][str(dropout_rates[i])] = mean_squared_error(y_test_all, y_pred)
        
        

#%%
from Plots import PlotConfusionMatrix
PlotConfusionMatrix(cf_matrix_val_acc.astype(np.float64, copy=False), title="Validation accuracy")
PlotConfusionMatrix(cf_matrix_val_loss.astype(np.float64, copy=False), title="Validation loss")
PlotConfusionMatrix(cf_matrix_val_loss_vs_loss.astype(np.float64, copy=False), title="Validation loss - loss")
PlotConfusionMatrix(cf_matrix_test_acc.astype(np.float64, copy=False), title="Test accuracy")
PlotConfusionMatrix(cf_matrix_test_loss.astype(np.float64, copy=False), title="Test loss")


#%% Plot confusion matrix
from Plots import PlotPerformanceMatrix
for model in models:
    y_pred = model.predict(X_test_all)
    y_pred = y_pred[:,0]
    y_pred = y_pred.round()
    PlotPerformanceMatrix(y_pred, y_test_all)

#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from Plots import PlotPerformanceMatrix

param_distribs = {
        'n_estimators': randint(low=1, high=300),
        'max_features': randint(low=1, high=50),
    }

forest_reg = RandomForestRegressor(random_state=42)

rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(X_train_all, y_train_all)

rnd_search.best_estimator_
forest_reg = rnd_search.best_estimator_

# scores = cross_val_score(forest_reg, X_train_all, y_train_all, scoring="neg_mean_squared_error", cv=10)
# display_scores(-scores, forest_reg)


#%%Plot
from Plots import PlotClasswiseSeries2
y_pred2 = forest_reg.predict(X_test_all)
model = models[0]
y_pred1 = model.predict(X_test_all)
y_pred1 = y_pred1[:,0]
num_outliers = sum(((y_pred1 > 0.6) & (y_test_all == 0)) | ((y_pred1 < 0.4) & (y_test_all == 1)))
PlotClasswiseSeries2(y_pred1, y_pred2, y_test_all)

#%% Cross val score
from sklearn.model_selection import cross_val_score
scores = []
for index, model in enumerate(models):
    score = cross_val_score(model, X_train_all, y_train_all, scoring="neg_mean_squared_error")
    scores[index] = score

#%% Display scores


def display_scores(scores):
    print("Scores:", scores)
    print("Scores mean:", scores.mean())
    print("Standard deviation:", scores.std())

from sklearn.model_selection import cross_val_score



#%% Regression model

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X_predict_test = X_test_all[:20]

lin_reg = LinearRegression()
lin_reg.fit(X_train_all_scaled, y_train_all)

print("LinearRegression")

scores = cross_val_score(lin_reg, X_test_all_scaled, y_test_all, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-scores)

display_scores(-scores)
#%% DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train_all_scaled, y_train_all)

print("DecisionTreeRegressor")

scores = cross_val_score(tree_reg, X_test_all_scaled, y_test_all, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(-scores)

#%% RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()

forest_reg.fit(X_train_all_scaled, y_train_all)

print("RandomForestRegressor")

scores = cross_val_score(forest_reg, X_test_all_scaled, y_test_all, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)
display_scores(-scores)

#%% SVM poly
from sklearn.svm import SVR

svm_poly_reg = SVR(kernel='poly', C=100, gamma='auto', degree=389, epsilon=.1, coef0=1)
svm_poly_reg.fit(X_test_all_scaled, y_test_all)

print("Support Vector Regression - Polynomial model")
scores = cross_val_score(svm_poly_reg, X_test_all_scaled, y_test_all, scoring="neg_mean_squared_error", cv=10)
svm_poly_rmse_scores = np.sqrt(-scores)
display_scores(-scores)