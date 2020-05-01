# -*- coding: utf-8 -*-

from lib.DataLoad import load_data
from lib.dataCleanUp import scaleGroup, replaceGroup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
raw_data = load_data()
#data = raw_data.drop(['id', 'idg', 'partner', 'position', 'positin1', 'career', "career_c", 'field', 'undergra', 'tuition', 'from', 'zipcode', 'income', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing','reading', 'tv', 'theater', 'movies','concerts', 'music', 'shopping', 'yoga', 'income', 'mn_sat' ], axis=1)
#data = data[data.columns.drop(list(data.filter(regex='_3')))]


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
#lookfor_after1_vs_datescore_diff = diff_after_1.apply(calcLength, axis=1)

#Invert scaling
lookfor_before_vs_datescore_diff = 100 - lookfor_before_vs_datescore_diff
lookfor_before_vs_datescore_diff.name = "lookfor_before_vs_datescore_diff"

data = pd.concat([data, pd.DataFrame(lookfor_before_vs_datescore_diff)], axis=1)

corr = data.corr()
corr_dec = corr['dec'].sort_values(ascending=False)
corr_match = corr["match"].sort_values(ascending=False)


##%% Check to see what the different genders value most on paper
#from Plots import PlotBarSeries
#male_rows = data[data['gender'] == 1]
#female_rows = data[data["gender"] == 0]
#male_avg = male_rows.mean()
#female_avg = female_rows.mean()
#
#self_look_for_before_average_male = male_avg[['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']]
#self_look_for_before_average_female = female_avg[['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']]
#dataframe = pd.concat([self_look_for_before_average_male, self_look_for_before_average_female],axis=1).T
#dataframe.index = ["male", "female"]
#PlotBarSeries(dataframe, "Mean value","Attribute value mean by gender (round 1_1)")
#
#
##%% Mean values by attribute for dec = 1
#all_dec_1_male = data[(data["dec"] == 1) & (data["gender"] == 1)]
#all_dec_1_female = data[(data["dec"] == 1) & (data["gender"] == 0)]
#
#attrs = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']
#
#male_attrs_dec_avg = all_dec_1_male[attrs]
#female_attrs_dec_avg = all_dec_1_female[attrs]
#dataframe = pd.concat([male_attrs_dec_avg.mean(), female_attrs_dec_avg.mean()], axis=1).T
#dataframe.index = ["male", "female"]
#PlotBarSeries(dataframe, "Score mean value", "Date score mean value for dec=1")
#
#all_dec_0_male = data[(data["dec"] == 0) & (data["gender"] == 1)]
#all_dec_0_female = data[(data["dec"] == 0) & (data["gender"] == 0)]
#
#male_attrs_dec_avg = all_dec_0_male[attrs]
#female_attrs_dec_avg = all_dec_0_female[attrs]
#dataframe = pd.concat([male_attrs_dec_avg.mean(), female_attrs_dec_avg.mean()], axis=1).T
#dataframe.index = ["male", "female"]
#PlotBarSeries(dataframe, "Score mean value", "Date score mean value for dec=0")
#
##%% Mean values by attribute for dec = 1 compared to what men/women say they want
#
#
#
##%%Yes vs no scores
#from Plots import PlotHeatmap
#dec_yes = data[data["dec"] == 1]
#dec_no = data[data["dec"] == 0]
#dec_yes_attr = dec_yes[attrs + ["like"]]
#dec_no_attr = dec_no[attrs + ["like"]]
#dec_yes_mean = pd.DataFrame(dec_yes_attr.mean()).T
#dec_yes_mean.index = ["Yes"]
#dec_no_mean = pd.DataFrame(dec_no_attr.mean()).T
#dec_no_mean.index = ["No"]
#df = pd.concat([dec_yes_mean, dec_no_mean], axis=0)
#PlotBarSeries(df, "Mean rating", "Mean rating of partner for yes and no")
#
#corr_yes = dec_yes_attr.corr()
#corr_no = dec_no_attr.corr()
#PlotHeatmap(corr_yes, "Yes", 0, 1)
#PlotHeatmap(corr_no, "No", 0, 1)
#
#
##%% Check to see if you can predict your own score accurately. Which score predicts better? Prior or during the speed dating event?
#attrs = ['attr', 'sinc', 'intel', 'fun', 'amb']
#diff_scores = pd.DataFrame()
#for i in np.arange(552):
#    #Get rows for current participant
#    rows = data[data["iid"] == i]
#    self_score_1 = pd.DataFrame(rows[list(rows.filter(regex="3_1"))].mean()).T
#    my_score = pd.DataFrame(rows[[attr + "_o" for attr in attrs]].mean()).T
#    self_score_diff = self_score_1 - my_score.values
#    result = pd.concat([self_score_diff], axis=1)
#    diff_scores = pd.concat([diff_scores, result], axis=0)
#    
#diff_scores_mean = diff_scores.mean()
#df = pd.DataFrame([diff_scores_mean])
#
#df.index = ["3_1"]
#df.columns = attrs
#
#PlotBarSeries(df, "Mean difference", "Attribute score prediction for 3_1")
#
##%%Does the number of dec=1 impact the way you feel about yourself?
#
#
##%% Does int_corr correlate with shar? or are we not able to evaluate shared interests in 4 minutes?
#int_corr = data[["int_corr"]]
#shar = data[["shar", "shar_o"]]
#df = pd.concat([int_corr, shar], axis=1)
#corr = df.corr()
#PlotHeatmap(corr, "Shared interests", 0, 1)
#
##%%How good are we at predicting the other persons answer
#dec_yes_other = data[data["dec_o"] == 1]
#dec_no_other = data[data["dec_o"] == 0]
#dec_yes = data[data["dec"] == 1]
#dec_no = data[data["dec"] == 0]
#
#dec_yes_other_prob_mean = dec_yes_other["prob"].mean()
#dec_no_other_prob_mean = dec_no_other["prob"].mean()
#
#dec_yes_prob_mean = dec_yes["prob"].mean()
#dec_no_prob_mean = dec_no["prob"].mean()
#
#df = pd.DataFrame([[dec_yes_prob_mean, dec_yes_other_prob_mean], [dec_no_prob_mean, dec_no_other_prob_mean]], columns=["Own decision", "Other decision"], index=["Yes", "No"])
#PlotBarSeries(df,"prob mean-value" ,"Average probability for other say yes for dec=yes and dec=no")
    

#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data_algorithm = data.drop(['match', 'iid', "id","idg", "condtn", "wave", "round", "position", "partner", "pid", "career_c", "sports", "tvsports", 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing','reading', 'tv', 'theater', 'movies','concerts', 'music', 'shopping', 'yoga'], axis=1)
def remove_by_contains(searchString, inputData):
    matching = [s for s in inputData.columns if searchString in s]
    inputData = inputData.drop(matching, axis=1)
    return inputData

# Remove data from "other" person. (The other persons opinion)
data_algorithm = remove_by_contains('_o', data_algorithm)

#Generate full dataset
features_all = data_algorithm.drop('dec', axis=1)
labels_all = data_algorithm.dec

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(features_all, labels_all, test_size=0.15, random_state=42)
X_train_all, X_valid_all, y_train_all, y_valid_all = train_test_split(X_train_all, y_train_all, test_size=0.15, random_state=42)

# Scaling the data
scaler = MinMaxScaler()
X_train_all_scaled = scaler.fit_transform(X_train_all)
X_valid_all_scaled = scaler.transform(X_valid_all)
X_test_all_scaled = scaler.transform(X_test_all)


#%%For both genders with same model
#from sklearn.model_selection import GridSearchCV
#
#from keras.models import Sequential
#from keras.layers import Dropout, Dense, Input
#from time import time 
#
#
#model = Sequential([
#    Input(shape=input_shape),
#    Dropout(rate=0.05),
#    Dense(300, activation="relu"),
#    Dropout(rate=0.2),
#    Dense(100, activation="relu"),
#    Dense(1, kernel_initializer="normal", activation="sigmoid")
#])
#
## Configure the model and start training
## Optimizer (pick the right one)
## batch_size=10, 
#
#model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['binary_crossentropy', 'accuracy', "mean_squared_error"])
#
#epochs = np.linspace(20, 150, 5)
#batch_size = np.linspace(10, 100, 5)
#param_grid = dict(epochs=epochs, batch_size=batch_size)
#start = time()
#random_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring="accuracy", n_jobs=-1)
#random_search.fit(X_train_all_scaled, y_train_all)
#t = time()-start


#history = model.fit(X_train_all_scaled, y_train_all, epochs=60, batch_size=20, verbose=0, validation_data=(X_valid_all_scaled, y_valid_all), use_multiprocessing=True)


#pd.DataFrame(history.history).plot(figsize=(8, 5))
#plt.grid(True)
#plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
#plt.show()
#
#mse_train = model.evaluate(X_train_all_scaled, y_train_all)
#mse_val = model.evaluate(X_valid_all_scaled, y_valid_all)
## Testing
#mse_test = model.evaluate(X_test_all_scaled, y_test_all)
#X_new = X_test_all[:20]
#y_proba = model.predict(X_new)
#
##%%
#pred_dec = model.predict(X_train_all_scaled)

#%% Find optimale lear
from lib.ParamTuning import build_keras_base
from keras.wrappers.scikit_learn import KerasClassifier
from time import time
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler
import math

input_shape = X_train_all.shape[1:]
#hidden_layers = [(300,100,50), (100,20,30), (300,50,50,50)]
#dropout = [(0,0.2,0.2), (0.1,0.4,0.4), (0.4,0.3,0.3)]
models = []
model_keras = KerasClassifier(
            build_fn = build_keras_base, 
            hidden_layers=[300,100], 
            dropout_rate=[0.2,0.2],
            n_input=input_shape[0],
            n_class=1, 
            optimizer=Adam(learning_rate=10**-5),
            default_dropout=0.2)
# specify other extra parameters pass to the .fit
# number of epochs is set to a large number, we'll
# let early stopping terminate the training process
epochs = 100
def my_learning_rate(epoch, lr):
    return lr * np.exp(np.log(10**6)/epochs)
lrs = LearningRateScheduler(my_learning_rate)

results = []

start = time()
early_history = model_keras.fit(X_train_all_scaled, y_train_all, epochs=epochs, callbacks=[lrs], validation_data=(X_valid_all_scaled, y_valid_all), verbose=0, batch_size=100)
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
models = []
#hidden_layers = [(300,100,50), (100,20,30), (300,50,50,50)]
#dropout = [(0,0.2,0.2), (0.1,0.4,0.4), (0.4,0.3,0.3)]
hidden_layers = [(300,100), (1000,200), (1000, 500, 300, 100, 50, 25), (200,50), (2000,1000,500,100,50), (2000,100,50)]
dropout_rates = [0.175,0.225,0.275]
#hidden_layers=[([600])]
#dropout_rates = [0.4]
for layer in hidden_layers:
    for dropout_rate in dropout_rates:
        model_keras = KerasClassifier(
        build_fn = build_keras_base, 
        hidden_layers=layer, 
        n_input=input_shape[0],
        n_class=1, 
        default_dropout=dropout_rate,
        optimizer=Adam(learning_rate=0.0015, ))
        models.append(model_keras)

# number of epochs is set to a large number, we'll
# let early stopping terminate the training process
epochs = 200
def step_decay(epoch): 
   initial_lrate = 0.0015
   drop = 0.6
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate
lrate = LearningRateScheduler(step_decay)
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5  , patience=10, min_lr=0.00005, cooldown=5)
early_stop = EarlyStopping(monitor='val_loss', patience=40, min_delta=0.001, restore_best_weights=True)
results = []
for model in models: 
    start = time()
    early_history = model.fit(X_train_all_scaled, y_train_all, epochs=epochs, callbacks=[early_stop, reduce_lr, lrate], validation_data=(X_valid_all_scaled, y_valid_all), verbose=1, batch_size=500)
    t = time()-start
    results.append(early_history)
    
#y_pred = models[0].predict(X_test_all_scaled)
#y_pred = y_pred[:,0]
#from lib.Plots import FullReport, PlotPerformanceMatrix
#conf_matrix = PlotPerformanceMatrix(y_pred, y_test_all)

    #b0, m0 = FullReport(rs_keras, X_test_all_scaled, y_test_all, t)
    #results.append([b0,m0], axis=1)
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
#Make confusion matrix for model performance
cf_matrix_val_acc = pd.DataFrame(index=[str(dr) for dr in dropout_rates], columns=[str(l) for l in hidden_layers])
cf_matrix_val_loss = pd.DataFrame(index=[str(dr) for dr in dropout_rates], columns=[str(l) for l in hidden_layers])
for index, hidden_layer in enumerate(hidden_layers):
    for i in np.arange(len(dropout_rates)):
        cf_matrix_val_acc[str(hidden_layer)][str(dropout_rates[i])] = pd.DataFrame(results[(index * len(dropout_rates)) + i].history)["val_accuracy"].max()
        cf_matrix_val_loss[str(hidden_layer)][str(dropout_rates[i])] = pd.DataFrame(results[(index * len(dropout_rates)) + i].history)["val_loss"].min()
from lib.Plots import PlotConfusionMatrix
PlotConfusionMatrix(cf_matrix_val_acc.astype(np.float64, copy=False))
PlotConfusionMatrix(cf_matrix_val_loss.astype(np.float64, copy=False))
#%%

#Test for different learning rates
#Try with singel large hidden layer of neurons
#Have many neurons and many layers and use early stopping and other ruglarization techniques to prevent overfitting
#Try with pyramid shape of neurons (many to few) and or first large and all others same size
#traning iterations does not need to be tweaked - use eraly stopping instead

# random search's parameter:
# specify the options and store them inside the dictionary
# batch size and training method can also be hyperparameters, 
# but it is fixed
#dropout_rate_opts  = [0, 0.2, 0.5]
#hidden_layers_opts = [[64, 64, 64, 64], [32, 32, 32, 32, 32], [100, 100, 100]]
#l2_penalty_opts = [0.01, 0.1, 0.5]
#keras_param_options = {
#    'hidden_layers': hidden_layers_opts,
#    'dropout_rate': dropout_rate_opts,  
#    'l2_penalty': l2_penalty_opts
#}


#%%
#    # Load the dataset
## Create model for KerasClassifier
#def create_model(hparams1=dvalue,
#                 hparams2=dvalue,
#                 ...
#                 hparamsn=dvalue):
#    # Model definition
#    ...
#
#model = KerasClassifier(build_fn=create_model) 
#
## Specify parameters and distributions to sample from
#hparams1 = randint(1, 100)
#hparams2 = ['elu', 'relu', ...]
#...
#hparamsn = uniform(0, 1)
#
## Prepare the Dict for the Search
#param_dist = dict(hparams1=hparams1, 
#                  hparams2=hparams2, 
#                  ...
#                  hparamsn=hparamsn)
#
## Search in action!
#n_iter_search = 16 # Number of parameter settings that are sampled.
#random_search = RandomizedSearchCV(estimator=model, 
#                                   param_distributions=param_dist,
#                                   n_iter=n_iter_search,
#                                   n_jobs=, 
#								   cv=, 
#								   verbose=)
#random_search.fit(X, Y)
#
## Show the results
#print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
#means = random_search.cv_results_['mean_test_score']
#stds = random_search.cv_results_['std_test_score']
#params = random_search.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))