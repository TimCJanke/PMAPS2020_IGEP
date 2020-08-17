"""
@author: Tim Janke, Energy Information Networks & Systems Lab @ TU Darmstadt, Germany

A collection of helper functions for loading and plotting data.
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# load data
def load_data(date_from, date_to):
    # Price data
    data_price = pd.read_csv("Price_2015_2017.csv", delimiter=';',index_col=0)
    data_price.index = pd.to_datetime(data_price.index, infer_datetime_format = False, format = '%d.%m.%Y %H:%M')
    data_price = data_price.loc[date_from:date_to]
    
    # Predictions
    data_pred = pd.read_csv("EnsemblePredictions_2016_2017.csv", delimiter=';',index_col=0)
    data_pred.index = pd.to_datetime(data_pred.index, infer_datetime_format = False , format = '%d.%m.%Y %H:%M')
    data_pred = data_pred.loc[date_from:date_to]
    
    return data_pred, data_price


# prepare data in right format with train and test splits according to dates_train and dates_test
def make_training_data(data_pred, data_price, dates_train, dates_test, dim=24, scale_data=False):
    prices = data_price.copy()
    pred = data_pred.copy()
    if scale_data:
        scaler = preprocessing.StandardScaler().fit(prices.loc[dates_train].values)
        prices.iloc[:,[0]] = scaler.transform(prices.values)
        for i in range(pred.shape[1]):
             pred.iloc[:,[i]] = scaler.transform(pred.values[:,[i]])
        
    ens_mean = pred.mean(axis=1)
    ens_delta = 0.5 * (pred.max(axis=1) - pred.min(axis=1))
    
    x_train = [ens_mean.loc[dates_train].values.reshape((-1, dim, 1)), ens_delta[dates_train].values.reshape((-1, dim, 1))]
    x_test =  [ens_mean.loc[dates_test].values.reshape((-1, dim, 1)), ens_delta[dates_test].values.reshape((-1, dim, 1))]

    y_train = prices.loc[dates_train].values.reshape((-1, dim, 1))
    y_test = prices.loc[dates_test].values.reshape((-1, dim, 1))
    
    if scale_data:
        return x_train, y_train, x_test, y_test, scaler
    else:
        return x_train, y_train, x_test, y_test
    


# plot predictive scenarios, ensemble predictions, and true values
def plot_scenarios(y_true, y_pred_scenarios, y_pred_ens, n_samples=25):
    plt.figure(figsize=(25/2.54,14/2.54))
    plt.plot(y_pred_scenarios.values[:,0:min(n_samples,25)], linewidth = 0.5, alpha=0.8)
    plt.plot(y_true.values, label = 'true price', color = 'blue', linewidth = 2.0)
    for m in range(y_pred_ens.shape[1]-1):
        plt.plot(y_pred_ens.values[:,m], color = 'black', linewidth = 1)
    plt.plot(y_pred_ens.values[:,m+1], color = 'black', linewidth = 1,label='ensemble models')
    plt.legend(loc='upper left',prop={'size':8})
    plt.ylabel('€/MWh')
    plt.xticks(np.arange(0,y_true.shape[0],24))
    ax = plt.gca()
    ax.set_xticklabels(list(pd.date_range(start=y_true.index[0].date(),end=y_true.index[-1].date(),freq='D').date))
    ax.set_facecolor('gainsboro')
    plt.xlim((0,y_true.shape[0]))
    plt.title("Scenarios from predictive distribution")
    plt.tight_layout()
    
    
# plot predictive quantiles, ensemble predictions, and true values 
def plot_quantiles(y_true, y_pred_quantiles, y_pred_ens):
    qs1 = [0, 4, 9, 19, 29, 39, 49]
    qs2 = [98, 94, 89, 79, 69, 59, 50]
    xx = np.arange(1,len(y_true)+1)
    fig, ax = plt.subplots(figsize=(25/2.54,14/2.54))
    for i in range(len(qs1)):
        alpha = 0.5*(i+1)/len(qs1) # Modify the alpha value for each iteration.
        ax.fill_between(xx, y_pred_quantiles.values[:,qs2[i]], y_pred_quantiles.values[:,qs1[i]], color='red', alpha=alpha)
    for m in range(y_pred_ens.shape[1]-1):
        ax.plot(xx, y_pred_ens.iloc[:,m].values, color = 'black', linewidth = 0.5)
    ax.plot(xx, y_pred_ens.iloc[:,m].values, color = 'black', linewidth = 0.5, label="ensemble models")
    ax.plot(xx, y_true.values, label = 'true price', color = 'blue', linewidth=1)
    ax.set_xticks(np.arange(0,y_true.shape[0],24))
    ax.set_xticklabels(list(pd.date_range(start=y_true.index[0].date(),end=y_true.index[-1].date(),freq='D').date))
    ax.set_facecolor('gainsboro')
    plt.xlim((0,y_true.shape[0]))
    plt.title("Predictive quantiles")
    plt.ylabel('€/MWh')
    plt.legend()
    plt.tight_layout()


# plot PIT histograms for each dimension
def PIT_histograms(y_true, y_pred_quantiles, dims, nrows=6, ncols=4):
    fig, axes = plt.subplots(nrows, ncols, sharey=True, sharex=True, figsize=(30/2.54,20/2.54))
    bins = np.array([5,15,25,35,45,55,65,75,85,95])*0.01
    for i, dim in enumerate(dims):
        idx_h = np.arange(0,y_true.shape[0],24)+dim
        y_true_d = y_true[idx_h,:]
        aux = y_true_d - np.concatenate([np.zeros((len(y_true_d),1))+min(y_true_d)-1,y_pred_quantiles[idx_h,:],np.zeros((len(y_true_d),1))+max(y_true_d)+1],axis=1)
        pit_values = np.argmax(aux<0, axis=1)
        pit_hist_vals = np.histogram(pit_values, bins = [0,10,20,30,40,50,60,70,80,90,100])
        ax = axes.T.flatten()[i]        
        ax.bar(bins, pit_hist_vals[0]/len(y_true_d), width=0.095)
        ax.plot(np.linspace(0,1,len(bins)),np.ones(len(bins))/len(bins),linestyle = ':', color = 'black',lw=0.75)
        ax.set_xticks(np.linspace(0,1,len(bins)+1))
        ax.set_ylim(0,0.15)
        ax.set_xlim(0,1)
        ax.text(x=0.065,y=0.13,s='h'+str(dim+1),fontdict={'color': 'black', 
                                                          'fontsize': 8, 
                                                          'ha': 'center', 
                                                          'va': 'center', 
                                                          'bbox': dict(boxstyle="round", fc="white", ec="white", pad=0.1)})
        ax.set_facecolor('gainsboro')
    fig.tight_layout()