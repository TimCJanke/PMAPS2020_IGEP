"""
@author: Tim Janke, Energy Information Networks & Systems Lab @ TU Darmstadt, Germany.

Main script to reproduce
Janke & Steinke (2020): "Probabilistic multivariate electricity price forecasting using implicit generative ensemble post-processing"
https://arxiv.org/abs/2005.13417

IGEP models can be used to generate samples an implicit multivariate predictive distribution given the predictions of an 
ensemble of point forecasting models.


This script requires the following custom modules:
    - ImplicitGenerativeEnsemblePostprocessing: Module that implements class for training and prediction of IGEP models
    - scoringRules: Module for calculating scores, mainly wraps R functions from the scoringRules package
    - utils: A collection of helper functions for loading and plotting data


This script loads the following data:
    - Price_2015_2017.csv: csv file containing prices from EPEX spot between 2015/01/01 and 2017/12/01
    - EnsemblePredictions_2016_2017.csv: csv file containing point predictions for EPEX spot between 2016/01/01 and 2017/12/01

"""
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from scoringRules import es_sample, crps_sample, pinball_score
from igep import ImplicitGenerativeEnsemblePostprocessing as IGEP
import utils


#%% OPTIONS
# set initial training and test dates
date_train_start = datetime.datetime(2016,1,1,0)
date_train_end = datetime.datetime(2016,12,31,23)
date_test_start = datetime.datetime(2017,1,1,0)
date_test_end = datetime.datetime(2017,12,31,23)

REFIT_PERIOD = 1000 # days between retraining model in moving window, set to large number for no refit

# set model hyper parameters
N_SAMPLES_TRAIN = 25        # number of samples drawn during training
DIM_LATENT = 10             # number of non-adaptive latent variables
EPOCHS = 100                # number of epochs in model training
BATCH_SIZE = 4              # batch size in model training
LATENT_DIST = "uniform"     # family of latent variable distributions

# misc options
VERBOSE = 0                     # determines how much info is given about fitting (can be 0, 1, or 2.
                                # If 1 or 2, will plot model summary, shape, and info  on epochs during training)
SCALE_DATA = True               # scale data before model fitting
PLOT_LEARNING_CURVE = False     # if True, the model's learning curve is plotted
N_SAMPLES_TEST = 1000           # number of samples used for calculating scores
DIM = 24                        # dimension of target values
TAUS = np.linspace(1,99,99)/100 # quantiles to evaluate
PLOT_RESULTS = True             # plot examplary predictions and PIT histograms


#%% LOAD DATA AND TRAIN MODEL
#load data
data_ens, data_price = utils.load_data(date_train_start, date_test_end)
y_test = data_price.loc[date_test_start:date_test_end]
y_test_ens = data_ens.loc[date_test_start:date_test_end]

# set initial train-test split
train_start_tmp = date_train_start
train_end_tmp = date_train_end
test_start_tmp = date_test_start
test_end_tmp = test_start_tmp + datetime.timedelta(hours=REFIT_PERIOD*DIM-1)
if test_end_tmp >= date_test_end:
    test_end_tmp = date_test_end
dt = datetime.timedelta(hours=REFIT_PERIOD*DIM)

pbar = tqdm(total=np.ceil((date_test_end-date_test_start).days/REFIT_PERIOD)) # progressbar
S = [] # to store predictions
while True:
    # get training and test data
    dates_train_tmp = pd.date_range(start=train_start_tmp, end=train_end_tmp, freq='H')
    dates_test_tmp  = pd.date_range(start=test_start_tmp, end=test_end_tmp, freq='H')    
    if SCALE_DATA:
        x_train_tmp, y_train_tmp, x_test_tmp, y_test_tmp, scaler = utils.make_training_data(data_ens, data_price, dates_train_tmp, dates_test_tmp,scale_data=SCALE_DATA)
    else:
        x_train_tmp, y_train_tmp, x_test_tmp, y_test_tmp = utils.make_training_data(data_ens, data_price, dates_train_tmp, dates_test_tmp,scale_data=SCALE_DATA)
        
    
    # initialize model
    mdl = IGEP(dim_out=DIM, 
               dim_in_mean=x_train_tmp[0].shape[-1], 
               dim_latent=DIM_LATENT, 
               n_samples_train=N_SAMPLES_TRAIN, 
               latent_dist=LATENT_DIST)
    
    # fit model
    mdl.fit(x=x_train_tmp, 
            y=y_train_tmp, 
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS, 
            verbose=VERBOSE, 
            validation_split=0.0,
            validation_data=None,         
            sample_weight=None,
            plot_learning_curve=PLOT_LEARNING_CURVE)
    
    # predict and append to list
    if SCALE_DATA:
        S.append(scaler.inverse_transform(mdl.predict(x_test_tmp, N_SAMPLES_TEST)))
    else:
        S.append(mdl.predict(x_test_tmp, N_SAMPLES_TEST))

    
    # move rolling window
    train_start_tmp = train_start_tmp + dt
    train_end_tmp = train_end_tmp + dt
    test_start_tmp = test_start_tmp + dt
    test_end_tmp = test_end_tmp + dt
    
    pbar.update(1)
    
    # break if we are out of data
    if test_start_tmp > date_test_end:
        break
    # make sure we don't go to far
    if test_end_tmp >= date_test_end:
        test_end_tmp = date_test_end

S = np.concatenate(S,axis=0)
pbar.close()


#%% SCORES
print("\nComputing Scores...")
y_pred_samples = pd.DataFrame(np.reshape(S, (S.shape[0]*S.shape[1],-1)), index=y_test.index)
y_pred_mean = y_pred_samples.mean(axis=1).to_frame(name="price")
y_pred_median = y_pred_samples.median(axis=1).to_frame(name="price")
y_pred_quantiles = y_pred_samples.quantile(TAUS, axis=1).T

MAE = np.mean(abs(y_test.values - y_pred_median.values))
RMSE = np.sqrt(np.mean((y_test.values - y_pred_mean.values)**2))
PB = pinball_score(y=y_test.values, dat=y_pred_quantiles.values, taus=TAUS)
CRPS = crps_sample(y=np.ravel(y_test.values), dat=y_pred_samples.values)
ES = es_sample(y=np.reshape(y_test.values, (-1,DIM)), dat=S)

print("\nTest set MAE:")
print(MAE)
print("\nTest set RMSE:")
print(RMSE)
print("\nTest set Pinball Loss:")
print(PB)
print("\nTest set CRPS:")
print(CRPS)
print("\nTest set Energy Score:")
print(ES)


#%% PLOTS
if PLOT_RESULTS:    
    a = datetime.datetime(2017, 1, 29, 2, 0)
    b = datetime.datetime(2017, 2, 2, 22, 0)
    utils.plot_scenarios(y_test.loc[a:b,:], y_pred_samples.loc[a:b,:], y_test_ens.loc[a:b,], n_samples=50)
    utils.plot_quantiles(y_test.loc[a:b,:], y_pred_quantiles.loc[a:b,:], y_test_ens.loc[a:b,:])

    a = datetime.datetime(2017, 11, 18, 2, 0)
    b = datetime.datetime(2017, 11, 22, 22, 0)
    utils.plot_scenarios(y_test.loc[a:b,:], y_pred_samples.loc[a:b,:], y_test_ens.loc[a:b,], n_samples=50)
    utils.plot_quantiles(y_test.loc[a:b,:], y_pred_quantiles.loc[a:b,:], y_test_ens.loc[a:b,:])
    
    utils.PIT_histograms(y_test.values, y_pred_quantiles.values, np.arange(0,DIM,1))
