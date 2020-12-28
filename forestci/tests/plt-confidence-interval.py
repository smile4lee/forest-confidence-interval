#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# import __init__ to add py scripts root dir into system path
import __init__

from osgeo import gdal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

from scipy.stats import gaussian_kde

import joblib
import forestci as fci


from sklearn.ensemble import RandomForestRegressor

## Ignore warnings
# import warnings
# warnings.filterwarnings('ignore')


# In[2]:


var = "rs_clim_weat"
var = "rs"

in_csv = r"/root/onedrive-ttu/research/brsoy.hpcc/train.{0}/tuned/out.corr/brsoy-yield-input.training.pred.csv".format(var)
test_in_csv = r"/root/onedrive-ttu/research/brsoy.hpcc/train.{0}/tuned/out.corr/brsoy-yield-input.test.pred.csv".format(var)
jpg_dir = r"/root/onedrive-ttu/research/brsoy.hpcc/data/{0}/jpg-bias-correction.confidence-interval".format(var)
model_file = r"/root/onedrive-ttu/research/brsoy.hpcc/train.{0}/tuned/out.corr/rf.corr.model".format(var)

train_csv = r"/root/onedrive-ttu/research/brsoy.hpcc/data/{0}/brsoy-yield-input.training.csv".format(var)
test_csv = r"/root/onedrive-ttu/research/brsoy.hpcc/data/{0}/brsoy-yield-input.test.csv".format(var)

np.set_printoptions(precision=3)
pd.options.display.float_format = '{:.3f}'.format



# In[3]:


# rs_clim_weat
# feature_columns = ['C_cld_10', 'C_cld_11', 'C_cld_12', 'C_cld_4', 'C_cld_5', 'C_dtr_1', 'C_dtr_10', 'C_dtr_11', 'C_dtr_12', 'C_dtr_2', 'C_dtr_3', 'C_dtr_4', 'C_dtr_5', 'C_pet_10', 'C_pet_11', 'C_pet_12', 'C_pet_3', 'C_pet_4', 'C_pet_5', 'C_pre_10', 'C_pre_11', 'C_pre_12', 'C_pre_2', 'C_pre_4', 'C_pre_5', 'C_tmn_1', 'C_vap_10', 'C_wet_10', 'C_wet_11', 'C_wet_4', 'C_wet_5', 'L_EVI_avmin25', 'L_EVI_deltaloss', 'L_EVI_maxgain', 'L_EVI_mingain', 'L_EVI_minloss', 'L_EVI_sd', 'L_NDVI_av2550', 'L_NDVI_deltagain', 'L_NDVI_deltaloss', 'L_NDVI_maxloss', 'L_NDVI_prc75', 'L_NDWI_avmin10', 'L_NDWI_deltagain', 'L_NDWI_deltaloss', 'L_NDWI_max', 'L_NDWI_minloss', 'L_NDWI_sd', 'L_green_sd', 'L_nir_maxloss', 'L_nir_min', 'L_nir_prc75', 'L_nir_sd', 'L_rankNDVI_green_av90max', 'L_rankNDVI_swir1_av5090', 'L_rankNDVI_swir2_av5090', 'L_rankNDVI_swir2_max', 'L_swir1_maxloss', 'L_swir1_minloss', 'L_swir2_deltagain', 'L_swir2_sd', 'M_EVI_deltaloss', 'M_EVI_minloss', 'M_NDVI_av2550', 'M_NDVI_av90max', 'M_NDVI_deltagain', 'M_NDVI_maxgain', 'M_NDVI_sd', 'M_NDWI_deltagain', 'M_NDWI_min', 'M_NDWI_reg', 'M_NDWI_sd', 'M_nir_reg', 'M_rankNDVI_green_av75max', 'M_rankNDVI_nir_av75max', 'M_rankNDVI_swir1_av75max', 'M_swir1_av1025', 'T_dem', 'T_slope', 'W_cld_1', 'W_cld_10', 'W_cld_11', 'W_cld_12', 'W_cld_2', 'W_cld_3', 'W_cld_4', 'W_cld_5', 'W_dtr_1', 'W_dtr_10', 'W_dtr_11', 'W_dtr_12', 'W_dtr_2', 'W_dtr_3', 'W_dtr_4', 'W_dtr_5', 'W_pet_10', 'W_pet_11', 'W_pet_12', 'W_pet_2', 'W_pet_3', 'W_pet_4', 'W_pre_1', 'W_pre_10', 'W_pre_11', 'W_pre_12', 'W_pre_2', 'W_pre_3', 'W_pre_4', 'W_pre_5', 'W_tmn_1', 'W_tmn_10', 'W_tmn_12', 'W_tmn_2', 'W_tmn_3', 'W_tmn_4', 'W_tmn_5', 'W_tmx_3', 'W_vap_2', 'W_wet_1', 'W_wet_10', 'W_wet_11', 'W_wet_12', 'W_wet_2', 'W_wet_3', 'W_wet_4', 'W_wet_5']

# rs
feature_columns = ['L_EVI_avmin25', 'L_EVI_deltaloss', 'L_EVI_maxgain', 'L_EVI_mingain', 'L_EVI_minloss', 'L_EVI_sd', 'L_NDVI_av2550', 'L_NDVI_deltagain', 'L_NDVI_deltaloss', 'L_NDVI_maxloss', 'L_NDVI_prc75', 'L_NDWI_avmin10', 'L_NDWI_deltagain', 'L_NDWI_deltaloss', 'L_NDWI_max', 'L_NDWI_minloss', 'L_NDWI_sd', 'L_green_sd', 'L_nir_maxloss', 'L_nir_min', 'L_nir_prc75', 'L_nir_sd', 'L_rankNDVI_green_av90max', 'L_rankNDVI_swir1_av5090', 'L_rankNDVI_swir2_av5090', 'L_rankNDVI_swir2_max', 'L_swir1_maxloss', 'L_swir1_minloss', 'L_swir2_deltagain', 'L_swir2_sd', 'M_EVI_deltaloss', 'M_EVI_minloss', 'M_NDVI_av2550', 'M_NDVI_av90max', 'M_NDVI_deltagain', 'M_NDVI_maxgain', 'M_NDVI_sd', 'M_NDWI_deltagain', 'M_NDWI_min', 'M_NDWI_reg', 'M_NDWI_sd', 'M_nir_reg', 'M_rankNDVI_green_av75max', 'M_rankNDVI_nir_av75max', 'M_rankNDVI_swir1_av75max', 'M_swir1_av1025', 'T_dem', 'T_slope']


# In[4]:


df_input_train = pd.read_csv(train_csv)
print("shape: ", df_input_train.shape)
df_input_train.head(3)


# In[5]:


df_input_test = pd.read_csv(test_csv)
print("shape: ", df_input_test.shape)
df_input_test.head(3)


# In[6]:


df = pd.read_csv(in_csv)
print("shape: ", df.shape)
df.head(3)


# In[7]:


df_test = pd.read_csv(test_in_csv)
print("shape: ", df_test.shape)
df_test.head(3)


Y_train = df_input_train['yield'].astype(int)
Y_train


# In[15]:


X_train = df_input_train.drop(['year', 'yield', 'BRMID', 'npix'], axis=1)[feature_columns]
X_train


# In[16]:


Y_test = df_input_test['yield'].astype(int)
Y_test


# In[17]:


X_test = df_input_test.drop(['year', 'yield', 'BRMID', 'npix'], axis=1)[feature_columns]
X_test


# In[18]:


# model = joblib.load(model_file)
# n_trees = 1000
# model.n_estimators = n_trees

# model parameters
# max_depth=18
# max_features=100
# min_samples_leaf=10
# min_samples_split=40
# n_estimators=100

# rs
max_depth=16
max_features=25
min_samples_leaf=10
min_samples_split=40
n_estimators=100

rf = RandomForestRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, max_features=max_features,
                           n_estimators=100, oob_score=True, random_state=42, n_jobs=-1)

# print("start:", func.now())
# # Train the model on training data
# rf.fit(X_train, Y_train)
# print("  end:", func.now())
# 
# print(rf.get_params())
# 
# y_pred_test = rf.predict(X_test)


# In[21]:


import csv
import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.model_selection as xval
import forestci as fci

import numpy as np

# from econml.sklearn_extensions.ensemble import SubsampledHonestForest
from forestci.ensemble import SubsampledHonestForest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


# n_mpg_x = xscaler.fit_transform(mpg_X)
# n_mpg_y = yscaler.fit_transform(mpg_Y.reshape(-1, 1))

# regr = SubsampledHonestForest(n_estimators=200, random_state=42)

# regr.fit(X_train, Y_train)
# regr.feature_importances_
# mpg_y_hat = regr.predict(X_test)
# interval = regr.predict_interval(X_test, alpha=.05)
# stderr = regr.prediction_stderr(X_test)
# score = regr.score(X_test, Y_test)


# In[25]:

rfr = joblib.load(model_file)

def main():

    pred_rfr = rfr.predict(X_test)
    print(rfr.feature_importances_)



    regr = SubsampledHonestForest(rfr=rfr)
    print(type(regr))
    regr.fit(X_train, Y_train)
    # regr.feature_importances_
    pred_regr = regr.predict(X_test)
    interval = regr.predict_interval(X_test, alpha=.05)
    stderr = regr.prediction_stderr(X_test)
    score = regr.score(X_test, Y_test)

    df = pd.DataFrame()
    df['pred_rfr'] = pred_rfr
    df['pred_regr'] = pred_regr
    df['lower'] = interval[0]
    df['upper'] = interval[1]
    df['stderr'] = stderr
    print("score: %s" % score)
    print(df.describe())


def do_fci():
    # Calculate the variance
    mpg_V_IJ_unbiased, pred_mean_t = fci.random_forest_error(rfr, X_train, X_test)
    # mpg_V_IJ_unbiased = fci.random_forest_error(rfr, X_train, X_test, calibrate=False)
    print(mpg_V_IJ_unbiased.shape)
    print(mpg_V_IJ_unbiased)

    pred_rf = rfr.predict(X_test)

    import pandas as pd
    df = pd.DataFrame()
    df['pred_rf'] = pred_rf
    df['pred_mean_t'] = pred_mean_t
    df['mpg_V_IJ_unbiased'] = mpg_V_IJ_unbiased
    df['mpg_V_IJ_unbiased_sqrt'] = np.sqrt(mpg_V_IJ_unbiased)
    pd.options.display.max_columns = df.shape[1]
    print(df.describe())


if __name__ == '__main__':
    # main()

    do_fci()
