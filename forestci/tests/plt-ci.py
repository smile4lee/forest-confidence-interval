#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

import csv
import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.model_selection as xval
import forestci as fci

import numpy as np

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


# In[2]:


var = "rs_clim_weat"
# var = "rs"

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
feature_columns = ['C_cld_10', 'C_cld_11', 'C_cld_12', 'C_cld_4', 'C_cld_5', 'C_dtr_1', 'C_dtr_10', 'C_dtr_11', 'C_dtr_12', 'C_dtr_2', 'C_dtr_3', 'C_dtr_4', 'C_dtr_5', 'C_pet_10', 'C_pet_11', 'C_pet_12', 'C_pet_3', 'C_pet_4', 'C_pet_5', 'C_pre_10', 'C_pre_11', 'C_pre_12', 'C_pre_2', 'C_pre_4', 'C_pre_5', 'C_tmn_1', 'C_vap_10', 'C_wet_10', 'C_wet_11', 'C_wet_4', 'C_wet_5', 'L_EVI_avmin25', 'L_EVI_deltaloss', 'L_EVI_maxgain', 'L_EVI_mingain', 'L_EVI_minloss', 'L_EVI_sd', 'L_NDVI_av2550', 'L_NDVI_deltagain', 'L_NDVI_deltaloss', 'L_NDVI_maxloss', 'L_NDVI_prc75', 'L_NDWI_avmin10', 'L_NDWI_deltagain', 'L_NDWI_deltaloss', 'L_NDWI_max', 'L_NDWI_minloss', 'L_NDWI_sd', 'L_green_sd', 'L_nir_maxloss', 'L_nir_min', 'L_nir_prc75', 'L_nir_sd', 'L_rankNDVI_green_av90max', 'L_rankNDVI_swir1_av5090', 'L_rankNDVI_swir2_av5090', 'L_rankNDVI_swir2_max', 'L_swir1_maxloss', 'L_swir1_minloss', 'L_swir2_deltagain', 'L_swir2_sd', 'M_EVI_deltaloss', 'M_EVI_minloss', 'M_NDVI_av2550', 'M_NDVI_av90max', 'M_NDVI_deltagain', 'M_NDVI_maxgain', 'M_NDVI_sd', 'M_NDWI_deltagain', 'M_NDWI_min', 'M_NDWI_reg', 'M_NDWI_sd', 'M_nir_reg', 'M_rankNDVI_green_av75max', 'M_rankNDVI_nir_av75max', 'M_rankNDVI_swir1_av75max', 'M_swir1_av1025', 'T_dem', 'T_slope', 'W_cld_1', 'W_cld_10', 'W_cld_11', 'W_cld_12', 'W_cld_2', 'W_cld_3', 'W_cld_4', 'W_cld_5', 'W_dtr_1', 'W_dtr_10', 'W_dtr_11', 'W_dtr_12', 'W_dtr_2', 'W_dtr_3', 'W_dtr_4', 'W_dtr_5', 'W_pet_10', 'W_pet_11', 'W_pet_12', 'W_pet_2', 'W_pet_3', 'W_pet_4', 'W_pre_1', 'W_pre_10', 'W_pre_11', 'W_pre_12', 'W_pre_2', 'W_pre_3', 'W_pre_4', 'W_pre_5', 'W_tmn_1', 'W_tmn_10', 'W_tmn_12', 'W_tmn_2', 'W_tmn_3', 'W_tmn_4', 'W_tmn_5', 'W_tmx_3', 'W_vap_2', 'W_wet_1', 'W_wet_10', 'W_wet_11', 'W_wet_12', 'W_wet_2', 'W_wet_3', 'W_wet_4', 'W_wet_5']

# rs
# feature_columns = ['L_EVI_avmin25', 'L_EVI_deltaloss', 'L_EVI_maxgain', 'L_EVI_mingain', 'L_EVI_minloss', 'L_EVI_sd', 'L_NDVI_av2550', 'L_NDVI_deltagain', 'L_NDVI_deltaloss', 'L_NDVI_maxloss', 'L_NDVI_prc75', 'L_NDWI_avmin10', 'L_NDWI_deltagain', 'L_NDWI_deltaloss', 'L_NDWI_max', 'L_NDWI_minloss', 'L_NDWI_sd', 'L_green_sd', 'L_nir_maxloss', 'L_nir_min', 'L_nir_prc75', 'L_nir_sd', 'L_rankNDVI_green_av90max', 'L_rankNDVI_swir1_av5090', 'L_rankNDVI_swir2_av5090', 'L_rankNDVI_swir2_max', 'L_swir1_maxloss', 'L_swir1_minloss', 'L_swir2_deltagain', 'L_swir2_sd', 'M_EVI_deltaloss', 'M_EVI_minloss', 'M_NDVI_av2550', 'M_NDVI_av90max', 'M_NDVI_deltagain', 'M_NDVI_maxgain', 'M_NDVI_sd', 'M_NDWI_deltagain', 'M_NDWI_min', 'M_NDWI_reg', 'M_NDWI_sd', 'M_nir_reg', 'M_rankNDVI_green_av75max', 'M_rankNDVI_nir_av75max', 'M_rankNDVI_swir1_av75max', 'M_swir1_av1025', 'T_dem', 'T_slope']


# In[4]:


df_input_train = pd.read_csv(train_csv)
print("shape: ", df_input_train.shape)
# df_input_train.head(3)
df_input_test = pd.read_csv(test_csv)
print("shape: ", df_input_test.shape)
#df_input_test.head(3)


# In[5]:


df = pd.read_csv(in_csv)
print("shape: ", df.shape)
# df.head(3)
df_test = pd.read_csv(test_in_csv)
print("shape: ", df_test.shape)
# df_test.head(3)


# In[6]:


Y_train = df_input_train['yield'].astype(int)
X_train = df_input_train.drop(['year', 'yield', 'BRMID', 'npix'], axis=1)[feature_columns]
Y_test = df_input_test['yield'].astype(int)
X_test = df_input_test.drop(['year', 'yield', 'BRMID', 'npix'], axis=1)[feature_columns]


# In[7]:


rfr = joblib.load(model_file)


# In[19]:

n_trees = 100

def do_fci():
    # Calculate the variance
    # mpg_V_IJ_unbiased = fci.random_forest_error(rfr, X_train, X_test)
    rfr.n_estimators = n_trees
    rfr.fit(X_train, Y_train)
    pred_test = rfr.predict(X_test).round(0).astype(int)
    mpg_V_IJ_unbiased = fci.random_forest_error(rfr, X_train, X_test)
    # mpg_V_IJ_unbiased = fci.random_forest_error(rfr, X_train, X_test, calibrate=False)

    df_test['pred_test'] = pred_test
    df_test['mpg_V_IJ_unbiased'] = mpg_V_IJ_unbiased
    df_test['mpg_V_IJ_unbiased_sqrt'] = np.sqrt(mpg_V_IJ_unbiased)
    # df_test['lower'] = interval[0]
    # df_test['upper'] = interval[1]
    # df_test['diff'] = df_test['yield_pred'] - mpg_y_hat
    # df_test['stderr'] = stderr

    pd.options.display.max_columns = df_test.shape[1]
    print(df_test.describe())
    out_csv = r"out/out.{0}.csv".format(n_trees)
    df_test.describe().to_csv(out_csv, index=True, header=True, sep=',')


if __name__ == '__main__':
    # main()

    do_fci()


