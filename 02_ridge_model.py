# %%
# %%
import numpy as np
import pandas as pd
import sklearn as ktl
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from mlxtend.plotting import scatterplotmatrix 

from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import scipy.stats as stats

from condensedModels import RidgeModel

# Set up features and labels names 
feature_name = ["Log EGDP", 
            "Agricultural land", 
            "Urban land",
            "Percentage of urban land",
            "Log Population",
            "Log PM2.5",
            "log Land temperature",
            "Log NTL",
            "Log Distance to road",
            "Log GHSL",
            "Distance to Diamonds Extraction site", 
            "Log Malaria rate",
            "Log Distance to water",
            "Log Elevation",
            "Log Distance to drug site",
            "Photovoltaic potential",
            "Log Access to city",
            "Log Slope",
            "Log Precipitation",
            "Log Population Density",
            "Croplands",
            "Forest land",
            "Urban Built up",
            "Savannas and Grasslands",
            "shrublands",
            "Vegetation",
            'Air Temperature']

feature_code = [
        'lnEGDPpc2012', 
        'lnagr_land2012', 
        'lnurb_land2012',
        'perUrb_land2012',
        'ln_tr400_pop2012',
        'ln_pm25_2012',
        'ln_land_temp2012',
        'ln_t400NTLpc2012',
        'ln_dist_road2017',
        'ln_ghsl2015',
        'dist_diamond2015', 
        'ln_mal_inci_rt_mean',
        'ln_dist_water2017mean',
        'ln_elev2017mean',
        'ln_dist_drug2017mean',
        'photov2019mean',
        'ln_access2016mean',
        'ln_slope500m2017mean',
        'ln_precCRU2012mean',
        "ln_density_pop2015count",
        "land_per_area_2012_croplands",
        "land_per_area_2012_full_forest",
        "land_per_area_2012_urban_and_builtup",
        "land_per_area_2012_full_savannas_grasslands",
        "land_per_area_2012_full_shrublands",
        "land_per_area_2012_cropland_natural_vegetation_mosaic",
        'airTemp2012.mean']

label_description = [
        "SDG 1: No poverty",
        "SDG 2: Zero hunger",
        "SDG 3: Good health and well-being",
        "SDG 4: Quality education",
        "SDG 5: Gender equality",
        "SDG 6: Clean water and sanitization",
        "SDG 7: Affordable and clean energy",
        "SDG 8: Decent work and economic growth",
        "SDG 9: Industry, innovation and infraestructure",
        "SDG 10: Reduced inequalities",
        "SDG 11: Sustainable cities and communities",
        "SDG 13: Climate action",
        "SDG 15: Life on land",
        "SDG 16: Peace, justice and strong institutions",
        "SDG 17: Parnerships for the goals",
        "Sustainable Development Index"
]
# %%
# TODO: Create a functiton that deletes the values of the dataframes ridge_predict, ridge_results, opt_ridge_results
# %%
def run_all(): 
 # FIXME: When called - does not have access to the modified X and Y variables 
    global Xs 
    global ys
    global ridge_predict
    global ridge_results
    global opt_ridge_results
    # Stores scores of the basic model 
    ridge_results = pd.DataFrame(columns=["Feature", "r2", "MAE", "MSE"])

    # Stores the y_preds and y_test values 
    ridge_predict = pd.DataFrame()

    i=1
    for y_variable, X_variable in zip(ys, Xs):
        # Set up model 
        y = sdg_indexes[y_variable]
        X = sat_mod[X_variable]

        model = RidgeModel(y_variable, X, y)
        model.set_model()
        model.get_coef()
        ridge_results = model.evaluate_preds(ridge_results)
        #model.scatter_hist()
        #ridge_predict = model.save_predict(ridge_predict)

        i = i+1

# %%
# %% GRAPHS 
def scatterplots(title1, x1, y1, cvr21,
                 title2, x2, y2, cvr22,
                 title3, x3, y3, cvr23,
                 title4, x4, y4, cvr24,): 
    
    fig, ((ax0, ax1, ax2, ax3)) = plt.subplots(nrows=1, 
                                            ncols=4, 
                                            figsize=(20, 7))

    # Graph 1 
    g_x = ridge_predict[x1]
    g_y = ridge_predict[y1]

    ax0.scatter(x = g_x, y = g_y)
    ax0.set(xlabel=x1, ylabel=y1, title=title1)

    # add trendline
    z = np.polyfit(g_x, g_y, 1)
    p = np.poly1d(z)
    ax0.plot(g_x,p(g_x),"r-")

    # Labels 
    corr = np.corrcoef(g_x, g_y)[0,1]
    ax0.text(0.05, 0.95, f"Corr. coef: {corr:.2f}",
             transform=ax0.transAxes,
             fontsize=12,
             verticalalignment="top")
    ax0.text(0.05, 0.90, f"CV. R2: {cvr21:.2f}%",
             transform=ax0.transAxes,
             fontsize=12,
             verticalalignment="top")
    # ================================
    # Graph 2 
    g_x = ridge_predict[x2]
    g_y = ridge_predict[y2]

    ax1.scatter(x = g_x, y = g_y)
    ax1.set(xlabel=x2, ylabel=y2, title=title2)

    # add trendline
    z = np.polyfit(g_x, g_y, 1)
    p = np.poly1d(z)
    ax1.plot(g_x,p(g_x),"r-")
    
    # Labels 
    corr = np.corrcoef(g_x, g_y)[0,1]
    ax1.text(0.05, 0.95, f"Corr. coef: {corr:.2f}",
             transform=ax1.transAxes,
             fontsize=12,
             verticalalignment="top")
    ax1.text(0.05, 0.90, f"CV. R2: {cvr22:.2f}%",
             transform=ax1.transAxes,
             fontsize=12,
             verticalalignment="top")
    # ================================    
    # Graph 3 
    g_x = ridge_predict[x3]
    g_y = ridge_predict[y3]

    ax2.scatter(x = g_x, y = g_y)
    ax2.set(xlabel=x3, ylabel=y3, title=title3)

    # add trendline
    z = np.polyfit(g_x, g_y, 1)
    p = np.poly1d(z)
    ax2.plot(g_x,p(g_x),"r-")

    corr = np.corrcoef(g_x, g_y)[0,1]
    ax2.text(0.05, 0.95, f"Corr. coef: {corr:.2f}",
             transform=ax2.transAxes,
             fontsize=12,
             verticalalignment="top")
    ax2.text(0.05, 0.90, f"CV. R2: {cvr23:.2f}%",
             transform=ax2.transAxes,
             fontsize=12,
             verticalalignment="top")
    # ================================    
    
    # Graph 4 
    g_x = ridge_predict[x4]
    g_y = ridge_predict[y4]

    ax3.scatter(x = g_x, y = g_y)
    ax3.set(xlabel=x4, ylabel=y4, title=title4)

    # add trendline
    z = np.polyfit(g_x, g_y, 1)
    p = np.poly1d(z)
    ax3.plot(g_x,p(g_x),"r-")

    corr = np.corrcoef(g_x, g_y)[0,1]
    ax3.text(0.05, 0.95, f"Corr. coef: {corr:.2f}",
             transform=ax3.transAxes,
             fontsize=12,
             verticalalignment="top")
    ax3.text(0.05, 0.90, f"CV. R2: {cvr24:.2f}%",
             transform=ax3.transAxes,
             fontsize=12,
             verticalalignment="top")

# %%
def feature_usage_table(model):
    mask = []    
    # Only uses the features that are in the curated feature list without dummies
    for var in feature_code:
        if var in model.X_name:
            mask.append(1)
        else:
            mask.append(0) 
    usage_table.loc[len(usage_table)] = mask

# %% 
def feature_coef_table(model):
    global coef_table

    features_temp = pd.DataFrame(columns=["feature", model.name])

    features = model.X.columns 
    coefs = model.fitted_model.coef_.flatten()

    for coef, feature in zip(coefs, features):
        features_temp.loc[len(features_temp.index)] = [feature, coef]

    if coef_table.empty:
        coef_table = features_temp
    else: 
        coef_table = coef_table.merge(features_temp, on="feature", how="outer")
    
# %%
def optimize():
    global opt_ridge_results

    # Stores scores of the tuned model 
    opt_ridge_results = pd.DataFrame(columns=["Feature", "r2", "MAE", "MSE"]) 

    models = [sdg1_model, sdg2_model, sdg3_model, sdg4_model, sdg5_model, sdg6_model, sdg7_model,
              sdg8_model,sdg9_model,sdg10_model,sdg11_model,sdg13_model,
              sdg17_model,imds_model]
    
    for model in models:
        opt_ridge_results = model.model_optimizer(opt_ridge_results)
        
# %% [markdown]
# # Import satellite and SDG data 

# %%
sdg_indexes = pd.read_csv("data/sdg_prediction/sdg_indexes.csv")
sat_mod = pd.read_csv("data/sdg_prediction/sat_mod.csv")
sdg_indicators = pd.read_csv("data/sdg_prediction/sdg_indicators_norm.csv")
#sat_mod = pd.read_csv("./data/sdg_prediction/sat_true.csv")

# Creates dommies based on the Bolivian departments  
sat_mod = sat_mod.join(pd.get_dummies(sat_mod.dep))
#sat_mod = sat_mod.join(pd.get_dummies(sat_mod.dep))

# %%
#   Basic predictors 
#            'ln_ghsl2015', 'lnagr_land2012', 'lnurb_land2012','ln_land_temp2012', 'ln_tr400_pop2012', 'ln_dist_road2017',
#             'ln_dist_drug2017mean', 'ln_pm25_2012', 'photov2019mean', 'Beni', 'Chuquisaca', 'Cochabamba', 'La Paz',
#              'Oruro', 'Pando', 'Potosí', 'Santa Cruz', 'Tarija', 'ln_precCRU2012min'

# TODO: Add missing X variables 
#sat_mod variables 
X = ['Beni','Chuquisaca', 'Cochabamba', 'La Paz', 'Oruro', 'Pando', 'Potosí', 'Santa Cruz', 'Tarija', 
     'ln_dist_drug2017mean', 'ln_dist_road2017', 'ln_elev2017mean', 'ln_ghsl2015', 'ln_land_temp2012', 
     'ln_pm25_2012','ln_precCRU2012mean', 'ln_t400NTLpc2012', 'ln_tr400_pop2012', 
      'photov2019mean','ln_slope500m2017mean','ln_access2016mean', "ln_density_pop2015count", 
     'land_per_area_2012_full_forest','land_per_area_2012_cropland_natural_vegetation_mosaic',
     'lnurb_land2012', "lnagr_land2012",'lnEGDPpc2012', "airTemp2012.mean"]

#'land_per_area_2012_urban_and_builtup'
# %% 
# %% [markdown]

#'dist_diamond2015'  was not relevant on any model 
# 'land_per_area_2012_water' was not relevant on any model 
# 'ln_mal_inci_rt_mean' Didnt make any difference

# %%
# Stores scores of the basic model 
ridge_results = pd.DataFrame(columns=["Label", "r2", "MAE", "MSE"])

# Stores scores of the tuned model 
opt_ridge_results = pd.DataFrame(columns=["Label", "r2", "MAE", "MSE"])

# Stores the y_preds and y_test values 
ridge_predict = pd.DataFrame()

# %% 
# Instance for the SDG 1 

# Predictors NOT included in the model 
erase_x1 = ['Beni', 'La Paz', 'Oruro', 'Pando', 'ln_elev2017mean', 'ln_land_temp2012', 'ln_precCRU2012min', 
            'ln_dist_drug2017mean', 'ln_slope500m2017mean', 'ln_dist_road2017', 'lnagr_land2012', 
            'land_per_area_2012_full_forest', 'land_per_area_2012_cropland_natural_vegetation_mosaic',
            'dist_coast2017mean']

X_index_1 = [e for e in X if e not in erase_x1]

sdg1_model = RidgeModel(label_description[0], sat_mod[X_index_1], sdg_indexes["index_sdg1"])
sdg1_model.set_model()
sdg1_model.get_coef()
ridge_results = sdg1_model.evaluate_preds(ridge_results)
ridge_predict = sdg1_model.save_predict(ridge_predict)
#sdg1_model.scatter_hist()

# %%
# Instance for the SDG 2 

erase_x2 = ['Beni', 'La Paz', 'ln_dist_drug2017mean','ln_t400NTLpc2012','lnagr_land2012','photov2019mean',
            'ln_land_temp2012', 'Chuquisaca', 'ln_pm25_2012', 'ln_dist_road2017', 'Cochabamba', 
            'land_per_area_2012_full_forest', 'land_per_area_2012_cropland_natural_vegetation_mosaic',
            'lnurb_land2012','ln_tr400_pop2012','ln_elev2017mean','ln_precCRU2012mean', 'airTemp2012.mean',
            'dist_coast2017mean']

X_index_2 = [e for e in X if e not in erase_x2]

sdg2_model = RidgeModel(label_description[1],sat_mod[X_index_2], sdg_indexes["index_sdg2"])
sdg2_model.set_model()
sdg2_model.get_coef()
ridge_results = sdg2_model.evaluate_preds(ridge_results)
ridge_predict = sdg2_model.save_predict(ridge_predict)
#sdg2_model.scatter_hist()
# %% Instance for the SDG 3 

# Predictors NOT included in the model 
erase_x3 = ['Beni','Chuquisaca', 'Cochabamba', 'La Paz', 'Oruro', 'ln_dist_drug2017mean', 'ln_dist_road2017', 
            'ln_elev2017mean', 'ln_land_temp2012', 'ln_precCRU2012min', 'ln_t400NTLpc2012', 'lnagr_land2012', 
            'photov2019mean', "ln_slope500m2017mean", 'ln_access2016mean', 'ln_precCRU2012mean',
            'land_per_area_2012_full_forest', 'airTemp2012.mean','dist_coast2017mean']

X_index_3 = [e for e in X if e not in erase_x3]
#        * Should add NTL later again in increased the score slightly but was irrelevant

sdg3_model = RidgeModel(label_description[2],sat_mod[X_index_3], sdg_indexes["index_sdg3"])
sdg3_model.set_model()
sdg3_model.get_coef()
ridge_results = sdg3_model.evaluate_preds(ridge_results)
ridge_predict = sdg3_model.save_predict(ridge_predict)

# %% Instance for the SDG 4 

erase_x4 = ['Cochabamba', 'ln_dist_road2017', 'ln_elev2017mean', 'ln_land_temp2012', 'ln_pm25_2012',  
            'ln_t400NTLpc2012', "ln_slope500m2017mean", 'ln_access2016mean', 'lnagr_land2012', 'photov2019mean',
            'Beni', 'Chuquisaca', 'dist_coast2017mean']

X_index_4 = [e for e in X if e not in erase_x4]

sdg4_model = RidgeModel(label_description[3],sat_mod[X_index_4], sdg_indexes["index_sdg4"])
sdg4_model.set_model()
sdg4_model.get_coef()
ridge_results = sdg4_model.evaluate_preds(ridge_results)
ridge_predict = sdg4_model.save_predict(ridge_predict)

# %% Instance for the SDG 5 

erase_x5 = ['Chuquisaca', 'Pando', 'Santa Cruz', 'Tarija', 'ln_t400NTLpc2012', 'ln_tr400_pop2012', 
            'lnurb_land2012', "ln_slope500m2017mean", 'ln_access2016mean', 'ln_dist_road2017', 
            'ln_land_temp2012', 'ln_density_pop2015count', 
            'land_per_area_2012_cropland_natural_vegetation_mosaic','lnEGDPpc2012', 'airTemp2012.mean',
            'dist_coast2017mean']

X_index_5 = [e for e in X if e not in erase_x5]
#        * Should add NTL later again in increased the score slightly but was irrelevant

sdg5_model = RidgeModel(label_description[4], sat_mod[X_index_5], sdg_indexes["index_sdg5"])
sdg5_model.set_model()
sdg5_model.get_coef()
ridge_results = sdg5_model.evaluate_preds(ridge_results)
ridge_predict = sdg5_model.save_predict(ridge_predict)
# %% Instance for the SDG 6 

erase_x6 = ['Chuquisaca', 'Cochabamba', 'La Paz', 'Pando', 'ln_dist_drug2017mean', 'ln_elev2017mean', 
            'ln_land_temp2012', 'ln_precCRU2012min', "ln_slope500m2017mean", 'ln_access2016mean', 
            'ln_precCRU2012mean', 'ln_density_pop2015count',
            'land_per_area_2012_cropland_natural_vegetation_mosaic','lnEGDPpc2012', 'airTemp2012.mean',
            'dist_coast2017mean']

X_index_6 = [e for e in X if e not in erase_x6]
#        * Should add 'ln_access2016mean' later again in increased the score slightly but was irrelevant

sdg6_model = RidgeModel(label_description[5], sat_mod[X_index_6], sdg_indexes["index_sdg6"])
sdg6_model.set_model()
sdg6_model.get_coef()
ridge_results = sdg6_model.evaluate_preds(ridge_results)
ridge_predict = sdg6_model.save_predict(ridge_predict)
# %% Instance for the SDG 7 

erase_x7 = ['La Paz', 'Oruro', 'ln_land_temp2012', 'ln_precCRU2012min', 'photov2019mean', 
            "ln_slope500m2017mean", 'ln_dist_drug2017mean', 'land_per_area_2012_full_forest',
            'land_per_area_2012_cropland_natural_vegetation_mosaic', 'Potosí','airTemp2012.mean']
X_index_7 = [e for e in X if e not in erase_x7]

sdg7_model = RidgeModel(label_description[6], sat_mod[X_index_7], sdg_indexes["index_sdg7"])
sdg7_model.set_model()
sdg7_model.get_coef()
ridge_results = sdg7_model.evaluate_preds(ridge_results)
ridge_predict = sdg7_model.save_predict(ridge_predict)
# %% Instance for the SDG 8 

erase_x8 = ['Cochabamba', 'Oruro', 'Potosí', 'ln_dist_road2017', 'ln_elev2017mean','ln_t400NTLpc2012', 
            'ln_tr400_pop2012','photov2019mean','ln_access2016mean','ln_land_temp2012', 'ln_density_pop2015count',
            'land_per_area_2012_full_forest', 'land_per_area_2012_cropland_natural_vegetation_mosaic',
            'lnurb_land2012', 'lnagr_land2012' , 'airTemp2012.mean']
X_index_8 = [e for e in X if e not in erase_x8]

sdg8_model = RidgeModel(label_description[7], sat_mod[X_index_8], sdg_indexes["index_sdg8"])
sdg8_model.set_model()
sdg8_model.get_coef()
ridge_results = sdg8_model.evaluate_preds(ridge_results)
ridge_predict = sdg8_model.save_predict(ridge_predict)
# %% Instance for the SDG 9

erase_x9 = ['Beni', 'Potosí', 'Santa Cruz', 'ln_land_temp2012', 'ln_precCRU2012min', 'lnagr_land2012',
            "ln_slope500m2017mean", 'ln_access2016mean', 'ln_precCRU2012mean', 'lnurb_land2012',
            'land_per_area_2012_full_forest', 'land_per_area_2012_cropland_natural_vegetation_mosaic',
            'lnEGDPpc2012' , 'airTemp2012.mean']
X_index_9 = [e for e in X if e not in erase_x9]

sdg9_model = RidgeModel(label_description[8], sat_mod[X_index_9], sdg_indexes["index_sdg9"])
sdg9_model.set_model()
sdg9_model.get_coef()
ridge_results = sdg9_model.evaluate_preds(ridge_results)
ridge_predict = sdg9_model.save_predict(ridge_predict)
# %% Instance for the SDG 10 

erase_x10 = ['Santa Cruz', 'ln_dist_drug2017mean', 'ln_dist_road2017', 'ln_land_temp2012', 'ln_pm25_2012', 
             'ln_t400NTLpc2012',"ln_slope500m2017mean", 'Tarija', 'lnagr_land2012', 'ln_density_pop2015count',
             ]

X_index_10 = [e for e in X if e not in erase_x10]
#        * Should add NTL later again in increased the score slightly but was irrelevant

sdg10_model = RidgeModel(label_description[9], sat_mod[X_index_10], sdg_indexes["index_sdg10"])
sdg10_model.set_model()
sdg10_model.get_coef()
ridge_results = sdg10_model.evaluate_preds(ridge_results)
ridge_predict = sdg10_model.save_predict(ridge_predict)
# %% Instance for the SDG 11

erase_x11 = ['ln_dist_drug2017mean','Pando','ln_slope500m2017mean', 'ln_access2016mean',
             'land_per_area_2012_full_forest','ln_pm25_2012','lnagr_land2012', 'ln_t400NTLpc2012',
             'ln_precCRU2012mean', 'airTemp2012.mean']        
X_index_11 = [e for e in X if e not in erase_x11]
#        * Should add NTL later again in increased the score slightly but was irrelevant

sdg11_model = RidgeModel(label_description[10], sat_mod[X_index_11], sdg_indexes["index_sdg11"])
sdg11_model.set_model()
sdg11_model.get_coef()
ridge_results = sdg11_model.evaluate_preds(ridge_results)
ridge_predict = sdg11_model.save_predict(ridge_predict)
# %% Instance for the SDG 13

erase_x13 = ['Chuquisaca', 'La Paz', 'Oruro', 'Potosí', 'ln_ghsl2015','ln_land_temp2012','ln_tr400_pop2012', 
             'ln_dist_road2017', 'photov2019mean', "Beni",'land_per_area_2012_full_forest',
             'land_per_area_2012_cropland_natural_vegetation_mosaic', 'ln_t400NTLpc2012',
             'lnagr_land2012']
X_index_13 = [e for e in X if e not in erase_x13]
#        * Should add NTL later again in increased the score slightly but was irrelevant

sdg13_model = RidgeModel(label_description[11], sat_mod[X_index_13], sdg_indexes["index_sdg13"])
sdg13_model.set_model()
sdg13_model.get_coef()
ridge_results = sdg13_model.evaluate_preds(ridge_results)
ridge_predict = sdg13_model.save_predict(ridge_predict)
# %% Instance for the SDG 14

erase_x15 = ['La Paz', 'Oruro', 'Potosí', 'Santa Cruz', 'Tarija','ln_dist_drug2017mean', 'ln_ghsl2015', 'ln_land_temp2012', 
             'ln_precCRU2012min', 'ln_t400NTLpc2012', 'ln_tr400_pop2012','ln_slope500m2017mean']
X_index_15 = [e for e in X if e not in erase_x15]

sdg15_model = RidgeModel(label_description[12], sat_mod[X_index_15], sdg_indexes["index_sdg15"])
sdg15_model.set_model()
sdg15_model.get_coef()
ridge_results = sdg15_model.evaluate_preds(ridge_results)
ridge_predict = sdg15_model.save_predict(ridge_predict)

# %% Instance for the SDG 16

erase_x16 = ['Oruro', 'Pando', 'Potosí', 'Santa Cruz', 'ln_dist_drug2017mean', 'ln_elev2017mean', 'ln_ghsl2015', 
             'ln_land_temp2012','ln_pm25_2012', 'ln_precCRU2012min', 'ln_t400NTLpc2012', 'ln_tr400_pop2012', 
             'photov2019mean', "ln_slope500m2017mean", 'ln_access2016mean', 'ln_precCRU2012mean', 'lnurb_land2012']
X_index_16 = [e for e in X if e not in erase_x16]

sdg16_model = RidgeModel(label_description[13], sat_mod[X_index_16], sdg_indexes["index_sdg16"])
sdg16_model.set_model()
sdg16_model.get_coef()
ridge_results = sdg16_model.evaluate_preds(ridge_results)
ridge_predict = sdg16_model.save_predict(ridge_predict)
# %% Instance for the SDG 17

erase_x17 = ['Chuquisaca', 'Cochabamba', 'Potosí', 'Santa Cruz','ln_dist_drug2017mean', 'ln_dist_road2017', 'ln_ghsl2015', 
             'ln_land_temp2012', "ln_slope500m2017mean", 'lnurb_land2012']
X_index_17 = [e for e in X if e not in erase_x17]

sdg17_model = RidgeModel(label_description[14], sat_mod[X_index_17], sdg_indexes["index_sdg17"])
sdg17_model.set_model()
sdg17_model.get_coef()
ridge_results = sdg17_model.evaluate_preds(ridge_results)
ridge_predict = sdg17_model.save_predict(ridge_predict)

# %% Instance for the SDG imds

erase_imds =['ln_land_temp2012','Santa Cruz','Oruro', 'Beni', 'ln_dist_drug2017mean','ln_elev2017mean',
             'land_per_area_2012_cropland_natural_vegetation_mosaic','ln_slope500m2017mean',
             'ln_dist_road2017','ln_access2016mean','ln_pm25_2012', 'airTemp2012.mean']

X_imds = [e for e in X if e not in erase_imds]

imds_model = RidgeModel(label_description[15], sat_mod[X_imds], sdg_indexes["imds"])
imds_model.set_model()
imds_model.get_coef()
ridge_results = imds_model.evaluate_preds(ridge_results)
ridge_predict = imds_model.save_predict(ridge_predict) 
# %% Definition for the iterative instancing 
Xs = [X_index_1, X_index_2, X_index_3, X_index_4, X_index_5, X_index_6, X_index_7, X_index_8, X_index_9, 
        X_index_10, X_index_11, X_index_13, X_index_15, X_index_16, X_index_17, X_imds]

ys = sdg_indexes.drop(columns= {"id", "mun_id"})
# %% [markdown]
# # Graph best fitters 
# ## Subtitle 
# ### a smaller title 
# Normal text  **bold text**    


# %%
scatterplots(sdg1_model.name, sdg1_model.y_true_label, sdg1_model.y_preds_label, sdg1_model.cvr2,
             imds_model.name, imds_model.y_true_label, imds_model.y_preds_label, imds_model.cvr2,
             sdg7_model.name, sdg7_model.y_true_label, sdg7_model.y_preds_label, sdg7_model.cvr2,
             sdg9_model.name, sdg9_model.y_true_label, sdg9_model.y_preds_label, sdg9_model.cvr2,
             )

scatterplots(sdg13_model.name, sdg13_model.y_true_label, sdg13_model.y_preds_label, sdg13_model.cvr2,
             sdg11_model.name, sdg11_model.y_true_label, sdg11_model.y_preds_label, sdg11_model.cvr2,
             sdg10_model.name, sdg10_model.y_true_label, sdg10_model.y_preds_label, sdg10_model.cvr2,
             sdg6_model.name, sdg6_model.y_true_label, sdg6_model.y_preds_label, sdg6_model.cvr2,)
# %% [markdown]
# # Adjust Hyperparameters 

# %%
#mer = pd.merge(sat_mod[X_index_1 + ["id"]], sdg_indexes[["id", "index_sdg1"]], on="id", how="outer")

# %% [Markdown]

# ###This table indicated wich features are being used on each model

# %%
dep_dummies = ['Beni', 'Chuquisaca','Cochabamba', 'La Paz', 'Oruro', 'Pando', 'Potosí', 'Santa Cruz','Tarija',] 
all_X = [e for e in X if e not in dep_dummies]

usage_table = pd.DataFrame()
coef_table = pd.DataFrame()

for var in feature_code:
    # Assigns the columns names to the df  
    usage_table[var] = []

models_sdg = [sdg1_model, sdg2_model, sdg3_model, sdg4_model, sdg5_model, sdg6_model, sdg7_model,
              sdg8_model,sdg9_model,sdg10_model,sdg11_model,sdg13_model,sdg15_model,sdg16_model,
              sdg17_model,imds_model]

# Fills the table using 1 and 0 
for model in models_sdg:
    # Fill the df with 1 and 0 depending if the model uses the feature or not
    feature_usage_table(model)
    feature_coef_table(model)

# ================= Usage Table ==============================
# Rename the table columns 
for code, name in zip(feature_code,feature_name):
    usage_table = usage_table.rename(columns={code:name})

# Change the name of the labels and uses the description instead SDG1 -> No Poverty 
usage_table["SDGs"] = label_description
usage_table.set_index("SDGs", inplace=True)

usage_table.to_csv("./data/sdg_prediction/used_x_models.csv")

# =============================================================


# ================== Features Coef table ======================
# Make the features the columns of the df 
coef_table = coef_table.set_index("feature")
coef_table = coef_table.transpose()

coef_table = coef_table.drop(columns=dep_dummies) # Drop all the fix effects coefficients 
coef_table = coef_table.fillna(0) # Features that are not using the model will be assigned a 0 coefficient

# Rename the table columns 
for code, name in zip(feature_code,feature_name):
    coef_table = coef_table.rename(columns={code:name})

# Organizes the table acording to the order in the feature_name list 
ordered_columns = [col for col in feature_name if col in coef_table.columns]
coef_table = coef_table[ordered_columns]   

# Change the name of the SDG by its descriptions 
coef_table["SDG"] = label_description
coef_table.set_index("SDG", inplace=True)

coef_table = coef_table.round(2)
coef_table.to_csv("./data/sdg_prediction/coef_table.csv")

# =============================================================
# %% [Markdown]
# ## Latex tables 
# %%
print(ridge_results.to_latex(index=False,
                       float_format= "{:.2f}".format))

# %% [Markdown]

# ## R2 boxplot

# %%
# Models list - the models 15 and 16 are not here since they are negatives 
r2_models = [sdg1_model, sdg2_model, sdg3_model, sdg4_model, sdg5_model, sdg6_model, sdg7_model,
              sdg8_model,sdg9_model,sdg10_model,sdg11_model,sdg13_model,
              sdg17_model,imds_model]

dic = {}
for model in r2_models: 
    # The dictionary will store the 5 folds data using the model name as a key
    dic[model.name] = model.r2_folds

dic_data = []
for model in r2_models:
    df = pd.DataFrame({
        "model": [model.name] * len(model.r2_folds),
        "r2_value": model.r2_folds

    })
    dic_data.append(df)

plot_data=pd.concat(dic_data, ignore_index=True)

grouped = plot_data.loc[:,["model", "r2_value"]] \
            .groupby(["model"]) \
            .median() \
            .sort_values(by="r2_value", ascending=False)

plt.figure(figsize=(14, 8))
ax = sns.boxplot(data=plot_data, y="model", x="r2_value", 
                 width=0.7,
                 boxprops={"facecolor":"tab:blue",  "alpha":0.5},
                 order=grouped.index, 
                 showmeans=True,
                 meanprops = {'marker':'|','markeredgecolor':'tab:red','markersize':'8'},
                 legend="full",
                 )
ax.set_title("Satellite Data Shows a Consistent High Predictive Power for SDG 1 and SDI", 
             fontsize=16,
             pad=13);

ax.set_xlabel("R2 Values", fontsize=14, fontdict={"weight": "bold"})
ax.set_ylabel("SDG Models", fontsize=14, fontdict={"weight": "bold"})

ax.axvline(70, color="black", dashes=(2,2));
ax.text(0.83, 0.3, "R2 = 70",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        color="darkred");

# fig, ax = plt.subplots(figsize=(10,7))
# ax.boxplot(dic.values(), vert=0)
# ax.set_yticklabels(dic.keys())
# ax.set_title("R2 of each SDG in 5 folds")
# ax.set(xlabel="R2")
# plt.vlines([70], ymin=0, ymax=15 , colors="r", linestyles="--")
# #ax.set_xlim(0, 100) 
# ax.text(0.83, 0.3, "R2 = 70",
#         transform=ax.transAxes,
#         fontsize=10,
#         verticalalignment="top")

# %%

# %%
# create sample DataFrame
# create plot
# color = ["red", "white", "blue"]
# n_bins = 1000
# cmap_name = "custom_cmap"

# cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, color, N=n_bins)
plt.figure(figsize=(14, 10))
ax = sns.heatmap(data=coef_table, 
                 cbar=False,
                 cmap="vlag_r", 
                 mask=(coef_table==0),
                 annot=True,                 
                 annot_kws={"fontsize":12})
# ax.tick_params(axis='y', labelrotation=45, labelsize=12)
ax.set_title("Coefficients of the Predictors Used on Each Model", pad=15, fontsize=21, fontdict={"weight": "bold"})
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=12)

ax.set_xlabel("Satellite Predictors", fontsize=18, fontdict={"weight":"bold"})
ax.set_ylabel("SDG Models", fontsize=18, fontdict={"weight":"bold"})

# show plot
plt.show()
# %%
