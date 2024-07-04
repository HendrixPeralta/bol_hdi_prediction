# %%
# %%
import numpy as np
import pandas as pd
import sklearn as ktl
import pickle
import matplotlib.pylab as plt
import seaborn

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# %%

# Returns the R2, MAE and MSE for each model which is later stored in a dataframe

def evaluate_preds(model, X, y):
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels
    on a classification.
    """
    r2 = (np.mean(cross_val_score(model, X, y, scoring="r2")))*100
    mae = np.mean(cross_val_score(model, X, y, scoring="neg_mean_absolute_error"))
    mse = np.mean(cross_val_score(model, X, y, scoring="neg_mean_squared_error"))
    metric_dict = [r2, mae, mse]
    
    return metric_dict

# %%

# Shows the coeffients for each of the X variables 
def model_coef(X,y):
    
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = linear_model.Ridge()
    model.fit(X_train, y_train);

    print("*"*20, y_variable, "*"*20, "\n")
    print(f"X variables:\n {np.array(X.columns)}\n")
    print(f"Non CV score: {(model.score(X_test, y_test)*100).round(2)}\n")

    coeff = model.coef_
    
    relevance = 1 

    # Calculates relevant coefficients 
    pos_rel = np.array(X.columns)[coeff>relevance]
    neg_rel = np.array(X.columns)[coeff<-relevance]

    # Calcuates NON relevant coefficients

    pos_non = np.array(X.columns)[(coeff<relevance) & (coeff>0)]
    neg_non = np.array(X.columns)[(coeff>-relevance) & (coeff<0)]
#filtered_columns_str = ", ".join(filtered_columns)
    
    print("Relevant and positive:")
    print(pos_rel)
    print("\nRelevant and negative:")
    print(neg_rel)
    print("\n")
    
    print("NON relevant and positive:")
    print(pos_non)
    print("\n NON relevant and negative:")
    print(neg_non)
    print("\n")
    
    print(model.coef_)
    print("="*80)
    print("\n\n")

# %% [markdown]
# # Import satellite and SDG data 

# %%
sdg_indexes = pd.read_csv("data/sdg_prediction/sdg_indexes.csv")
sat_mod = pd.read_csv("data/sdg_prediction/sat_mod.csv")
sdg_indicators = pd.read_csv("data/sdg_prediction/sdg_indicators_norm.csv")

# Creates dommies based on the Bolivian departments  
sat_mod = sat_mod.join(pd.get_dummies(sat_mod.dep))

# %% [markdown]
# # Ridge model 

# %%
X_obsolete = sat_mod[['ln_ghsl2015', 'lnagr_land2012', 'lnurb_land2012','ln_land_temp2012', 'ln_tr400_pop2012', 'ln_dist_road2017',
                        'ln_dist_drug2017mean', 'ln_pm25_2012', 'photov2019mean', 'Beni', 'Chuquisaca', 'Cochabamba', 'La Paz',
                        'Oruro', 'Pando', 'Potosí', 'Santa Cruz', 'Tarija', 'ln_precCRU2012min']]

#X = sat_mod[[ "ln_t400NTLpc2012", "ln_tr400_pop2012", 'lnEGDPpc2012', 'ln_perUrb_land2012',  "ln_pm25_2012", "ln_land_temp2012"]]

#X = sat_mod[['ln_t400NTLpc2012', "ln_perUrb_land2012", 'ln_land_temp2012','ln_tr400_pop2012','ln_dist_road2017','ln_ghsl2015', 
#             "ln_dist_water2017mean",'ln_dist_drug2017mean', 'ln_elev2017mean', 'ln_pm25_2012', 'photov2019mean', 
#             'lnagr_land2012', 'lnurb_land2012','ln_access2016mean']]
  #'dist_diamond2015',   'lnagr_land2012', 'lnurb_land2012',


y = sdg_indexes["imds"]

# %% [markdown]

#'dist_diamond2015'  was not relevant on any model 

# %% [markdown]
## Test for all labels 

# %%
#   Basic predictors 
# 'ln_ghsl2015', 'lnagr_land2012', 'lnurb_land2012','ln_land_temp2012', 'ln_tr400_pop2012', 'ln_dist_road2017',
#                        'ln_dist_drug2017mean', 'ln_pm25_2012', 'photov2019mean', 'Beni', 'Chuquisaca', 'Cochabamba', 'La Paz',
#                        'Oruro', 'Pando', 'Potosí', 'Santa Cruz', 'Tarija', 'ln_precCRU2012min'

# Defining Predictors for each SDG

#  "+" in the Eliminated secction means that i failed to record which variables were eliminated beforehand 

X_index_1 = ['ln_ghsl2015', 'lnagr_land2012', 'lnurb_land2012', 'ln_pm25_2012', 'Potosí','Santa Cruz', 'Tarija', 
             'ln_tr400_pop2012', 'ln_dist_road2017', 'ln_dist_drug2017mean','photov2019mean', 
             'Chuquisaca', 'Cochabamba', "ln_t400NTLpc2012"]
#       Eliminated  'ln_land_temp2012' 'Beni' 'Oruro' 'ln_precCRU2012min' 'La Paz' +

X_index_2 = ['ln_ghsl2015', 'lnurb_land2012','ln_land_temp2012', 'ln_tr400_pop2012', 'ln_dist_road2017', 
             'ln_pm25_2012',  'Chuquisaca', 'Cochabamba', 'Oruro', 'Pando', 'Potosí', 'Santa Cruz', 'Tarija', 'ln_precCRU2012min', 'ln_elev2017mean']
#       Eliminated 'Beni' 'lnagr_land2012' 'La Paz' 'dist_diamond2015' 'photov2019mean' "ln_t400NTLpc2012" 
#                   'ln_dist_drug2017mean'

X_index_3 = ['ln_ghsl2015', 'lnurb_land2012', 'ln_tr400_pop2012', 'ln_pm25_2012', 'Pando', 'Potosí', 'Santa Cruz', 'Tarija']
#       Eliminated 'lnagr_land2012' 'ln_land_temp2012' 'ln_dist_road2017' 'ln_dist_drug2017mean' 'Chuquisaca' 'Cochabamba' 
#                   'La Paz' 'photov2019mean' 'Beni' 'ln_precCRU2012min' 'Oruro' "ln_t400NTLpc2012" 
#        * Should add NTL later again in increased the score slightly but was irrelevant

X_index_4 = ['ln_ghsl2015', 'lnagr_land2012', 'lnurb_land2012', 'ln_tr400_pop2012','ln_dist_drug2017mean', 
             'photov2019mean', 'Beni', 'Chuquisaca', 'La Paz', 'Oruro', 'Pando', 'Potosí', 'Santa Cruz', 
             'Tarija', 'ln_precCRU2012min']
#       Eliminated  'ln_land_temp2012' 'ln_pm25_2012' 'Cochabamba' 'ln_dist_road2017' "ln_t400NTLpc2012" 

X_index_5 = ['lnagr_land2012', 'ln_land_temp2012', 'ln_dist_road2017', 'ln_dist_drug2017mean', 'ln_pm25_2012', 
             'photov2019mean', 'Beni', 'Cochabamba', 'La Paz', 'Oruro', 'Potosí', 'ln_precCRU2012min']
#       Eliminated 'ln_ghsl2015' 'ln_tr400_pop2012' 'Chuquisaca' 'Tarija' 'lnurb_land2012' 'Santa Cruz' 'Pando'
#        * Should add NTL later again in increased the score slightly but was irrelevant

X_index_6 = ['ln_ghsl2015', 'lnagr_land2012', 'lnurb_land2012', 'ln_tr400_pop2012', 'ln_dist_road2017',
             'ln_pm25_2012', 'photov2019mean', 'Beni', 'Oruro', 'Potosí', 'Santa Cruz', 'Tarija',"ln_t400NTLpc2012"]
#       Eliminated 'ln_dist_drug2017mean' 'ln_precCRU2012min' 'ln_land_temp2012' 'Pando' 'Chuquisaca' 'Cochabamba' 'La Paz'

X_index_7 = ['ln_ghsl2015', 'lnagr_land2012', 'lnurb_land2012', 'ln_pm25_2012', 'Beni', 'Potosí', 'Santa Cruz', 'Tarija',
              'ln_tr400_pop2012', 'ln_dist_road2017','ln_dist_drug2017mean', 'Chuquisaca', 'Cochabamba', 
             'Pando',"ln_t400NTLpc2012"]
#       Eliminated  'photov2019mean' 'ln_precCRU2012min' 'La Paz' 'Oruro' 'ln_land_temp2012' +

X_index_8 = ['ln_ghsl2015', 'lnagr_land2012', 'lnurb_land2012','ln_land_temp2012','ln_dist_drug2017mean', 
             'ln_pm25_2012', 'Beni', 'Chuquisaca', 'La Paz', 'Pando', 'Santa Cruz', 'Tarija', 'ln_precCRU2012min']
#       Eliminated 'ln_tr400_pop2012' 'ln_dist_road2017' 'Cochabamba' 'photov2019mean' 'Oruro' 'Potosí'

X_index_9 = ['ln_ghsl2015', 'lnurb_land2012', 'photov2019mean', 'La Paz','Oruro', 'Tarija', 'ln_precCRU2012min',
             'ln_tr400_pop2012', 'ln_dist_road2017', 'ln_dist_drug2017mean','ln_pm25_2012', 'Beni', 'Chuquisaca', 'Cochabamba', 
             'Pando',"ln_t400NTLpc2012"]
#       Eliminated 'ln_land_temp2012' 'Potosí' 'Santa Cruz' +

X_index_10 = ['ln_ghsl2015', 'lnagr_land2012', 'lnurb_land2012','ln_land_temp2012', 'ln_tr400_pop2012',
              'photov2019mean', 'Beni', 'Chuquisaca', 'Cochabamba', 'La Paz','Oruro', 'Pando', 'Potosí', 'Tarija', 
              'ln_precCRU2012min',"ln_t400NTLpc2012"]
#       Eliminated 'ln_dist_road2017' 'ln_dist_drug2017mean' 'Santa Cruz' 'ln_pm25_2012'

X_index_11 = ['ln_ghsl2015', 'lnagr_land2012', 'lnurb_land2012', 'La Paz', 'Potosí','Tarija', 'ln_precCRU2012min',
              'ln_land_temp2012', 'ln_tr400_pop2012', 'ln_dist_road2017', 'ln_pm25_2012','Chuquisaca', 
               'Oruro', 'Santa Cruz',"ln_t400NTLpc2012"]
#       Eliminated 'photov2019mean' 'ln_dist_drug2017mean' 'Pando' +

X_index_13 = [ 'lnagr_land2012', 'lnurb_land2012', 'ln_dist_road2017', 'ln_dist_drug2017mean', 'ln_pm25_2012', 'photov2019mean', 
              'Beni', 'Cochabamba','Pando', 'Santa Cruz', 'Tarija', 'ln_precCRU2012min', 'ln_elev2017mean',"ln_t400NTLpc2012"]
#      Eliminated 'Oruro' 'ln_tr400_pop2012' 'La Paz' 'Potosí' 'Chuquisaca' 'ln_ghsl2015' 'ln_land_temp2012'

X_index_15 = ['ln_ghsl2015', 'lnagr_land2012', 'lnurb_land2012','ln_land_temp2012', 'ln_dist_road2017', 'ln_pm25_2012', 
              'photov2019mean', 'Beni', 'Chuquisaca', 'Cochabamba', 'La Paz', 'Pando', 'Potosí', 'Santa Cruz', 
              'Tarija', 'ln_precCRU2012min',"ln_t400NTLpc2012"]
#       Eliminated 'Oruro' 'ln_tr400_pop2012' 'ln_dist_drug2017mean'

X_index_16 = ['lnagr_land2012', 'lnurb_land2012', 'ln_dist_road2017','Beni', 'Chuquisaca', 'Cochabamba', 'La Paz', 'Tarija',
              "ln_t400NTLpc2012"]
#       Eliminated 'ln_ghsl2015' 'ln_tr400_pop2012' 'Oruro' 'Potosí' 'ln_land_temp2012' 'ln_dist_drug2017mean' 'ln_pm25_2012' 
# 'photov2019mean' 'Pando' 'Santa Cruz' 'ln_precCRU2012min'

X_index_17 = ['ln_ghsl2015', 'lnagr_land2012', 'lnurb_land2012', 'ln_tr400_pop2012', 'ln_pm25_2012', 
              'photov2019mean', 'Beni', 'Chuquisaca', 'Cochabamba', 'La Paz', 'Oruro', 'Pando', 'Potosí', 'Tarija', 
              'ln_precCRU2012min',"ln_t400NTLpc2012"]
#       Eliminated 'Santa Cruz''ln_dist_road2017' 'ln_dist_drug2017mean' 'ln_land_temp2012'

X_imds = ['ln_ghsl2015', 'lnagr_land2012', 'lnurb_land2012', 'Oruro', 'Santa Cruz','Tarija', 'ln_tr400_pop2012', 
          'ln_dist_road2017', 'ln_dist_drug2017mean','ln_pm25_2012', 'photov2019mean', 'Chuquisaca', 'Cochabamba', 'Pando',
          'Potosí',"ln_t400NTLpc2012"]
#       Eliminated 'La Paz' 'ln_land_temp2012' 'Beni' 'ln_precCRU2012min' + 

Xs = [X_index_1, X_index_2, X_index_3, X_index_4, X_index_5, X_index_6, X_index_7, X_index_8, X_index_9, 
      X_index_10, X_index_11, X_index_13, X_index_15, X_index_16, X_index_17, X_imds]

# %%


y_variables = sdg_indexes[['index_sdg1', 'index_sdg2', 'index_sdg3', 'index_sdg4',
       'index_sdg5', 'index_sdg6', 'index_sdg7', 'index_sdg8', 'index_sdg9',
       'index_sdg10', 'index_sdg11', 'index_sdg13', 'index_sdg15',
       'index_sdg16', 'index_sdg17', 'imds']]

#y_variables = sdg_indicators.drop(columns = {"id"})
# %%
# Training the Ridge model 

#x = sat_mod[[ "ln_t400NTLpc2012", "ln_tr400_pop2012", 'lnEGDPpc2012', 'ln_perUrb_land2012',  
#               "ln_pm25_2012", "ln_land_temp2012",'ln_dist_road2017' ,'ln_ghsl2015']]
# This ones make some sdg more relevant 

ridge_results = pd.DataFrame(columns=["Feature", "r2", "MAE", "MSE"])

for y_variable, X in zip(y_variables, Xs):
    
    y = y_variables[y_variable]

    # ==================
    X = sat_mod[X]

    model_coef(X,y)
    # ==================
    np.random.seed(42)
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3) 
    ridge_model_it = linear_model.Ridge()

    scores = evaluate_preds(ridge_model_it, X, y)

    ridge_results.loc[len(ridge_results.index)] = [y_variable, scores[0], scores[1], scores[2]]

    # Store the cross valuation resilts into a df 
    results = ridge_results.round(4).sort_values(by="r2", ascending=False)


# %% [markdown]
# ## Graph best fitters 

# %%

elaN_predict = pd.DataFrame()

# Prediction df 

for y_variable in y_variables:

    ridge_model_it.fit(x_train,y_train)
    y_pred = ridge_model_it.predict(x_test)

    col0 = y_variable + "_true"
    col1 = y_variable + "_pred"
    temp_predict = pd.DataFrame({col0: y_test, col1: y_pred}, index=y_test.index)
    temp_predict.index.name = "id"
    
    if elaN_predict.empty:
        elaN_predict = temp_predict
    else:
        elaN_predict = elaN_predict.merge(temp_predict, on="id", how="outer")

# %%
fig, ((ax0, ax1, ax2, ax3)) = plt.subplots(nrows=1, 
                                         ncols=4, 
                                         figsize=(20, 7))

# Graph 1 
g_x = elaN_predict["index_sdg1_true"]
g_y = elaN_predict["index_sdg1_pred"]

ax0.scatter(x = g_x, y = g_y)
ax0.set(xlabel="sdg1_1_pubn_abs_true", ylabel="sdg1_1_pubn_abs_pred", title="SDG1")

# add trendline
z = np.polyfit(g_x, g_y, 1)
p = np.poly1d(z)
ax0.plot(g_x,p(g_x),"r-")


# Graph 1 
g_x = elaN_predict["index_sdg7_true"]
g_y = elaN_predict["index_sdg7_pred"]

ax1.scatter(x = g_x, y = g_y)
ax1.set(xlabel="sdg9_c_hf_abs_true", ylabel="sdg9_c_hf_abs_pred", title="index_sdg7")

# add trendline
z = np.polyfit(g_x, g_y, 1)
p = np.poly1d(z)
ax1.plot(g_x,p(g_x),"r-")


# Graph 1 
g_x = elaN_predict["index_sdg11_true"]
g_y = elaN_predict["index_sdg11_pred"]

ax2.scatter(x = g_x, y = g_y)
ax2.set(xlabel="sdg1_1_dtl_abs_true", ylabel="sdg1_1_dtl_abs_pred", title="index_sdg11")

# add trendline
z = np.polyfit(g_x, g_y, 1)
p = np.poly1d(z)
ax2.plot(g_x,p(g_x),"r-")


# Graph 1 
g_x = elaN_predict["index_sdg9_true"]
g_y = elaN_predict["index_sdg9_pred"]

ax3.scatter(x = g_x, y = g_y)
ax3.set(xlabel="sdg3_2_fb_abs_true", ylabel="sdg3_2_fb_abs_pred", title="index_sdg9")

# add trendline
z = np.polyfit(g_x, g_y, 1)
p = np.poly1d(z)
ax3.plot(g_x,p(g_x),"r-")
# %% [markdown]
# # Adjust Hyperparameters 
# %%
model_tuned = linear_model.Ridge(alpha=0.0001, max_iter=1000)
model_tuned.fit(X_train, y_train);
model.score(X_test, y_test)

# %%
