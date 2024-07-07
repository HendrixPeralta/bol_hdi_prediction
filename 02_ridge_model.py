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
from sklearn.linear_model import Ridge

from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from mlxtend.plotting import scatterplotmatrix 

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
def model_coef(fitted_model, X,y):
    
    print("*"*20, y_variable, "*"*20, "\n")
    print(f"X variables:\n {np.array(X.columns)}\n")
    print(f"Non CV score: {(fitted_model.score(X_test, y_test)*100).round(2)}\n")

    coeff = fitted_model.coef_
    
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
    
    print(fitted_model.coef_)
    print("="*80)
    print("\n\n")


# %%

# Optimizes and save the models 
def model_optimizer(model):
   
    alpha_space = np.logspace(-4,0,30)
    alpha_space 

    grid = {"alpha": alpha_space,
            "copy_X": [True, False],
            "max_iter": [None, 10, 100, 200, 500, 1000, 10000], 
            "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg"]}

    np.random.seed(42)
    opt_ri_model= RandomizedSearchCV(estimator = model,
                                    param_distributions=grid,
                                    n_iter=100,
                                    cv=5,
                                    verbose=0)

    return opt_ri_model


    #rs_y_preds = opt_ri_model.predict(X_test)
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
#X = sat_mod[[ "ln_t400NTLpc2012", "ln_tr400_pop2012", 'lnEGDPpc2012', 'ln_perUrb_land2012',  "ln_pm25_2012", "ln_land_temp2012"]]

#X = sat_mod[['ln_access2016mean' 'ln_elev2017mean']]

# %% [markdown]

#'dist_diamond2015'  was not relevant on any model 

# %% [markdown]
## Test for all labels 

# %%
#   Basic predictors 
#            'ln_ghsl2015', 'lnagr_land2012', 'lnurb_land2012','ln_land_temp2012', 'ln_tr400_pop2012', 'ln_dist_road2017',
#             'ln_dist_drug2017mean', 'ln_pm25_2012', 'photov2019mean', 'Beni', 'Chuquisaca', 'Cochabamba', 'La Paz',
#              'Oruro', 'Pando', 'Potosí', 'Santa Cruz', 'Tarija', 'ln_precCRU2012min'

# Defining Predictors for each SDG

#  "+" in the Eliminated secction means that i failed to record which variables were eliminated beforehand 

#SDG Indicators Y
#y_variables = sdg_indicators.drop(columns= {"id", "mun_id"})


# Based Model 
#SDG indexed Y
y_variables = sdg_indexes.drop(columns= {"id", "mun_id"})

X = ['Beni','Chuquisaca', 'Cochabamba', 'La Paz', 'Oruro', 'Pando', 'Potosí', 'Santa Cruz', 'Tarija','ln_dist_drug2017mean', 
       'ln_dist_road2017', 'ln_elev2017mean', 'ln_ghsl2015', 'ln_land_temp2012', 'ln_pm25_2012', 'ln_precCRU2012min',
       'ln_t400NTLpc2012', 'ln_tr400_pop2012', 'lnagr_land2012', 'lnurb_land2012', 'photov2019mean', "ln_slope500m2017mean", 
       'ln_access2016mean']

#'land_per_area_2012_urban_and_builtup',
erase_x1 = ['Beni', 'La Paz', 'Oruro', 'Pando', 'ln_elev2017mean', 'ln_land_temp2012', 'ln_precCRU2012min'
        ]
X_index_1 = [e for e in X if e not in erase_x1]

erase_x2 = ['Beni', 'La Paz', 'ln_dist_drug2017mean','ln_t400NTLpc2012','lnagr_land2012','photov2019mean',
            'ln_land_temp2012', 'Chuquisaca', 'ln_pm25_2012', 'ln_dist_road2017']
X_index_2 = [e for e in X if e not in erase_x2]

erase_x3 = ['Beni','Chuquisaca', 'Cochabamba', 'La Paz', 'Oruro', 'ln_dist_drug2017mean', 'ln_dist_road2017', 
            'ln_elev2017mean', 'ln_land_temp2012', 'ln_precCRU2012min', 'ln_t400NTLpc2012', 'lnagr_land2012', 
            'photov2019mean', "ln_slope500m2017mean", 'ln_access2016mean']
X_index_3 = [e for e in X if e not in erase_x3]
#        * Should add NTL later again in increased the score slightly but was irrelevant

erase_x4 = ['Cochabamba', 'ln_dist_road2017', 'ln_elev2017mean', 'ln_land_temp2012', 'ln_pm25_2012', 'ln_t400NTLpc2012',
            "ln_slope500m2017mean", 'ln_access2016mean']
X_index_4 = [e for e in X if e not in erase_x4]

erase_x5 = ['Chuquisaca', 'Pando', 'Santa Cruz', 'Tarija', 'ln_t400NTLpc2012', 'ln_tr400_pop2012', 'lnurb_land2012',
             "ln_slope500m2017mean", 'ln_access2016mean']
X_index_5 = [e for e in X if e not in erase_x5]
#        * Should add NTL later again in increased the score slightly but was irrelevant

erase_x6 = ['Chuquisaca', 'Cochabamba', 'La Paz', 'Pando', 'ln_dist_drug2017mean', 'ln_elev2017mean', 'ln_land_temp2012', 
            'ln_precCRU2012min', "ln_slope500m2017mean", 'ln_access2016mean']
X_index_6 = [e for e in X if e not in erase_x6]
#        * Should add 'ln_access2016mean' later again in increased the score slightly but was irrelevant

erase_x7 = ['La Paz', 'Oruro', 'ln_land_temp2012', 'ln_precCRU2012min', 'photov2019mean', 
            "ln_slope500m2017mean"]
X_index_7 = [e for e in X if e not in erase_x7]

erase_x8 = ['Cochabamba', 'Oruro', 'Potosí', 'ln_dist_road2017', 'ln_elev2017mean','ln_t400NTLpc2012', 'ln_tr400_pop2012', 
            'photov2019mean','ln_access2016mean']
X_index_8 = [e for e in X if e not in erase_x8]

erase_x9 = ['Beni', 'Potosí', 'Santa Cruz', 'ln_land_temp2012', 'ln_precCRU2012min', 'lnagr_land2012',
            "ln_slope500m2017mean", 'ln_access2016mean']
X_index_9 = [e for e in X if e not in erase_x9]

erase_x10 = ['Santa Cruz', 'ln_dist_drug2017mean', 'ln_dist_road2017', 'ln_land_temp2012', 'ln_pm25_2012', 
             'ln_t400NTLpc2012',"ln_slope500m2017mean"]
X_index_10 = [e for e in X if e not in erase_x10]
#        * Should add NTL later again in increased the score slightly but was irrelevant

erase_x11 = ['Beni', 'Cochabamba', 'Pando','Santa Cruz', 'ln_dist_drug2017mean', 'photov2019mean',
             'ln_slope500m2017mean', 'ln_elev2017mean', 'ln_access2016mean']
X_index_11 = [e for e in X if e not in erase_x11]
#        * Should add NTL later again in increased the score slightly but was irrelevant

erase_x13 = ['Chuquisaca', 'La Paz', 'Oruro', 'Potosí', 'ln_ghsl2015','ln_land_temp2012', 'ln_t400NTLpc2012', 
             'ln_tr400_pop2012', 'ln_dist_road2017', 'photov2019mean']
X_index_13 = [e for e in X if e not in erase_x13]
#        * Should add NTL later again in increased the score slightly but was irrelevant

erase_x15 = ['La Paz', 'Oruro', 'Potosí', 'Santa Cruz', 'Tarija','ln_dist_drug2017mean', 'ln_ghsl2015', 'ln_land_temp2012', 
             'ln_precCRU2012min', 'ln_t400NTLpc2012', 'ln_tr400_pop2012',]
X_index_15 = [e for e in X if e not in erase_x15]

erase_x16 = ['Oruro', 'Pando', 'Potosí', 'Santa Cruz', 'ln_dist_drug2017mean', 'ln_elev2017mean', 'ln_ghsl2015', 
             'ln_land_temp2012','ln_pm25_2012', 'ln_precCRU2012min', 'ln_t400NTLpc2012', 'ln_tr400_pop2012', 
             'photov2019mean', "ln_slope500m2017mean"]
X_index_16 = [e for e in X if e not in erase_x16]

erase_x17 = ['Chuquisaca', 'Cochabamba', 'Potosí', 'Santa Cruz','ln_dist_drug2017mean', 'ln_dist_road2017', 'ln_ghsl2015', 
             'ln_land_temp2012', "ln_slope500m2017mean"]
X_index_17 = [e for e in X if e not in erase_x17]

erase_imds = ['Beni', 'La Paz', 'Oruro', 'Potosí', 'Santa Cruz', 'ln_elev2017mean', 'ln_land_temp2012', 'ln_precCRU2012min',
               'lnagr_land2012', "ln_slope500m2017mean"]
X_imds = [e for e in X if e not in erase_imds]

Xs = [X_index_1, X_index_2, X_index_3, X_index_4, X_index_5, X_index_6, X_index_7, X_index_8, X_index_9, 
      X_index_10, X_index_11, X_index_13, X_index_15, X_index_16, X_index_17, X_imds]


# %%
# Training the Ridge model 

#x = sat_mod[[ "ln_t400NTLpc2012", "ln_tr400_pop2012", 'lnEGDPpc2012', 'ln_perUrb_land2012',  
#               "ln_pm25_2012", "ln_land_temp2012",'ln_dist_road2017' ,'ln_ghsl2015']]
# This ones make some sdg more relevant 

ridge_results = pd.DataFrame(columns=["Feature", "r2", "MAE", "MSE"])
opt_ridge_results = pd.DataFrame(columns=["Feature", "r2", "MAE", "MSE"])
# Stores the y_preds and y_test values 
ridge_predict = pd.DataFrame()

for y_variable, X in zip(y_variables, Xs):
    
    # Set up model 
    y = y_variables[y_variable]
    X = sat_mod[X]

    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3) 
    ridge_model_it = linear_model.Ridge()
    
    # ==================
    # Shows the coefficients for each predictor
    fitted_model = ridge_model_it.fit(X_train, y_train);
    model_coef(fitted_model,X,y)
    # ==================

    # Store the cross evaluation resilts into a df 
    scores = evaluate_preds(ridge_model_it, X, y)
    ridge_results.loc[len(ridge_results.index)] = [y_variable, scores[0], scores[1], scores[2]]  
    results = ridge_results.round(4).sort_values(by="r2", ascending=False)

    # =================
    # Optimizer 
    #opt_ri_model = model_optimizer(ridge_model_it)
    #opt_ri_model.fit(X_train, y_train);
    #print("Best parameters for: ", y_variable)
    #print(opt_ri_model.best_params_)
    #print("="*80)
    #print("\n\n")
    
    #opt_scores = evaluate_preds(opt_ri_model, X, y)
    #opt_ridge_results.loc[len(opt_ridge_results.index)] = [y_variable, opt_scores[0], opt_scores[1], 
    #                                                       opt_scores[2]]

    #opt_results = opt_ridge_results.round(4).sort_values(by="r2", ascending=False)
    #==================

    # Predicts and stores the prediction and real values to make graphs 
    y_pred = ridge_model_it.predict(X_test)

    col0 = y_variable + "_true"
    col1 = y_variable + "_pred"
    temp_predict = pd.DataFrame({col0: y_test, col1: y_pred}, index=y_test.index)
    temp_predict.index.name = "id"
    
    if ridge_predict.empty:
        ridge_predict = temp_predict
    else:
        ridge_predict = ridge_predict.merge(temp_predict, on="id", how="outer")


    

# %% [markdown]
# ## Graph best fitters 
    
# %% GRAPHS 
fig, ((ax0, ax1, ax2, ax3)) = plt.subplots(nrows=1, 
                                         ncols=4, 
                                         figsize=(20, 7))

# Graph 1 
g_x = ridge_predict["index_sdg1_true"]
g_y = ridge_predict["index_sdg1_pred"]

ax0.scatter(x = g_x, y = g_y)
ax0.set(xlabel="index_sdg1_true", ylabel="index_sdg1_pred", title="SDG1")

# add trendline
z = np.polyfit(g_x, g_y, 1)
p = np.poly1d(z)
ax0.plot(g_x,p(g_x),"r-")


# Graph 1 
g_x = ridge_predict["imds_true"]
g_y = ridge_predict["imds_pred"]

ax1.scatter(x = g_x, y = g_y)
ax1.set(xlabel="imds_true", ylabel="imds_abs_pred", title="imds")

# add trendline
z = np.polyfit(g_x, g_y, 1)
p = np.poly1d(z)
ax1.plot(g_x,p(g_x),"r-")


# Graph 1 
g_x = ridge_predict["index_sdg11_true"]
g_y = ridge_predict["index_sdg11_pred"]

ax2.scatter(x = g_x, y = g_y)
ax2.set(xlabel="index_sdg11_true", ylabel="index_sdg11_pred", title="index_sdg11")

# add trendline
z = np.polyfit(g_x, g_y, 1)
p = np.poly1d(z)
ax2.plot(g_x,p(g_x),"r-")


# Graph 1 
g_x = ridge_predict["index_sdg9_true"]
g_y = ridge_predict["index_sdg9_pred"]

ax3.scatter(x = g_x, y = g_y)
ax3.set(xlabel="index_sdg9_true", ylabel="index_sdg9_pred", title="index_sdg9")

# add trendline
z = np.polyfit(g_x, g_y, 1)
p = np.poly1d(z)
ax3.plot(g_x,p(g_x),"r-")

# %% [markdown]
# # Adjust Hyperparameters 

# %%
mer = pd.merge(sat_mod[X_index_1 + ["id"]], sdg_indexes[["id", "index_sdg1"]], on="id", how="outer")

# %%

cols = list(mer.columns)

cols.remove("id")
#cols.remove("dep")
#cols.remove("beni")
cols.remove('Chuquisaca')
cols.remove('Cochabamba')
#cols.remove("La Paz")
#cols.remove("Oruro")
#cols.remove("Pando")
cols.remove("Potosí")
cols.remove("Santa Cruz")
cols.remove("Tarija")
scatterplotmatrix(mer[cols].values, figsize=(50,50), names=cols, alpha=0.5)
plt.tight_layout()
plt.show()

# %%
