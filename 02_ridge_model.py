# %%
# %%
import numpy as np
import pandas as pd
import sklearn as ktl
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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
import scipy.stats as stats


# %% 

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
            "Vegetation"]

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
        "land_per_area_2012_full_forest ",
        "land_per_area_2012_urban_and_builtup",
        "land_per_area_2012_full_savannas_grasslands",
        "land_per_area_2012_full_shrublands",
        "land_per_area_2012_cropland_natural_vegetation_mosaic"]

label_description = [
        "No poverty",
        "Zero hunger",
        "Good health and well-being",
        "Quality education",
        "Gender equality",
        "Clean water and sanitization",
        "Affordable and clean energy",
        "Decent work and economic growth",
        "Industry, innovation and infraestructure",
        "Reduced inequalities",
        "Sustainable cities and communities",
        "Climate action",
        "Life on land",
        "Peace, justice and strong institutions",
        "Parnerships for the goals",
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

    # Stores scores of the tuned model 
    opt_ridge_results = pd.DataFrame(columns=["Feature", "r2", "MAE", "MSE"])

    # Stores the y_preds and y_test values 
    ridge_predict = pd.DataFrame()
    # Makes sure that the df are empty 
    #ridge_results.drop(ridge_results.index, inplace=True)
    #ridge_results.drop(ridge_results.columns, axis=0, inplace=True)

    #ridge_predict.drop(ridge_predict.index, inplace=True)
    #ridge_predict.drop(ridge_predict.columns, axis=0, inplace=True)
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
        ridge_predict = model.predict(ridge_predict)

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
def fill_usage_table(model):
    mask = []    
    #used_X = [e for e in model.X_name if e not in dep_dummies]
    for var in feature_code:
        if var in model.X_name:
            mask.append(1)
        else:
            mask.append(0) 
    usage_table.loc[len(usage_table)] = mask
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
# # Ridge model 
class RidgeModel: 

    def __init__(self, name, X=pd.DataFrame, y=pd.DataFrame, test_size=0.3, model = None):
        self.name = name
        self.X = X
        self.y = y
        self.model = model
        self.fitted_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_size = test_size
        self.full_df = None
        self.cvr2 = None
        self.X_name = X.columns
        X.index.name = "id"
        y.index.name = "id"
        self.full_df = X.merge(y, on="id", how="outer")
        self.r2_folds = None
    # Set up model 
    def set_model(self):
        np.random.seed(42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size = self.test_size) 
        self.model = linear_model.Ridge()
        self.fitted_model = self.model.fit(self.X_train, self.y_train);
        print("model fitted")

    # ==================    
    # Shows the coefficients for each predictor
    def get_coef(self):

        print("*"*20, self.name, "*"*20, "\n")
        print(f"X variables:\n {np.array(self.X.columns)}\n")
        print(f"Non CV score: {(self.fitted_model.score(self.X_test, self.y_test)*100).round(2)}\n")

        coeff = self.fitted_model.coef_.flatten()      
        relevance = 1 

        # Calculates relevant coefficients 
        pos_rel = np.array(self.X.columns)[coeff>relevance]
        neg_rel = np.array(self.X.columns)[coeff<-relevance]

        # Calcuates NON relevant coefficients
        pos_non = np.array(self.X.columns)[(coeff<relevance) & (coeff>0)]
        neg_non = np.array(self.X.columns)[(coeff>-relevance) & (coeff<0)]
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
        
        print(np.round(self.fitted_model.coef_.flatten(),4))
        print("="*80)
        print("\n\n")
        
        #model_coef(self.fitted_model,X,y)
    # ==================
   
    # Store the cross evaluation resilts into a df 
    def evaluate_preds(self, score_results):
        """
        Performs evaluation comparison on y_true labels vs. y_pred labels
        on a classification.
        """
        self.r2_folds = cross_val_score(self.model, self.X, self.y, scoring="r2", cv=10)*100
        r2 = np.mean(self.r2_folds)
        mae = np.mean(cross_val_score(self.model, self.X, self.y, scoring="neg_mean_absolute_error", cv=10))
        mse = np.mean(cross_val_score(self.model, self.X, self.y, scoring="neg_mean_squared_error", cv=10))
        scores = [r2, mae, mse]
        
        score_results.loc[len(score_results.index)] = [self.name, scores[0], scores[1], scores[2]]  
        score_results = score_results.round(4).sort_values(by="r2", ascending=False)
        
        # Saves the CV R2 into the object
        self.cvr2 = r2

        return score_results

    # Predicts and stores the prediction and real values to make graphs 
    def predict(self, store_predict):   
        y_pred = self.model.predict(self.X_test)

        col0 = self.name + "_true"
        col1 = self.name + "_pred"
        temp_predict = pd.DataFrame({col0: self.y_test, col1: y_pred}, index=self.y_test.index)
        temp_predict.index.name = "id"
        if store_predict.empty:
            store_predict = temp_predict
        else:
            store_predict = store_predict.merge(temp_predict, on="id", how="outer")
        #print("Added Prediction results to the ridge_predict df")
        return store_predict
    
    def scatter_hist(self):
        cols = list(self.full_df.columns)
        non_continuous_vars = ['Beni', 'Chuquisaca','Cochabamba', 'La Paz', 'Oruro', 'Pando', 'Potosí', 
                               'Santa Cruz','Tarija', "id", "dep"]

        col_eval = [col for col in cols if col not in non_continuous_vars]

        scatterplotmatrix(self.full_df[col_eval].values, figsize=(50,50), names=col_eval, alpha=0.5)
        plt.tight_layout()
        plt.show()

    def model_optimizer(self):
    
        alpha_space = np.logspace(-4,0,100)
        alpha_space 

        grid = {"alpha": alpha_space,
                "copy_X": [True, False],
                "max_iter": [None, 10, 100, 200, 500, 1000, 10000], 
                "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg"]}

        np.random.seed(42)
        opt_ri_model= RandomizedSearchCV(estimator = self.model,
                                        param_distributions=grid,
                                        n_iter=200,
                                        cv=10,
                                        verbose=0)

        
        opt_ri_model.fit(self.X_train, self.y_train)
        r2 = np.mean(cross_val_score(opt_ri_model, self.X, self.y, scoring="r2")*100)
        print(r2)
    # =================
    # Optimizer 
    #opt_ri_model = model_optimizer(self.model)
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

# %%
#   Basic predictors 
#            'ln_ghsl2015', 'lnagr_land2012', 'lnurb_land2012','ln_land_temp2012', 'ln_tr400_pop2012', 'ln_dist_road2017',
#             'ln_dist_drug2017mean', 'ln_pm25_2012', 'photov2019mean', 'Beni', 'Chuquisaca', 'Cochabamba', 'La Paz',
#              'Oruro', 'Pando', 'Potosí', 'Santa Cruz', 'Tarija', 'ln_precCRU2012min'


#sat_mod variables 
X = ['Beni','Chuquisaca', 'Cochabamba', 'La Paz', 'Oruro', 'Pando', 'Potosí', 'Santa Cruz', 'Tarija', 
     'ln_dist_drug2017mean', 'ln_dist_road2017', 'ln_elev2017mean', 'ln_ghsl2015', 'ln_land_temp2012', 
     'ln_pm25_2012','ln_precCRU2012mean', 'ln_t400NTLpc2012', 'ln_tr400_pop2012', 
      'photov2019mean','ln_slope500m2017mean','ln_access2016mean', "ln_density_pop2015count", 
     'land_per_area_2012_full_forest','land_per_area_2012_cropland_natural_vegetation_mosaic',
     'lnurb_land2012', "lnagr_land2012",'lnEGDPpc2012']
     
  # 'lnurb_land2012',    "lnagr_land2012", 
 #    'perUrb_land2012', 
 #    'dist_diamond2015', 'mal_inci_rt_mean', 'dist_water2017mean', 'ln_elev2017mean',  
        
#'land_per_area_2012_urban_and_builtup'
# %% 
# %% [markdown]

#'dist_diamond2015'  was not relevant on any model 
# 'land_per_area_2012_water' was not relevant on any model 
# 'ln_mal_inci_rt_mean' Didnt make any difference

# %%
# Stores scores of the basic model 
ridge_results = pd.DataFrame(columns=["Feature", "r2", "MAE", "MSE"])

# Stores scores of the tuned model 
opt_ridge_results = pd.DataFrame(columns=["Feature", "r2", "MAE", "MSE"])

# Stores the y_preds and y_test values 
ridge_predict = pd.DataFrame()

# %% 
# Instance for the SDG 1 

# Predictors NOT included in the model 
erase_x1 = ['Beni', 'La Paz', 'Oruro', 'Pando', 'ln_elev2017mean', 'ln_land_temp2012', 'ln_precCRU2012min', 
            'ln_dist_drug2017mean', 'ln_slope500m2017mean', 'ln_dist_road2017', 'lnagr_land2012', 
            'land_per_area_2012_full_forest', 'land_per_area_2012_cropland_natural_vegetation_mosaic']

X_index_1 = [e for e in X if e not in erase_x1]

sdg1_model = RidgeModel("Index SDG 1", sat_mod[X_index_1], sdg_indexes["index_sdg1"])
sdg1_model.set_model()
sdg1_model.get_coef()
ridge_results = sdg1_model.evaluate_preds(ridge_results)
ridge_predict = sdg1_model.predict(ridge_predict)
#sdg1_model.scatter_hist()

# %%
# Instance for the SDG 2 

#erase_x2 = ['Beni', 'La Paz', 'ln_dist_drug2017mean','ln_t400NTLpc2012','lnagr_land2012','photov2019mean',
#            'ln_land_temp2012', 'Chuquisaca', 'ln_pm25_2012', 'ln_dist_road2017', 'Cochabamba', 
#            'land_per_area_2012_full_forest', 'land_per_area_2012_cropland_natural_vegetation_mosaic',
#            'lnurb_land2012']

erase_x2 = ['Beni', 'La Paz', 'ln_dist_drug2017mean','ln_t400NTLpc2012','lnagr_land2012','photov2019mean',
            'ln_land_temp2012', 'Chuquisaca', 'ln_pm25_2012', 'ln_dist_road2017', 'Cochabamba', 
            'land_per_area_2012_full_forest', 'land_per_area_2012_cropland_natural_vegetation_mosaic',
            'lnurb_land2012','ln_tr400_pop2012','ln_elev2017mean','ln_precCRU2012mean']
X_index_2 = [e for e in X if e not in erase_x2]

sdg2_model = RidgeModel("Index SDG 2",sat_mod[X_index_2], sdg_indexes["index_sdg2"])
sdg2_model.set_model()
sdg2_model.get_coef()
ridge_results = sdg2_model.evaluate_preds(ridge_results)
ridge_predict = sdg2_model.predict(ridge_predict)
#sdg2_model.scatter_hist()
# %% Instance for the SDG 3 

# Predictors NOT included in the model 
erase_x3 = ['Beni','Chuquisaca', 'Cochabamba', 'La Paz', 'Oruro', 'ln_dist_drug2017mean', 'ln_dist_road2017', 
            'ln_elev2017mean', 'ln_land_temp2012', 'ln_precCRU2012min', 'ln_t400NTLpc2012', 'lnagr_land2012', 
            'photov2019mean', "ln_slope500m2017mean", 'ln_access2016mean', 'ln_precCRU2012mean',
            'land_per_area_2012_full_forest']
X_index_3 = [e for e in X if e not in erase_x3]
#        * Should add NTL later again in increased the score slightly but was irrelevant

sdg3_model = RidgeModel("Index SDG 3",sat_mod[X_index_3], sdg_indexes["index_sdg3"])
sdg3_model.set_model()
sdg3_model.get_coef()
ridge_results = sdg3_model.evaluate_preds(ridge_results)
ridge_predict = sdg3_model.predict(ridge_predict)

# %% Instance for the SDG 4 

erase_x4 = ['Cochabamba', 'ln_dist_road2017', 'ln_elev2017mean', 'ln_land_temp2012', 'ln_pm25_2012',  
            'ln_t400NTLpc2012', "ln_slope500m2017mean", 'ln_access2016mean', 'lnagr_land2012', 'photov2019mean',
            'Beni', 'Chuquisaca']
X_index_4 = [e for e in X if e not in erase_x4]

sdg4_model = RidgeModel("Index SDG 4",sat_mod[X_index_4], sdg_indexes["index_sdg4"])
sdg4_model.set_model()
sdg4_model.get_coef()
ridge_results = sdg4_model.evaluate_preds(ridge_results)
ridge_predict = sdg4_model.predict(ridge_predict)

# %% Instance for the SDG 5 

erase_x5 = ['Chuquisaca', 'Pando', 'Santa Cruz', 'Tarija', 'ln_t400NTLpc2012', 'ln_tr400_pop2012', 'lnurb_land2012',
             "ln_slope500m2017mean", 'ln_access2016mean', 'ln_dist_road2017', 'ln_land_temp2012', 
             'ln_density_pop2015count', 'land_per_area_2012_cropland_natural_vegetation_mosaic','lnEGDPpc2012']
X_index_5 = [e for e in X if e not in erase_x5]
#        * Should add NTL later again in increased the score slightly but was irrelevant

sdg5_model = RidgeModel("Index SDG 5", sat_mod[X_index_5], sdg_indexes["index_sdg5"])
sdg5_model.set_model()
sdg5_model.get_coef()
ridge_results = sdg5_model.evaluate_preds(ridge_results)
ridge_predict = sdg5_model.predict(ridge_predict)
# %% Instance for the SDG 6 

erase_x6 = ['Chuquisaca', 'Cochabamba', 'La Paz', 'Pando', 'ln_dist_drug2017mean', 'ln_elev2017mean', 
            'ln_land_temp2012', 'ln_precCRU2012min', "ln_slope500m2017mean", 'ln_access2016mean', 
            'ln_precCRU2012mean', 'ln_density_pop2015count',
            'land_per_area_2012_cropland_natural_vegetation_mosaic','lnEGDPpc2012']

X_index_6 = [e for e in X if e not in erase_x6]
#        * Should add 'ln_access2016mean' later again in increased the score slightly but was irrelevant

sdg6_model = RidgeModel("Index SDG 6", sat_mod[X_index_6], sdg_indexes["index_sdg6"])
sdg6_model.set_model()
sdg6_model.get_coef()
ridge_results = sdg6_model.evaluate_preds(ridge_results)
ridge_predict = sdg6_model.predict(ridge_predict)
# %% Instance for the SDG 7 

erase_x7 = ['La Paz', 'Oruro', 'ln_land_temp2012', 'ln_precCRU2012min', 'photov2019mean', 
            "ln_slope500m2017mean", 'ln_dist_drug2017mean', 'land_per_area_2012_full_forest',
            'land_per_area_2012_cropland_natural_vegetation_mosaic', 'Potosí']
X_index_7 = [e for e in X if e not in erase_x7]

sdg7_model = RidgeModel("Index SDG 7", sat_mod[X_index_7], sdg_indexes["index_sdg7"])
sdg7_model.set_model()
sdg7_model.get_coef()
ridge_results = sdg7_model.evaluate_preds(ridge_results)
ridge_predict = sdg7_model.predict(ridge_predict)
# %% Instance for the SDG 8 

erase_x8 = ['Cochabamba', 'Oruro', 'Potosí', 'ln_dist_road2017', 'ln_elev2017mean','ln_t400NTLpc2012', 
            'ln_tr400_pop2012','photov2019mean','ln_access2016mean','ln_land_temp2012', 'ln_density_pop2015count',
            'land_per_area_2012_full_forest', 'land_per_area_2012_cropland_natural_vegetation_mosaic',
            'lnurb_land2012', 'lnagr_land2012']
X_index_8 = [e for e in X if e not in erase_x8]

sdg8_model = RidgeModel("Index SDG 8", sat_mod[X_index_8], sdg_indexes["index_sdg8"])
sdg8_model.set_model()
sdg8_model.get_coef()
ridge_results = sdg8_model.evaluate_preds(ridge_results)
ridge_predict = sdg8_model.predict(ridge_predict)
# %% Instance for the SDG 9

erase_x9 = ['Beni', 'Potosí', 'Santa Cruz', 'ln_land_temp2012', 'ln_precCRU2012min', 'lnagr_land2012',
            "ln_slope500m2017mean", 'ln_access2016mean', 'ln_precCRU2012mean', 'lnurb_land2012',
            'land_per_area_2012_full_forest', 'land_per_area_2012_cropland_natural_vegetation_mosaic',
            'lnEGDPpc2012']
X_index_9 = [e for e in X if e not in erase_x9]

sdg9_model = RidgeModel("Index SDG 9", sat_mod[X_index_9], sdg_indexes["index_sdg9"])
sdg9_model.set_model()
sdg9_model.get_coef()
ridge_results = sdg9_model.evaluate_preds(ridge_results)
ridge_predict = sdg9_model.predict(ridge_predict)
# %% Instance for the SDG 10 

erase_x10 = ['Santa Cruz', 'ln_dist_drug2017mean', 'ln_dist_road2017', 'ln_land_temp2012', 'ln_pm25_2012', 
             'ln_t400NTLpc2012',"ln_slope500m2017mean", 'Tarija', 'lnagr_land2012', 'ln_density_pop2015count',
             ]

X_index_10 = [e for e in X if e not in erase_x10]
#        * Should add NTL later again in increased the score slightly but was irrelevant

sdg10_model = RidgeModel("Index SDG 10", sat_mod[X_index_10], sdg_indexes["index_sdg10"])
sdg10_model.set_model()
sdg10_model.get_coef()
ridge_results = sdg10_model.evaluate_preds(ridge_results)
ridge_predict = sdg10_model.predict(ridge_predict)
# %% Instance for the SDG 11

#erase_x11 = ['Beni', 'Cochabamba', 'Pando','Santa Cruz', 'ln_dist_drug2017mean', 'photov2019mean',
#             'ln_slope500m2017mean', 'ln_elev2017mean', 'ln_access2016mean','ln_t400NTLpc2012', 'ln_pm25_2012',
#             'land_per_area_2012_full_forest'            ]

erase_x11 = ['ln_dist_drug2017mean','Pando','ln_slope500m2017mean', 'ln_access2016mean',
             'land_per_area_2012_full_forest','ln_pm25_2012','lnagr_land2012', 'ln_t400NTLpc2012',
             'ln_precCRU2012mean']        
X_index_11 = [e for e in X if e not in erase_x11]
#        * Should add NTL later again in increased the score slightly but was irrelevant

sdg11_model = RidgeModel("Index SDG 11", sat_mod[X_index_11], sdg_indexes["index_sdg11"])
sdg11_model.set_model()
sdg11_model.get_coef()
ridge_results = sdg11_model.evaluate_preds(ridge_results)
ridge_predict = sdg11_model.predict(ridge_predict)
# %% Instance for the SDG 13

erase_x13 = ['Chuquisaca', 'La Paz', 'Oruro', 'Potosí', 'ln_ghsl2015','ln_land_temp2012','ln_tr400_pop2012', 
             'ln_dist_road2017', 'photov2019mean', "Beni",'land_per_area_2012_full_forest',
             'land_per_area_2012_cropland_natural_vegetation_mosaic', 'ln_t400NTLpc2012',
             'lnagr_land2012']
X_index_13 = [e for e in X if e not in erase_x13]
#        * Should add NTL later again in increased the score slightly but was irrelevant

sdg13_model = RidgeModel("Index SDG 13", sat_mod[X_index_13], sdg_indexes["index_sdg13"])
sdg13_model.set_model()
sdg13_model.get_coef()
ridge_results = sdg13_model.evaluate_preds(ridge_results)
ridge_predict = sdg13_model.predict(ridge_predict)
# %% Instance for the SDG 14

erase_x15 = ['La Paz', 'Oruro', 'Potosí', 'Santa Cruz', 'Tarija','ln_dist_drug2017mean', 'ln_ghsl2015', 'ln_land_temp2012', 
             'ln_precCRU2012min', 'ln_t400NTLpc2012', 'ln_tr400_pop2012','ln_slope500m2017mean']
X_index_15 = [e for e in X if e not in erase_x15]

sdg15_model = RidgeModel("Index SDG 15", sat_mod[X_index_15], sdg_indexes["index_sdg15"])
sdg15_model.set_model()
sdg15_model.get_coef()
ridge_results = sdg15_model.evaluate_preds(ridge_results)
ridge_predict = sdg15_model.predict(ridge_predict)

# %% Instance for the SDG 16

erase_x16 = ['Oruro', 'Pando', 'Potosí', 'Santa Cruz', 'ln_dist_drug2017mean', 'ln_elev2017mean', 'ln_ghsl2015', 
             'ln_land_temp2012','ln_pm25_2012', 'ln_precCRU2012min', 'ln_t400NTLpc2012', 'ln_tr400_pop2012', 
             'photov2019mean', "ln_slope500m2017mean", 'ln_access2016mean', 'ln_precCRU2012mean', 'lnurb_land2012']
X_index_16 = [e for e in X if e not in erase_x16]

sdg16_model = RidgeModel("Index SDG 16", sat_mod[X_index_16], sdg_indexes["index_sdg16"])
sdg16_model.set_model()
sdg16_model.get_coef()
ridge_results = sdg16_model.evaluate_preds(ridge_results)
ridge_predict = sdg16_model.predict(ridge_predict)
# %% Instance for the SDG 17

erase_x17 = ['Chuquisaca', 'Cochabamba', 'Potosí', 'Santa Cruz','ln_dist_drug2017mean', 'ln_dist_road2017', 'ln_ghsl2015', 
             'ln_land_temp2012', "ln_slope500m2017mean"]
X_index_17 = [e for e in X if e not in erase_x17]

sdg17_model = RidgeModel("Index SDG 17", sat_mod[X_index_17], sdg_indexes["index_sdg17"])
sdg17_model.set_model()
sdg17_model.get_coef()
ridge_results = sdg17_model.evaluate_preds(ridge_results)
ridge_predict = sdg17_model.predict(ridge_predict)

# %% Instance for the SDG imds

#erase_imds = ['Beni', 'La Paz', 'Oruro', 'Potosí', 'Santa Cruz', 'ln_elev2017mean', 'ln_land_temp2012', 
#              'ln_precCRU2012min', 'lnagr_land2012', "ln_slope500m2017mean", 'ln_access2016mean', 
#              'land_per_area_2012_full_forest', 'land_per_area_2012_cropland_natural_vegetation_mosaic',
#              'ln_dist_drug2017mean','ln_pm25_2012', 'land_per_area_2012_full_forest']
erase_imds =['ln_land_temp2012','Santa Cruz','Oruro', 'Beni', 'ln_dist_drug2017mean','ln_elev2017mean',
             'land_per_area_2012_cropland_natural_vegetation_mosaic','ln_slope500m2017mean',
             'ln_dist_road2017','ln_access2016mean','ln_pm25_2012']
X_imds = [e for e in X if e not in erase_imds]

imds_model = RidgeModel("SDI", sat_mod[X_imds], sdg_indexes["imds"])
imds_model.set_model()
imds_model.get_coef()
ridge_results = imds_model.evaluate_preds(ridge_results)
ridge_predict = imds_model.predict(ridge_predict) 
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
scatterplots("SDG1", "Index SDG 1_true", 'Index SDG 1_pred', sdg1_model.cvr2,
             "SDI", 'SDI_true', 'SDI_pred', imds_model.cvr2,
             "SDG7", 'Index SDG 7_true', 'Index SDG 7_pred', sdg7_model.cvr2,
             "SDG9", "Index SDG 9_true", 'Index SDG 9_pred', sdg9_model.cvr2,
             )

scatterplots("SDG13", 'Index SDG 13_true', 'Index SDG 13_pred', sdg13_model.cvr2,
             "SDG11", 'Index SDG 11_true', 'Index SDG 11_pred', sdg11_model.cvr2,
             "SDG10", "Index SDG 10_true", 'Index SDG 10_pred', sdg10_model.cvr2,
             "SDG2", 'Index SDG 6_true', 'Index SDG 6_pred', sdg6_model.cvr2,)
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
for var in feature_code:
    # Assigns the columns names to the df  
    usage_table[var] = []

models_sdg = [sdg1_model, sdg2_model, sdg3_model, sdg4_model, sdg5_model, sdg6_model, sdg7_model,
              sdg8_model,sdg9_model,sdg10_model,sdg11_model,sdg13_model,sdg15_model,sdg16_model,
              sdg17_model,imds_model]

# Fills the table using 1 and 0 
for model in models_sdg:
    # Fill the df with 1 and 0 depending if the model uses the feature or not
    fill_usage_table(model)

# Rename the table columns 
for code, name in zip(feature_code,feature_name):
    usage_table = usage_table.rename(columns={code:name})

# Change the name of the labels and uses the description instead SDG! -> No Poverty 
usage_table["SDGs"] = label_description
usage_table.set_index("SDGs", inplace=True)

usage_table.to_csv("./data/sdg_prediction/used_x_models.csv")
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

fig, ax = plt.subplots()
ax.boxplot(dic.values(), vert=0, )
ax.set_yticklabels(dic.keys())
ax.set_title("R2 of each SDG in 5 folds")
ax.set(xlabel="R2")
plt.vlines([70], ymin=0, ymax=15 , colors="r", linestyles="--")
#ax.set_xlim(0, 100) 
ax.text(0.83, 0.3, "R2 = 70",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top")

# %%
