import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from mlxtend.plotting import scatterplotmatrix


class RidgeModel: 

    def __init__(self, name, X=pd.DataFrame, y=pd.DataFrame, test_size=0.3, model = None):
        self.name = name
        self.X = X
        self.y = y
        self.y_preds_label = None
        self.y_true_label = None

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
        self.opt_r2_folds = None

    # Set up model 
    def set_model(self):
        np.random.seed(42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size = self.test_size) 
        self.model = Ridge()
        self.fitted_model = self.model.fit(self.X_train, self.y_train);
        print(self.fitted_model.get_params())
        print("model fitted")
    # ==================     
    
    def get_params(self):
        return self.model.get_params()
    
    
    # Shows the coefficients for each predictor
    def get_coef(self):

        print("*"*20, self.name, "*"*20, "\n")
        print(f"X variables:\n {np.array(self.X.columns)}\n")
        print(f"Non CV score: {round(self.fitted_model.score(self.X_test, self.y_test)*100,2)}\n")

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
        self.r2_folds = cross_val_score(self.model, self.X, self.y, scoring="r2", cv=5)*100
        r2 = np.mean(self.r2_folds)
        mae = np.mean(cross_val_score(self.model, self.X, self.y, scoring="neg_mean_absolute_error", cv=5))
        mse = np.mean(cross_val_score(self.model, self.X, self.y, scoring="neg_mean_squared_error", cv=5))
        scores = [r2, mae, mse]
        
        score_results.loc[len(score_results.index)] = [self.name, scores[0], scores[1], scores[2]]  
        score_results = score_results.round(4).sort_values(by="r2", ascending=False)
        
        # Saves the CV R2 into the object
        self.cvr2 = r2

        return score_results

    # Predicts and stores the prediction and real values to make graphs 
    def save_predict(self, store_predict):   
        y_pred = self.model.predict(self.X_test)

        self.y_true_label = self.name.split(" ",2)[2] + " true"
        self.y_preds_label = self.name.split(" ",2)[2] + " pred"
        temp_predict = pd.DataFrame({self.y_true_label: self.y_test, self.y_preds_label: y_pred}, 
                                    index=self.y_test.index)
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

    def model_optimizer(self, opt_ridge_results):
        
        alpha_space = np.logspace(-4,0,100)
        alpha_space 

        grid = {"alpha": alpha_space,
                # "copy_X": [True, False],
                # "max_iter": [None, 10, 100, 200, 500, 1000, 10000], 
                # "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg"]
                }

        np.random.seed(42)
        opt_ri_model= GridSearchCV(estimator = self.model,
                                        # param_distributions=grid,
                                        param_grid=grid,
                                        # n_iter=200,
                                        cv=5,
                                        verbose=1)

        
        opt_ri_model.fit(self.X_train, self.y_train)
        self.opt_r2_folds = cross_val_score(opt_ri_model, self.X, self.y, scoring="r2")*100
        r2 = np.mean(self.opt_r2_folds)
        mae = np.mean(cross_val_score(opt_ri_model, self.X, self.y, scoring="neg_mean_absolute_error"))
        mse = np.mean(cross_val_score(opt_ri_model, self.X, self.y, scoring="neg_mean_squared_error"))
        
        opt_ridge_results.loc[len(opt_ridge_results.index)] = [self.name, r2, mae, mse]
        opt_ridge_results = opt_ridge_results.round(4).sort_values(by="r2", ascending=False)

        # return opt_ri_model.best_params_
        return opt_ridge_results
