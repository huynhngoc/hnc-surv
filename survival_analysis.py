#!/usr/bin/env python
# coding: utf-8

# In[95]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from sksurv.base import SurvivalAnalysisMixin as s
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sksurv.preprocessing import encode_categorical
from sksurv.datasets import load_gbsg2
from sksurv.functions import StepFunction
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import (ComponentwiseGradientBoostingSurvivalAnalysis, 
                            RandomSurvivalForest, 
                            ExtraSurvivalTrees, 
                            GradientBoostingSurvivalAnalysis, 
                            ExtraSurvivalTrees)
from sksurv.meta import EnsembleSelection, EnsembleSelectionRegressor
from sksurv.metrics import integrated_brier_score
from matplotlib.colors import ListedColormap
from mlxtend.evaluate import paired_ttest_5x2cv
from mlxtend.evaluate import combined_ftest_5x2cv
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test, pairwise_logrank_test
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, KFold
from lifelines.plotting import add_at_risk_counts
import scipy.stats
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from sklearn.model_selection import cross_val_score
from sksurv.metrics import integrated_brier_score
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# In[96]:


# OUS 
OUS_D1 = pd.read_csv('OUS_D1.csv')
OUS_D2 = pd.read_csv('OUS_D2.csv')
OUS_D3 = pd.read_csv('OUS_D3.csv')
OUS_DFS_target = pd.read_csv('OUS_DFS_target.csv')
OUS_OS_target = pd.read_csv('OUS_OS_target.csv')
response_OUS = pd.read_csv('response_ous.csv', sep=';')

# MAASTRO
MAASTRO_D1 = pd.read_csv('MAASTRO_D1.csv')
MAASTRO_D2 = pd.read_csv('MAASTRO_D2.csv')
MAASTRO_D3 = pd.read_csv('MAASTRO_D3.csv')
MAASTRO_DFS_target = pd.read_csv('MAASTRO_DFS_target.csv')
MAASTRO_OS_target = pd.read_csv('MAASTRO_OS_target.csv')
response_MAASTRO = pd.read_csv('response_maastro.csv', sep=',')


# In[97]:


# OUS data shape 
print('OUS_D1', OUS_D1.shape)
print('OUS_D2', OUS_D2.shape)
print('OUS_D3', OUS_D3.shape)
print('OUS_DFS_target', OUS_DFS_target.shape)
print('OUS_OS_target', OUS_OS_target.shape)
print('response_OUS', response_OUS.shape)

# MAASTRO data shape 
print('MAASTRO_D1', MAASTRO_D1.shape)
print('MAASTRO_D2', MAASTRO_D2.shape)
print('MAASTRO_D3', MAASTRO_D3.shape)
print('MAASTRO_DFS_target', MAASTRO_DFS_target.shape)
print('MAASTRO_OS_target', MAASTRO_OS_target.shape)
print('response_MAASTRO', response_MAASTRO.shape)


# ## Data Preprocessing

# ### OUS:  Train data

# In[98]:


(OUS_D1['patient_id'] == OUS_OS_target['patient_id']).sum() # OUS_D1 patient_id is same as OUS_OS_target 


# In[99]:


# Need to choose patient_id from OUS_D1 and OUS_OS_target in response_OUS
data = list(OUS_D1['patient_id'])
mask = response_OUS['patient_id'].isin(data)
response_OUS = response_OUS[mask] 
response_OUS.head(10)


# In[100]:


# Merge OUS_D1 with response_OUS
clinical_train = pd.merge(OUS_D1, response_OUS, on='patient_id', how='inner')
clinical_train = clinical_train.loc[:, ~clinical_train.columns.isin(['DFS', 'event_DFS', 'LRC', 'event_LRC'])]
clinical_train


# In[101]:


# Drop patient_id column
clinical_train = clinical_train.drop('patient_id', axis=1)


# In[102]:


# Check null values
clinical_train.isnull().sum()


# In[103]:


# Check collinearity for OUS data 
df = clinical_train

# Create a correlation matrix
correlation_matrix = df.corr()

plt.figure(figsize=(15, 12))

# Plot a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()


# ### MAASTRO: Test data

# In[104]:


(MAASTRO_D1['patient_id'] == MAASTRO_OS_target['patient_id']).sum()


# In[105]:


response_MAASTRO.rename(columns = {'Index' : 'patient_id'}, inplace = True)


# In[106]:


# need to choose patient_id from OUS_D1 and OUS_OS_target in response_MAASTRO
data = list(MAASTRO_D1['patient_id'])
mask = response_MAASTRO['patient_id'].isin(data)
response_MAASTRO = response_MAASTRO[mask] 
response_MAASTRO


# In[107]:


# Merge MAASTRO_D1 with response_ous
clinical_test = pd.merge(MAASTRO_D1, response_MAASTRO, on='patient_id', how='inner')
clinical_test = clinical_test.loc[:, ~clinical_test.columns.isin(['DFS', 'DFS_event', 'LRC', 'LRC_event'])]
clinical_test


# In[108]:


# Check null values 
clinical_test.isnull().sum()


# In[109]:


# Some rows have null values in OS, OS_event -> Remove those rows
clinical_test[clinical_test.isnull().any(axis=1)]
clinical_test = clinical_test.dropna(how='any',axis=0) 


# In[110]:


# Drop patient_id column
clinical_test = clinical_test.drop('patient_id', axis=1)


# In[111]:


# Check collinearity for MAASTRO data 
df = clinical_test

# Create a correlation matrix
correlation_matrix = df.corr()

plt.figure(figsize=(15, 12))

# Plot a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()


# #### Check outliers for train

# In[112]:


from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

# Remove outliers method 1 
z_scores = zscore(clinical_train)
potential_outliers = (np.abs(z_scores) > 3)
potential_outliers.sum()

# Remove outliers method 2 
clf = IsolationForest(contamination=0.03)  # Adjust contamination based on your dataset
outlier_labels = clf.fit_predict(clinical_train)
potential_outliers = (outlier_labels == -1)
print(potential_outliers)
print(potential_outliers.sum())

potential_outliers.sum()


# #### Check outliers for test

# In[113]:


from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

# Remove outliers method 1 
z_scores = zscore(clinical_test)
potential_outliers = (np.abs(z_scores) > 3)
potential_outliers.sum()

# Remove outliers method 2 
clf = IsolationForest(contamination=0.03)  # Adjust contamination based on your dataset
outlier_labels = clf.fit_predict(clinical_test)
potential_outliers = (outlier_labels == -1)
print(potential_outliers)
print(potential_outliers.sum())

potential_outliers.sum()


# ### Check Kaplan Meier Curves & Log-rank test

# In[114]:


# Plot KM curve based on female column
T = clinical_train['OS']
E = clinical_train['event_OS']

ax = plt.subplot(111)

G1 = (clinical_train['female'] == 0)
kmf_G1 = KaplanMeierFitter()
kmf_G1.fit(T[G1], event_observed=E[G1], label="Group 0")
kmf_G1.plot_survival_function(ax=ax, show_censors=True, censor_styles={'ms': 8, 'marker': '+'})

G2 = (clinical_train['female'] == 1)
kmf_G2 = KaplanMeierFitter()
kmf_G2.fit(T[G2], event_observed=E[G2], label="Group 1")
kmf_G2.plot_survival_function(ax=ax, show_censors=True, censor_styles={'ms': 8, 'marker': '+'})

add_at_risk_counts(kmf_G1, kmf_G2, ax=ax)
plt.title("OUS KM survival curves based on a female column");

# Carry out multivariate log-rank test
result_mlrt = multivariate_logrank_test(clinical_train['OS'], 
                                        clinical_train['female'], 
                                        clinical_train['event_OS'])

result_mlrt.print_summary()


# In[115]:


# Plot KM curve based on hpv_related column
T = clinical_train['OS']
E = clinical_train['event_OS']

ax = plt.subplot(111)

G1 = (clinical_train['hpv_related'] == 0.)
kmf_G1 = KaplanMeierFitter()
kmf_G1.fit(T[G1], event_observed=E[G1], label="Group 0")
kmf_G1.plot_survival_function(ax=ax, show_censors=True, censor_styles={'ms': 8, 'marker': '+'})

G2 = (clinical_train['hpv_related'] == 1.)
kmf_G2 = KaplanMeierFitter()
kmf_G2.fit(T[G2], event_observed=E[G2], label="Group 1")
kmf_G2.plot_survival_function(ax=ax, show_censors=True, censor_styles={'ms': 8, 'marker': '+'})

from lifelines.plotting import add_at_risk_counts
add_at_risk_counts(kmf_G1, kmf_G2, ax=ax)
plt.title("OUS KM survival curves based on a hpv_related column");

# Carry out multivariate log-rank test
result_mlrt = multivariate_logrank_test(clinical_train['OS'], 
                                        clinical_train['hpv_related'], 
                                        clinical_train['event_OS'])

result_mlrt.print_summary()


# ## Train data from OUS for training models

# In[116]:


# X
X = clinical_train.loc[:, ~clinical_train.columns.isin(['OS', 'event_OS'])]

# y 
y = clinical_train.loc[:, ['OS', 'event_OS']]
lower, upper = np.percentile(y['OS'], [5, 95])
gbsg_times = np.arange(lower, upper + 1)

# split to train and test set 
X_train, X_test, y_train, y_test = train_test_split(
                                X, y, stratify=y['event_OS'], 
                                random_state=123)
# y into array 
lists = [] 
for i, j in zip(y['event_OS'], y['OS']): 
    lists.append((i, j))

y = np.array(lists, dtype=[('status', bool), ('time', np.int32)])


# In[117]:


# Shape
print('X_train: ', X.shape)
print('y_train: ', y.shape)


# In[118]:


clinical_test.rename(columns = {'OS_event' : 'event_OS'}, inplace = True)


# ## MAASTRO as Test data

# In[119]:


# X
X_MAASTRO = clinical_test.loc[:, ~clinical_test.columns.isin(['OS', 'event_OS'])]

# y y_MAASTRO
y_MAASTRO = clinical_test.loc[:, ['OS', 'event_OS']]
lower, upper = np.percentile(y_MAASTRO['OS'], [5, 95])
gbsg_times = np.arange(lower, upper + 1)

# y into array 
lists = [] 
for i, j in zip(y_MAASTRO['event_OS'], y_MAASTRO['OS']): 
    lists.append((i, j))

y_MAASTRO = np.array(lists, dtype=[('status', bool), ('time', np.int32)])


# ## Modelling

# ### 1. CoxPHSurvivalAnalysis

# In[120]:


# Stratified K fold 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=345)
skf.get_n_splits(X, y) 


# In[121]:


def create_objective(model_class, metric, X, y):
    def objective(trial): 
        # Create and fit survival model 
        model = model_class()
        
        scores = [] 
        
        for k, (train_idx, test_idx) in enumerate(skf.split(X, y)):  
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
    
            model.fit(X_train, y_train)
    
            if metric == "c-index":
                # Make predictions using C-index 
                c_index_score = model.score(X_test, y_test)
                scores.append(c_index_score)
                print(f"Fold {k + 1} C-index: {c_index_score}")
            elif metric == "ibs":
                # Make predictions using IBS 
                lower, upper = np.percentile(y_test["time"], [5, 95])
                gbsg_times = np.arange(lower, upper + 1)
                cox_surv_prob = np.row_stack([fn(gbsg_times) for fn in model.predict_survival_function(X_test)])
                ibs = integrated_brier_score(y_train, y_test, cox_surv_prob, gbsg_times)
                scores.append(ibs)
                print(f"Fold {k + 1} IBS: {ibs}")
            else:
                raise ValueError("Invalid metric. Use 'C-index' or 'ibs'.")
        
        # Return the mean of scores
        return np.mean(scores)
    
    return objective

# C-index
study_cindex = optuna.create_study(direction="maximize")
objective_cindex = create_objective(CoxPHSurvivalAnalysis, "c-index", X, y)
study_cindex.optimize(objective_cindex, n_trials=1, show_progress_bar=True)
print("\n")
print("* Best trial for C-index: \n", study_cindex.best_trial)
print("\n")
print("* Best Score for C-index: \n", study_cindex.best_value)

# Example usage for IBS
study_ibs = optuna.create_study(direction="minimize")
objective_ibs = create_objective(CoxPHSurvivalAnalysis, "ibs", X, y)
study_ibs.optimize(objective_ibs, n_trials=1, show_progress_bar=True)
print("\n")
print("* Best trial for IBS: \n", study_ibs.best_trial)
print("\n")
print("* Best Score for IBS: \n", study_ibs.best_value)


# ### 2. CoxnetSurvivalAnalysis - Ridge 

# In[122]:


def create_objective(model_class, metric, X, y):
    def objective(trial): 
        # Create and fit survival model 
        model = model_class(l1_ratio=0.00000001, 
                            fit_baseline_model=True)

        scores = [] 
        
        for k, (train_idx, test_idx) in enumerate(skf.split(X, y)):  
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
    
            model.fit(X_train, y_train)
    
            if metric == "c-index":
                # Make predictions using C-index 
                c_index_score = model.score(X_test, y_test)
                scores.append(c_index_score)
                print(f"Fold {k + 1} C-index: {c_index_score}")
            elif metric == "ibs":
                # Make predictions using IBS 
                lower, upper = np.percentile(y_test["time"], [5, 95])
                gbsg_times = np.arange(lower, upper + 1)
                rsf_surv_prob = np.row_stack([fn(gbsg_times) for fn in model.predict_survival_function(X_test)])
                ibs = integrated_brier_score(y_train, y_test, rsf_surv_prob, gbsg_times)
                scores.append(ibs)
                print(f"Fold {k + 1} IBS: {ibs}")
            else:
                raise ValueError("Invalid metric. Use 'C-index' or 'ibs'.")
        
        # Return the mean of scores
        return np.mean(scores)
    
    return objective

# C-index
study_cindex = optuna.create_study(direction="maximize")
objective_cindex = create_objective(CoxnetSurvivalAnalysis, "c-index", X, y)
study_cindex.optimize(objective_cindex, n_trials=1, show_progress_bar=True)
print("\n")
print("* Best trial for C-index: \n", study_cindex.best_trial)
print("\n")
print("* Best Score for C-index: \n", study_cindex.best_value)

# IBS
study_ibs = optuna.create_study(direction="minimize")
objective_ibs = create_objective(CoxnetSurvivalAnalysis, "ibs", X, y)
study_ibs.optimize(objective_ibs, n_trials=1, show_progress_bar=True)
print("\n")
print("* Best trial for IBS: \n", study_ibs.best_trial)
print("\n")
print("* Best Score for IBS: \n", study_ibs.best_value)


# ### 3. CoxnetSurvivalAnalysis - Lasso

# In[123]:


def create_objective(model_class, metric, X, y):
    def objective(trial): 
        # Create and fit survival model 
        model = model_class(l1_ratio=1, 
                            fit_baseline_model=True)

        scores = [] 
        
        for k, (train_idx, test_idx) in enumerate(skf.split(X, y)):  
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
    
            model.fit(X_train, y_train)
    
            if metric == "c-index":
                # Make predictions using C-index 
                c_index_score = model.score(X_test, y_test)
                scores.append(c_index_score)
                print(f"Fold {k + 1} C-index: {c_index_score}")
            elif metric == "ibs":
                # Make predictions using IBS 
                lower, upper = np.percentile(y_test["time"], [5, 95])
                gbsg_times = np.arange(lower, upper + 1)
                rsf_surv_prob = np.row_stack([fn(gbsg_times) for fn in model.predict_survival_function(X_test)])
                ibs = integrated_brier_score(y_train, y_test, rsf_surv_prob, gbsg_times)
                scores.append(ibs)
                print(f"Fold {k + 1} IBS: {ibs}")
            else:
                raise ValueError("Invalid metric. Use 'C-index' or 'ibs'.")
        
        # Return the mean of scores
        return np.mean(scores)
    
    return objective

# C-index
study_cindex = optuna.create_study(direction="maximize")
objective_cindex = create_objective(CoxnetSurvivalAnalysis, "c-index", X, y)
study_cindex.optimize(objective_cindex, n_trials=1, show_progress_bar=True)
print("\n")
print("* Best trial for C-index: \n", study_cindex.best_trial)
print("\n")
print("* Best Score for C-index: \n", study_cindex.best_value)

# IBS
study_ibs = optuna.create_study(direction="minimize")
objective_ibs = create_objective(CoxnetSurvivalAnalysis, "ibs", X, y)
study_ibs.optimize(objective_ibs, n_trials=1, show_progress_bar=True)
print("\n")
print("* Best trial for IBS: \n", study_ibs.best_trial)
print("\n")
print("* Best Score for IBS: \n", study_ibs.best_value)


# ### 4. CoxnetSurvivalAnalysis - ElasticNet

# In[149]:


def create_objective(model_class, metric, X, y):
    def objective(trial): 
        # Suggest values for hyperparameters
        l1_ratio = trial.suggest_float("l1_ratio", 0.0001, 1)
        
        # Create and fit survival model 
        model = model_class(l1_ratio=l1_ratio, 
                           fit_baseline_model=True)
        
        scores = [] 
        
        for k, (train_idx, test_idx) in enumerate(skf.split(X, y)):  
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
    
            model.fit(X_train, y_train)
    
            if metric == "c-index":
                # Make predictions using C-index 
                c_index_score = model.score(X_test, y_test)
                scores.append(c_index_score)
                print(f"Fold {k + 1} C-index: {c_index_score} \n")
            elif metric == "ibs":
                # Make predictions using IBS 
                lower, upper = np.percentile(y_test["time"], [5, 95])
                gbsg_times = np.arange(lower, upper + 1)
                cox_elastic_surv_prob = np.row_stack([fn(gbsg_times) for fn in model.predict_survival_function(X_test)])
                ibs = integrated_brier_score(y_train, y_test, cox_elastic_surv_prob, gbsg_times)
                scores.append(ibs)
                print(f"Fold {k + 1} IBS: {ibs} \n")
            else:
                raise ValueError("Invalid metric. Use 'C-index' or 'IBS'.")
        
        # Return the mean of scores
        return np.mean(scores)
    
    return objective

# C-index
study_cindex = optuna.create_study(direction="maximize")
objective_cindex = create_objective(CoxnetSurvivalAnalysis, "c-index", X, y)
study_cindex.optimize(objective_cindex, n_trials=50, show_progress_bar=True)
print("\n")
print("* Best trial for C-index: \n", study_cindex.best_trial)
print("\n")
print("* Best hyperparameters for C-index: \n", study_cindex.best_params)
print("\n")
print("* Best Score for C-index: \n", study_cindex.best_value)

# IBS
study_ibs = optuna.create_study(direction="minimize")
objective_ibs = create_objective(CoxnetSurvivalAnalysis, "ibs", X, y)
study_ibs.optimize(objective_ibs, n_trials=50, show_progress_bar=True)
print("\n")
print("* Best trial for IBS: \n", study_ibs.best_trial)
print("\n")
print("* Best hyperparameters for IBS: \n", study_ibs.best_params)
print("\n")
print("* Best Score for IBS: \n", study_ibs.best_value)


# ### 5. Random Survival Forest

# In[125]:


def create_objective(model_class, metric, X, y):
    def objective(trial): 
        # Suggest values for hyperparameters
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 20)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 50)
        n_estimators = trial.suggest_int("n_estimators", 20, 150)
        max_depth = trial.suggest_int("max_depth", 2, 30)
        max_samples = trial.suggest_int("max_samples", 2, 30)
        
        
        # Create and fit survival model 
        model = model_class(min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_leaf_nodes=max_leaf_nodes,
                            n_estimators=n_estimators, 
                            max_depth=max_depth,
                            max_samples=max_samples, 
                            random_state=123)
        
        scores = [] 
        
        for k, (train_idx, test_idx) in enumerate(skf.split(X, y)):  
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
    
            model.fit(X_train, y_train)
    
            if metric == "c-index":
                # Make predictions using C-index 
                c_index_score = model.score(X_test, y_test)
                scores.append(c_index_score)
                print(f"Fold {k + 1} C-index: {c_index_score}")
            elif metric == "ibs":
                # Make predictions using IBS 
                lower, upper = np.percentile(y_test["time"], [5, 95])
                gbsg_times = np.arange(lower, upper + 1)
                rsf_surv_prob = np.row_stack([fn(gbsg_times) for fn in model.predict_survival_function(X_test)])
                ibs = integrated_brier_score(y_train, y_test, rsf_surv_prob, gbsg_times)
                scores.append(ibs)
                print(f"Fold {k + 1} IBS: {ibs}")
            else:
                raise ValueError("Invalid metric. Use 'C-index' or 'IBS'.")
        
        # Return the mean of scores
        return np.mean(scores)
    
    return objective

# C-index 
study_cindex = optuna.create_study(direction="maximize")
objective_cindex = create_objective(RandomSurvivalForest, "c-index", X, y)
study_cindex.optimize(objective_cindex, n_trials=50, show_progress_bar=True)
print("\n")
print("* Best trial for C-index: \n", study_cindex.best_trial)
print("\n")
print("* Best hyperparameters for C-index: \n", study_cindex.best_params)
print("\n")
print("* Best Score for C-index: \n", study_cindex.best_value)

# IBS
study_ibs = optuna.create_study(direction="minimize")
objective_ibs = create_objective(RandomSurvivalForest, "ibs", X, y)
study_ibs.optimize(objective_ibs, n_trials=50, show_progress_bar=True)
print("\n")
print("* Best trial for IBS: \n", study_ibs.best_trial)
print("\n")
print("* Best hyperparameters for IBS: \n", study_ibs.best_params)
print("\n")
print("* Best Score for IBS: \n", study_ibs.best_value)


# ### 6. ExtraSurvivalTrees

# In[126]:


# loss > D 
# learning_rate > D
# n_estimators > D 
# subsample > D
# min_samples_split > D 
# min_samples_leaf > D 
# max_depth > D
# max_features > D
# max_leaf_nodes > D
# dropout_rate > D
# random_state > D


# In[147]:


def create_objective(model_class, metric, X, y):
    def objective(trial): 
        # Suggest values for hyperparameters 
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 20)
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 50)
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 2, 30)
        max_samples = trial.suggest_int("max_samples", 2, 30)
        
        # Create and fit survival model 
        model = model_class(min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_leaf_nodes=max_leaf_nodes,
                            n_estimators=n_estimators, 
                            max_samples=max_samples,
                            max_depth=max_depth, 
                            random_state=123) 
        
        scores = [] 
        
        for k, (train_idx, test_idx) in enumerate(skf.split(X, y)):  
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
    
            model.fit(X_train, y_train)
    
            if metric == "c-index":
                # Make predictions using C-index 
                c_index_score = model.score(X_test, y_test)
                scores.append(c_index_score)
                print(f"Fold {k + 1} C-index: {c_index_score}")
            elif metric == "ibs":
                # Make predictions using IBS 
                lower, upper = np.percentile(y_test["time"], [5, 95])
                gbsg_times = np.arange(lower, upper + 1)
                est_surv_prob = np.row_stack([fn(gbsg_times) for fn in model.predict_survival_function(X_test)])
                ibs = integrated_brier_score(y_train, y_test, est_surv_prob, gbsg_times)
                scores.append(ibs)
                print(f"Fold {k + 1} IBS: {ibs}")
            else:
                raise ValueError("Invalid metric. Use 'C-index' or 'IBS'.")
        
        # Return the mean of scores
        return np.mean(scores)
    
    return objective

# C-index
study_cindex = optuna.create_study(direction="maximize")
objective_cindex = create_objective(ExtraSurvivalTrees, "c-index", X, y)
study_cindex.optimize(objective_cindex, n_trials=50, show_progress_bar=True)
print("\n")
print("* Best trial for C-index: \n", study_cindex.best_trial)
print("* Best hyperparameters for C-index: \n", study_cindex.best_params)
print("* Best Score for C-index: \n", study_cindex.best_value)

# IBS
study_ibs = optuna.create_study(direction="minimize")
print("\n")
objective_ibs = create_objective(ExtraSurvivalTrees, "ibs", X, y)
print("\n")
study_ibs.optimize(objective_ibs, n_trials=50, show_progress_bar=True)
print("\n")
print("* Best trial for IBS: \n", study_ibs.best_trial)
print("\n")
print("* Best hyperparameters for IBS: \n", study_ibs.best_params)
print("\n")
print("* Best Score for IBS: \n", study_ibs.best_value)


# ### 7. GradientBoostingSurvivalAnalysis

# In[129]:


def create_objective(model_class, metric, X, y):
    def objective(trial): 
        # Suggest values for hyperparameters
        subsample = trial.suggest_float("subsample", 0.1, 1)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 1)
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 20)
        
        # Create and fit survival model 
        model = model_class(subsample=subsample,
                            learning_rate=learning_rate,
                            dropout_rate=dropout_rate,
                            n_estimators=n_estimators,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf, 
                            random_state=123)
        
        scores = [] 
       
        for k, (train_idx, test_idx) in enumerate(skf.split(X, y)):  
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
    
            model.fit(X_train, y_train)
    
            if metric == "c-index":
                # Make predictions using C-index 
                c_index_score = model.score(X_test, y_test)
                scores.append(c_index_score)
                print(f"Fold {k + 1} C-index: {c_index_score}")
            elif metric == "ibs":
                # Make predictions using IBS 
                lower, upper = np.percentile(y_test["time"], [5, 95])
                gbsg_times = np.arange(lower, upper + 1)
                gbs_surv_prob = np.row_stack([fn(gbsg_times) for fn in model.predict_survival_function(X_test)])
                ibs = integrated_brier_score(y_train, y_test, gbs_surv_prob, gbsg_times)
                scores.append(ibs)
                print(f"Fold {k + 1} IBS: {ibs}")
            else:
                raise ValueError("Invalid metric. Use 'c-index' or 'ibs'.")
        
        # Return the mean of scores
        return np.mean(scores)
    
    return objective

# C-index
study_cindex = optuna.create_study(direction="maximize")
objective_cindex = create_objective(GradientBoostingSurvivalAnalysis, "c-index", X, y)
study_cindex.optimize(objective_cindex, n_trials=50, show_progress_bar=True)
print("\n")
print("* Best trial for C-index: \n", study_cindex.best_trial)
print("\n")
print("* Best hyperparameters for C-index: \n", study_cindex.best_params)
print("\n")
print("* Best Score for C-index: \n", study_cindex.best_value)

# IBS
study_ibs = optuna.create_study(direction="minimize")
objective_ibs = create_objective(GradientBoostingSurvivalAnalysis, "ibs", X, y)
study_ibs.optimize(objective_ibs, n_trials=50, show_progress_bar=True)
print("\n")
print("* Best trial for IBS: \n", study_ibs.best_trial)
print("\n")
print("* Best hyperparameters for IBS: \n", study_ibs.best_params)
print("\n")
print("* Best Score for IBS: \n", study_ibs.best_value)


# ### 8. ComponentwiseGradientBoostingSurvivalAnalysis

# In[148]:


def create_objective(model_class, metric, X, y):
    def objective(trial): 
        # Suggest values for hyperparameters
        subsample = trial.suggest_float("subsample", 0.1, 1)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 1)
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1)
        
        # Create and fit survival model 
        model = model_class(subsample=subsample,
                            dropout_rate=dropout_rate,
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            random_state=123)
        
        scores = [] 
        
        for k, (train_idx, test_idx) in enumerate(skf.split(X, y)):  
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
    
            model.fit(X_train, y_train)
    
            if metric == "c-index":
                # Make predictions using C-index 
                c_index_score = model.score(X_test, y_test)
                scores.append(c_index_score)
                print(f"Fold {k + 1} C-index: {c_index_score}")
            elif metric == "ibs":
                # Make predictions using IBS 
                lower, upper = np.percentile(y_test["time"], [5, 95])
                gbsg_times = np.arange(lower, upper + 1)
                cgbs_surv_prob = np.row_stack([fn(gbsg_times) for fn in model.predict_survival_function(X_test)])
                ibs = integrated_brier_score(y_train, y_test, cgbs_surv_prob, gbsg_times)
                scores.append(ibs)
                print(f"Fold {k + 1} IBS: {ibs}")
            else:
                raise ValueError("Invalid metric. Use 'c-index' or 'ibs'.")
        
        # Return the mean of scores
        return np.mean(scores)
    
    return objective

# C-index
study_cindex = optuna.create_study(direction="maximize")
objective_cindex = create_objective(ComponentwiseGradientBoostingSurvivalAnalysis, "c-index", X, y)
study_cindex.optimize(objective_cindex, n_trials=50, show_progress_bar=True)
print("\n")
print("* Best trial for C-index: \n", study_cindex.best_trial)
print("\n")
print("* Best hyperparameters for C-index: \n", study_cindex.best_params)
print("\n")
print("* Best Score for C-index: \n", study_cindex.best_value)

# IBS
study_ibs = optuna.create_study(direction="minimize")
objective_ibs = create_objective(ComponentwiseGradientBoostingSurvivalAnalysis, "ibs", X, y)
study_ibs.optimize(objective_ibs, n_trials=50, show_progress_bar=True)
print("\n")
print("* Best trial for IBS: \n", study_ibs.best_trial)
print("\n")
print("* Best hyperparameters for IBS: \n", study_ibs.best_params)
print("\n")
print("* Best Score for IBS: \n", study_ibs.best_value)


# ## COX Assumption in Train data

# In[132]:


df = pd.DataFrame(clinical_train)

# Fit Cox proportional hazards model
cph = CoxPHFitter(penalizer=0.1)
cph.fit(df, duration_col='OS', event_col='event_OS')

# Perform the proportional hazards assumption test
results = proportional_hazard_test(cph, df)
print(results)


# ## Testing with MAASTRO Data

# In[159]:


Test_c_index = [] 
Test_ibs = [] 


# In[160]:


#1 CoxPHSurvivalAnalysis
cph = CoxPHSurvivalAnalysis()

cph.fit(X, y)

# C-index 
print('Concordance index:', cph.score(X_MAASTRO, y_MAASTRO))


# IBS 
lower, upper = np.percentile(y_MAASTRO["time"], [5, 95])
gbsg_times = np.arange(lower, upper + 1)
cph_surv_prob = np.row_stack([fn(gbsg_times) for fn in cph.predict_survival_function(X_MAASTRO)])

print('IBS score:', integrated_brier_score(y_MAASTRO, y_MAASTRO, cph_surv_prob, gbsg_times))

Test_c_index.append(cph.score(X_MAASTRO, y_MAASTRO))
Test_ibs.append(integrated_brier_score(y_MAASTRO, y_MAASTRO, cph_surv_prob, gbsg_times))


# In[161]:


#2 CoxnetSurvivalAnalysis_Ridge
cnsr = CoxnetSurvivalAnalysis(l1_ratio=0.00001,
                              fit_baseline_model=True)

cnsr.fit(X, y)

# C-index 
print('Concordance index:', cnsr.score(X_MAASTRO, y_MAASTRO))

# IBS 
lower, upper = np.percentile(y_MAASTRO["time"], [5, 95])
gbsg_times = np.arange(lower, upper + 1)
cnsr_surv_prob = np.row_stack([fn(gbsg_times) for fn in cnsr.predict_survival_function(X_MAASTRO)])

print('IBS score:', integrated_brier_score(y_MAASTRO, y_MAASTRO, cnsr_surv_prob, gbsg_times))

Test_c_index.append(cnsr.score(X_MAASTRO, y_MAASTRO))
Test_ibs.append(integrated_brier_score(y_MAASTRO, y_MAASTRO, cnsr_surv_prob, gbsg_times))


# In[162]:


#3 CoxnetSurvivalAnalysis_Lasso 
cnsl = CoxnetSurvivalAnalysis(l1_ratio=1,
                              fit_baseline_model=True)

cnsl.fit(X, y)

# C-index 
print('Concordance index:', cnsl.score(X_MAASTRO, y_MAASTRO))

# IBS 
lower, upper = np.percentile(y_MAASTRO["time"], [5, 95])
gbsg_times = np.arange(lower, upper + 1)
cnsl_surv_prob = np.row_stack([fn(gbsg_times) for fn in cnsl.predict_survival_function(X_MAASTRO)])

print('IBS score:', integrated_brier_score(y_MAASTRO, y_MAASTRO, cnsl_surv_prob, gbsg_times))

Test_c_index.append(cnsl.score(X_MAASTRO, y_MAASTRO))
Test_ibs.append(integrated_brier_score(y_MAASTRO, y_MAASTRO, cnsl_surv_prob, gbsg_times))


# In[163]:


#4 CoxnetSurvivalAnalysis_Elasticnet

# C-index
cnse = CoxnetSurvivalAnalysis(l1_ratio=0.0028463575552739127,
                              fit_baseline_model=True)

cnse.fit(X, y)

print('Concordance index:', cnse.score(X_MAASTRO, y_MAASTRO))

# IBS
cnse = CoxnetSurvivalAnalysis(l1_ratio=0.02194472394242783,
                              fit_baseline_model=True)

cnse.fit(X, y)

lower, upper = np.percentile(y_MAASTRO["time"], [5, 95])
gbsg_times = np.arange(lower, upper + 1)
cnse_surv_prob = np.row_stack([fn(gbsg_times) for fn in cnse.predict_survival_function(X_MAASTRO)])

print('IBS score:', integrated_brier_score(y_MAASTRO, y_MAASTRO, cnse_surv_prob, gbsg_times))

Test_c_index.append(cnse.score(X_MAASTRO, y_MAASTRO))
Test_ibs.append(integrated_brier_score(y_MAASTRO, y_MAASTRO, cnse_surv_prob, gbsg_times))


# In[164]:


#5 RandomSurvivalForest

# C-index 
rsf = RandomSurvivalForest(min_samples_split=14, 
                           min_samples_leaf=9, 
                           max_leaf_nodes=26, 
                           n_estimators=89, 
                           max_depth=5, 
                           max_samples=29,
                           random_state=123)

rsf.fit(X, y)

print('Concordance index:', rsf.score(X_MAASTRO, y_MAASTRO))

# IBS
rsf = RandomSurvivalForest(min_samples_split=12, 
                           min_samples_leaf=3, 
                           max_leaf_nodes=37, 
                           n_estimators=75, 
                           max_depth=27, 
                           max_samples=22,
                          random_state=123)

rsf.fit(X, y)

lower, upper = np.percentile(y_MAASTRO["time"], [5, 95])
gbsg_times = np.arange(lower, upper + 1)
rsf_surv_prob = np.row_stack([fn(gbsg_times) for fn in rsf.predict_survival_function(X_MAASTRO)])

print('IBS score:', integrated_brier_score(y_MAASTRO, y_MAASTRO, rsf_surv_prob, gbsg_times))

Test_c_index.append(rsf.score(X_MAASTRO, y_MAASTRO))
Test_ibs.append(integrated_brier_score(y_MAASTRO, y_MAASTRO, rsf_surv_prob, gbsg_times))


# In[165]:


#6 ExtraSurvivalTrees

# C-index 
est = ExtraSurvivalTrees(min_samples_split=15,
                         min_samples_leaf=5,
                         max_leaf_nodes=10,
                         n_estimators=85, 
                         max_depth=7,
                         max_samples=27, 
                         random_state=123)

est.fit(X, y)

print('Concordance index:', est.score(X_MAASTRO, y_MAASTRO))

# IBS
est = ExtraSurvivalTrees(min_samples_split=6,
                         min_samples_leaf=3,
                         max_leaf_nodes=49,
                         n_estimators=211, 
                         max_depth=12,
                         max_samples=16, 
                         random_state=123)

est.fit(X, y)

lower, upper = np.percentile(y_MAASTRO["time"], [5, 95])
gbsg_times = np.arange(lower, upper + 1)
est_surv_prob = np.row_stack([fn(gbsg_times) for fn in est.predict_survival_function(X_MAASTRO)])

print('IBS score:', integrated_brier_score(y_MAASTRO, y_MAASTRO, est_surv_prob, gbsg_times))

Test_c_index.append(est.score(X_MAASTRO, y_MAASTRO))
Test_ibs.append(integrated_brier_score(y_MAASTRO, y_MAASTRO, est_surv_prob, gbsg_times))


# In[166]:


#7 GradientBoostingSurvivalAnalysis

# C-index 
gbs = GradientBoostingSurvivalAnalysis(subsample=0.24317364187586368, 
                                       learning_rate=0.05346084350083162,
                                       dropout_rate=0.32840443885036763, 
                                       n_estimators=173, 
                                       min_samples_split=9, 
                                       min_samples_leaf=11,
                                       random_state=123)

gbs.fit(X, y)

print('Concordance index:', gbs.score(X_MAASTRO, y_MAASTRO))

# IBS
gbs = GradientBoostingSurvivalAnalysis(subsample=0.9198341676020564, 
                                       learning_rate=0.09907207133072297,
                                       dropout_rate=0.1001381164558643, 
                                       n_estimators=71, 
                                       min_samples_split=2, 
                                       min_samples_leaf=2,
                                       random_state=123)

gbs.fit(X, y)

lower, upper = np.percentile(y["time"], [5, 95])
gbsg_times = np.arange(lower, upper + 1)
gbs_surv_prob = np.row_stack([fn(gbsg_times) for fn in gbs.predict_survival_function(X_MAASTRO)])

print('IBS score:', integrated_brier_score(y_MAASTRO, y_MAASTRO, gbs_surv_prob, gbsg_times))

Test_c_index.append(gbs.score(X_MAASTRO, y_MAASTRO))
Test_ibs.append(integrated_brier_score(y_MAASTRO, y_MAASTRO, gbs_surv_prob, gbsg_times))


# In[167]:


#8 ComponentwiseGradientBoostingSurvivalAnalysis

# C-index 
cgb = ComponentwiseGradientBoostingSurvivalAnalysis(subsample=0.12449553600268824, 
                                                    dropout_rate=0.10820460307868818, 
                                                    n_estimators=262, 
                                                    learning_rate=0.07936684543536535)

cgb.fit(X, y)

print('Concordance index:', cgb.score(X_MAASTRO, y_MAASTRO))

# IBS
cgb = ComponentwiseGradientBoostingSurvivalAnalysis(subsample=0.10068995942627584, 
                                                    dropout_rate=0.2521230237935592, 
                                                    n_estimators=298, 
                                                    learning_rate=0.01715422674578889)

cgb.fit(X, y)

lower, upper = np.percentile(y_MAASTRO["time"], [5, 95])
gbsg_times = np.arange(lower, upper + 1)
cgb_surv_prob = np.row_stack([fn(gbsg_times) for fn in cgb.predict_survival_function(X_MAASTRO)])

print('IBS score:', integrated_brier_score(y_MAASTRO, y_MAASTRO, cgb_surv_prob, gbsg_times))


Test_c_index.append(cgb.score(X_MAASTRO, y_MAASTRO))
Test_ibs.append(integrated_brier_score(y_MAASTRO, y_MAASTRO, cgb_surv_prob, gbsg_times))


# ## Train Result Comparison 

# In[171]:


Train_c_index = [0.7376880491491739, 
                 0.5909652808674911, 
                 0.6162386855071665, 
                 0.6312386855071666, 
                 0.7691433576716967, 
                 0.74812260654511, 
                 0.7639039333997155, 
                 0.7539770535188426] 


Train_ibs = [0.22259654133176215, 
             0.21962018501802794,
             0.21787229871852826, 
             0.21514486257400542,
             0.18944981867585162, 
             0.1902963334786379, 
             0.20333524862117783, 
             0.193968194108332]


# In[177]:


# Train_c_index 
df_Train_c_index = Train_c_index.copy()
df_Train_c_index = pd.DataFrame(df_Train_c_index)
df_Train_c_index.index = ['CoxPH', 
                         'Coxnet Ridge', 
                         'Coxnet Lasso',
                         'Coxnet ElasticNet',
                         'Random Survival Forest',
                         'Extra Survival Trees',
                         'Gradient Boosting',
                         'Componentwise Gradient Boosting']

df_Train_c_index.rename(columns = {0:'C-index'}, inplace = True)

df_Train_c_index = df_Train_c_index.sort_values('C-index', ascending=False)
df_Train_c_index['C-index_rank'] = df_Train_c_index['C-index'].rank(ascending=False)

df_Train_c_index


# In[173]:


# Train_ibs 
df_Train_ibs = Train_ibs.copy()
df_Train_ibs = pd.DataFrame(df_Train_ibs)
df_Train_ibs.index = ['CoxPH', 
                     'Coxnet Ridge', 
                     'Coxnet Lasso',
                     'Coxnet ElasticNet',
                     'Random Survival Forest',
                     'Extra Survival Trees',
                     'Gradient Boosting',
                     'Componentwise Gradient Boosting']

df_Train_ibs.rename(columns = {0:'IBS'}, inplace = True)

df_Train_ibs = df_Train_ibs.sort_values('IBS', ascending=True)
df_Train_ibs['IBS_rank'] = df_Train_ibs['IBS'].rank(ascending=True)

df_Train_ibs


# ## Test Result Comparison 

# In[179]:


# Test_c_index 
df_Test_c_index = Test_c_index.copy()
df_Test_c_index = pd.DataFrame(df_Test_c_index)
df_Test_c_index.index = ['CoxPH', 
                         'Coxnet Ridge', 
                         'Coxnet Lasso',
                         'Coxnet ElasticNet',
                         'Random Survival Forest',
                         'Extra Survival Trees',
                         'Gradient Boosting',
                         'Componentwise Gradient Boosting']

df_Test_c_index.rename(columns = {0:'C-index'}, inplace = True)

df_Test_c_index = df_Test_c_index.sort_values('C-index', ascending=False)
df_Test_c_index['C-index_rank'] = df_Test_c_index['C-index'].rank(ascending=False)

df_Test_c_index


# In[175]:


# Test_ibs 
df_Test_ibs = Test_ibs.copy()
df_Test_ibs = pd.DataFrame(df_Test_ibs)
df_Test_ibs.index = ['CoxPH', 
                     'Coxnet Ridge', 
                     'Coxnet Lasso',
                     'Coxnet ElasticNet',
                     'Random Survival Forest',
                     'Extra Survival Trees',
                     'Gradient Boosting',
                     'Componentwise Gradient Boosting']

df_Test_ibs.rename(columns = {0:'IBS'}, inplace = True)

df_Test_ibs = df_Test_ibs.sort_values('IBS', ascending=True)
df_Test_ibs['IBS_rank'] = df_Test_ibs['IBS'].rank(ascending=True)

df_Test_ibs


# ## Statistical F-test

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




