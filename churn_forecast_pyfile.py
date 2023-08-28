import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

import tensorflow as tf
from tensorflow import keras

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from xgboost import XGBClassifier

from warnings import filterwarnings
filterwarnings("ignore")


df = pd.read_csv("https://raw.githubusercontent.com/atbalazs/Telecom_Churn_Analysis_and_Forecast/main/cell2cell_data/cell2cell_dataset.csv")
df_copy = df.copy()
# data explore
print(df.describe())
print(df.info())

## FUNC: Pre-processing
#----------------------------------------------------------
#Print amount of nulls per column
def null_summary(df): 
        for col in df.columns: 
            null_count = df[col].isnull().sum()
            if null_count > 0: 
                print(f"'{col}' : ", null_count, "nulls")

# clean and enumerate credit ratings
def clean_credit_col(df): 
    if(df["CreditRating"].dtypes == "object"): 
        df["CreditRating"] = df["CreditRating"].str[0].astype(int)

# delete negatives from cols that should have only positive ints
def drop_misplaced_negatives(df, excluded_cols = []): 
        numeric_cols = df.select_dtypes(include = ["number"]).drop(columns = excluded_cols)
        filtered_df = numeric_cols.loc[~(numeric_cols < 0). any(axis = 1)]
        df = filtered_df

# Check all customer IDs are unique
def uniqueID_check(df): 
    if (df["CustomerID"].value_counts().sum() == df.shape[0]): 
        print("Customer IDs are unique")


## FUNC: Model Plots
#----------------------------------------------------------
# Plot feature importance for random forest model
def plot_feature_importance_rf(df, rf_model): 
    importances_rf = rf_model.feature_importances_
    feature_labels = df.columns[1:]
    indices = np.argsort(-(importances_rf))

    plt.figure(figsize = (14, 4))
    plt.title("Random Forest: Feature Importance")
    plt.bar(range(len(feature_labels)), 
        importances_rf[indices], 
        align = "center")

    plt.xticks(range(len(feature_labels)), 
           feature_labels[indices], 
           rotation = 90)

    plt.tight_layout()
    plt.show()

## Random Forest
#----------------------------------------------------------
# model build + model run + conf matrix + feature importance

rf = RandomForestClassifier(n_estimators = 200, random_state = 1)
clf_rf = rf.fit(X_train_res, y_train_res)
y_pred_rf = clf_rf.predict(X_test)

plot_conf_matrix(y_test, y_pred_rf)
plot_feature_importance_rf(df, clf_rf)

## XGBClassifier 
#----------------------------------------------------------
# Hyperparameter tuning + model build + model run + conf matrix

xgb_params = {
    "eta": [0.1, 0.3, 0.5, 0.7],
    "gamma": [0.5, 1, 3, 5],
    "max_depth" : [2, 5, 9],
    "min_child_weight" : [0.5, 1, 5, 10],
    "subsample" : [0.4, 0.6, 0.8], 
    "colsample_bytree" : [0.4, 0.6, 0.8]
}

skf = StratifiedKFold(n_splits = 7, shuffle = True, random_state = 1)

rs_model = RandomizedSearchCV(estimator = XGBClassifier(), 
                              param_distributions = xgb_params, 
                              n_iter = 10, 
                              scoring = "accuracy", 
                              cv = skf, 
                              verbose = 1)

rs_model.fit(X_train_res, y_train_res)

print(rs_model.best_params_)
print(rs_model.best_score_)
xgb_params_optimized = rs_model.best_params_

xgb = XGBClassifier(**xgb_params_optimized)
clf_xgb = xgb.fit(X_train_res, y_train_res)
y_pred_xgb = xgb.predict(X_test)

plot_conf_matrix(y_test, y_pred_xgb)





print("\n====================================== \n Exited: no errors")


