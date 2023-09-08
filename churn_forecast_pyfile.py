import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, StrMethodFormatter

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras import optimizers

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay

import xgboost as xgb
from xgboost import XGBClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

from warnings import filterwarnings
filterwarnings("ignore")

## FUNC: Pre-processing
#----------------------------------------------------------
#Print amount of nulls per column
# Print amount of nulls per column
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
def remove_negatives(df, columns_to_exclude=[]):
    df_copy = df.copy()
    if columns_to_exclude:
        df_copy = df_copy.drop(columns=columns_to_exclude)

    df_copy[df_copy < 0] = None 
    df_copy.dropna(inplace=True)
    return df_copy

# Check all customer IDs are unique
def uniqueID_check(df): 
    if (df["CustomerID"].value_counts().sum() == df.shape[0]): 
        print("Customer IDs are unique")

## FUNC: Model Plots
#----------------------------------------------------------
# Plot feature importance for random forest model
def plot_feature_importance_rf(df, rf_model): 
    importances_rf = rf_model.feature_importances_
    feature_labels = df.columns
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

def plot_conf_matrix(y_test, y_pred): 
    conf_matrix = confusion_matrix(y_test, y_pred)
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
    labels = [f"{v1}:\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2,2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot = labels, linewidths= 0.7, fmt = "", cmap = "flare", annot_kws= {"size": "medium"})
    print(classification_report(y_test, y_pred))

def plot_sums_by_churn_probability(df, var_to_group, var_to_sum, num_intervals):
    df["Interval"] = pd.cut(df[var_to_group], bins=num_intervals, labels=False)
    percentile_ranges = pd.qcut(df[var_to_group], num_intervals, labels=False, retbins=True)[1]
    sums_by_interval = df.groupby('Interval')[var_to_sum].sum().reset_index()

    plt.figure(figsize=(9, 5))
    plt.bar(sums_by_interval["Interval"], sums_by_interval[var_to_sum], width=0.7)
    plt.xlabel("Predicted Probability to Churn")
    plt.ylabel(f"Total {var_to_sum}")
    plt.title(f"Total {var_to_sum} by {var_to_group}")
    plt.xticks(range(num_intervals), [f"{percentile_ranges[i+1] * 100:.0f}%" for i in range(num_intervals)])
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
    plt.grid(axis="y", alpha = 0.7)
    plt.show()

def plot_sums_by_threshold(df, thresholds, var_to_group, var_to_sum):
    sums = []
    for num in thresholds:
        sum_of_var = df.loc[df[var_to_group] > num, var_to_sum].sum()
        sums.append(sum_of_var)
    sorted_thresholds, sorted_sums = zip(*sorted(zip(thresholds, sums)))

    plt.figure(figsize=(9, 5))
    plt.bar(range(len(sorted_thresholds)), sorted_sums, width = 0.8)
    plt.xlabel("Predicted Churn Thresholds")
    plt.ylabel(f"Total {var_to_sum}")
    plt.title(f"Total {var_to_sum} for Thresholds of {var_to_group}")
    plt.xticks(np.arange(len(sorted_thresholds)), [f'{num * 100:.0f}%' for num in sorted_thresholds])
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    plt.show()

## Data Load and Preprocess
#----------------------------------------------------------
## convert to pipeline in later version
df = pd.read_csv("https://raw.githubusercontent.com/atbalazs/Telecom_Churn_Analysis_and_Forecast/main/cell2cell_data/cell2cell_dataset.csv")
df_copy = df.copy()
# data explore
print(df.describe())
print(df.info())

# checking nulls - 20,000 rows without label - fewer than 2% of nontarget rows with null, dropping all null for simplicity
null_summary(df)
print("Rows before drop all null: ", df.shape[0])
print("Rows after drop all null: ", df.dropna().shape[0])

df.dropna(inplace = True)

## Type conversion
df["HandsetModels"] = df["HandsetModels"].astype("object")
df["Handsets"] = df["Handsets"].astype(int)

for col in df.columns: 
        if df[col].dtype == "object": 
                print(df[col].value_counts())

clean_credit_col(df)
df.drop(["HandsetPrice", "NotNewCellphoneUser"], axis = 1, inplace = True)

# Check data balancing
df["Churn"].value_counts()

df_numerical = df.select_dtypes(include = "number")
df_categorical = df.select_dtypes(exclude = "number")

# Copying then dropping customerID as feature
customerIDs = df["CustomerID"]
df.drop("CustomerID", axis = 1, inplace = True)

# Encode churn values: 
churn_mapping = {"Yes": 1, "No": 0}
df["Churn"] = df["Churn"].map(churn_mapping)

# Label encoder
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
df[df_categorical.columns.tolist()] = df[df_categorical.columns.tolist()].apply(LabelEncoder().fit_transform)

# Check for all cols being numeric
print((df.dtypes == "object").any())

X = df.drop("Churn", axis = 1)
y = df["Churn"]

# chi2 feature selection
cols_with_neg = ["PercChangeMinutes", "PercChangeRevenues"]
df_pos = df.drop(columns = cols_with_neg)
df_pos = remove_negatives(df, cols_with_neg)

X_pos = df_pos.drop("Churn", axis = 1)
y_pos = df_pos["Churn"]

percent_feats_to_drop = 20
feature_selection_chi2 = SelectKBest(score_func = chi2, k = int(X.shape[1] * (1 - (percent_feats_to_drop/100))))

chi2_features = feature_selection_chi2.fit(X_pos, y_pos)
chi2_feature_names = chi2_features.get_feature_names_out()
selected_feature_names = np.concatenate((chi2_feature_names, cols_with_neg))

X = X[selected_feature_names]

print(X_pos.shape, y_pos.shape)
print(X.shape, y.shape)

# Train-test-split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, customer_ids_train, customer_ids_test = train_test_split(
    X, y, customerIDs, test_size=0.2, random_state=42)

X_train_copy = X_train.copy()
X_test_copy = X_test.copy()

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Upsample with random oversampling
from random import Random
from imblearn.over_sampling import RandomOverSampler
rand_oversampler = RandomOverSampler(random_state = 1)
X_train, y_train = rand_oversampler.fit_resample(X_train, y_train)

# Check resample balancing
np.unique(y_train, return_counts = True)[1]

## Simple Logistic Regression 
#----------------------------------------------------------
# model build + hyperparameter tuning + model fit + conf matrix

lr_grid = {
    "C": np.logspace(-4, 4, 50), 
    "penalty": ["l1","l2"]
    }

clf_GS = GridSearchCV(LogisticRegression(), 
                      param_grid = lr_grid,
                      verbose = 1, 
                      scoring = "accuracy"
                      )

clf_GS.fit(X_train, y_train)

lr_params_optimized = clf_GS.best_params_

lr = LogisticRegression(**lr_params_optimized)
clf_lr = lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)

plot_conf_matrix(y_test, y_pred_lr)

## Random Forest
#----------------------------------------------------------
# model build + hyperparameter tuning + model fit + conf matrix + feature importance

rf_params = { 
    "bootstrap": [True, False],
    "max_depth": [10, 20, 30, 40, 50, 60, 70, None],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 5, 10],
    "n_estimators": [50, 100, 200, 500, 750, 1000]}

rs_model_rf = RandomizedSearchCV(estimator = RandomForestClassifier(), 
                                 param_distributions = rf_params, 
                                 n_iter = 50, 
                                 cv = 3, 
                                 verbose = 3,
                                 random_state = 1)

rs_model_rf.fit(X_train, y_train)

rf_best_params = rs_model_rf.best_params_

rf = RandomForestClassifier(**rf_best_params)
clf_rf = rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

plot_conf_matrix(y_test, y_pred_rf)
plot_feature_importance_rf(X, clf_rf)

## XGBClassifier 
#----------------------------------------------------------
# Hyperparameter tuning + model build + model fit + conf matrix

xgb_params = {
    "eta": [0.1, 0.3, 0.5, 0.7],
    "gamma": [0.5, 1, 3, 5],
    "max_depth" : [2, 5, 9],
    "min_child_weight" : [0.5, 1, 5, 10],
    "subsample" : [0.4, 0.6, 0.8], 
    "colsample_bytree" : [0.4, 0.6, 0.8]
}

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1)

rs_model_xgb = RandomizedSearchCV(estimator = XGBClassifier(), 
                              param_distributions = xgb_params, 
                              n_iter = 10, 
                              scoring = "accuracy", 
                              cv = skf, 
                              verbose = 1)

rs_model_xgb.fit(X_train, y_train)

xgb_params_optimized = rs_model_xgb.best_params_

xgb = XGBClassifier(**xgb_params_optimized)
clf_xgb = xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

plot_conf_matrix(y_test, y_pred_xgb)

## Neural Network
#----------------------------------------------------------
# Model build + model fit + conf matrix

tf.random.set_seed(1)
hidden_units = 30
learning_rate = 0.01

nn = keras.Sequential()
nn.add(Dense(X_train.shape[1], input_dim = X_train.shape[1], activation = "relu"))
nn.add(Dropout(0.5))
nn.add(Dense(hidden_units, activation = "relu"))
nn.add(Dense(hidden_units, activation = "relu"))
nn.add(Dense(1, activation = "sigmoid"))

sgd = optimizers.legacy.SGD(lr = learning_rate)
nn.compile(loss = "binary_crossentropy", optimizer = sgd, metrics = ["accuracy"])

nn.fit(X_train, y_train, epochs = 30, verbose = 1)
nn.evaluate(X_test, y_test)
nn.summary()

## Identifying Revenue Risk and Highest-Risk Customers
#----------------------------------------------------------
# Best model selection + matching CustomerID with churn probability + revenue at risk + 

# Selected random forest with tuned hyperparameters due to high recall
predictions_rf = clf_rf.predict_proba(X_test)[:, 1]

churn_pred_by_customer = pd.DataFrame({"CustomerID": customer_ids_test,
                                    "ChurnPred" : predictions_rf})

predicted_test_df = pd.concat([churn_pred_by_customer, 
                              pd.DataFrame(X_test_copy, columns= selected_feature_names)], 
                              axis = 1).sort_values(by = "ChurnPred", ascending = False)

print(predicted_test_df.head(5))

plot_sums_by_churn_probability(predicted_test_df, "ChurnPred", "MonthlyRevenue", 12)
plot_sums_by_threshold(predicted_test_df, [0.75, 0.7, 0.65, 0.6], "ChurnPred", "MonthlyRevenue")

# Displaying CustomerID for customers in 90th percentile of churn probability, sorted by highest risk
highest_risk_customers = pd.DataFrame(predicted_test_df.loc[predicted_test_df["ChurnPred"] >= predicted_test_df["ChurnPred"].quantile(0.9), 
                                                            ["CustomerID", "ChurnPred"]]).sort_values(by = "ChurnPred", ascending = False)

print(highest_risk_customers)

print("\n====================================== \n Exited: no errors")


