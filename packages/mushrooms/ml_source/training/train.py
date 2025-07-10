from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

def prepare_dataset(df: pd.DataFrame, target_var: str, val_test_size: float, test_size: float):
    X = df.drop(columns = target_var)
    y = df[target_var]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_test_size)

    return X_train, X_val, y_train, y_val

def train_model(X_train, y_train):
    model = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth = 30)
    trained_model = model.fit(X_train, y_train)

    return trained_model

def evaluate_model(model, X_val, y_val):
    y_hat_val = model.predict(X_val)
    X_val['y_hat'] = y_hat_val
    X_val['y'] = y_val
    results_conf_mat = X_val.groupby(['y', 'y_hat']).agg(pct_results = ('stem_width', 'count')).reset_index()
    results_conf_mat['pct_results'] = results_conf_mat['pct_results']/len(X_val)
    results_conf_mat['model'] = 'DecisionTree_V1'
    results_conf_mat.to_json('results/conf_matrix_train_dt_v2.json')
    
    return results_conf_mat

def get_model_feature_importance(model, X_val):
    importances = model.feature_importances_
    features = X_val.columns
    importance_df = pd.DataFrame({'feature': features, 'importances': importances}, index = range(len(importances)))
    importance_df['model_name'] = 'DecisionTree_V1'
    importance_df.to_json('results/feature_importance_dt_v2.json')

    return importance_df
