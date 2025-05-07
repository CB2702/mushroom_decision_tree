from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

def prepare_dataset(df: pd.DataFrame, target_var: str, val_test_size: float, test_size: float):
    X = df.drop(columns = target_var)
    y = df[target_var]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=test_size)

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(X_train, y_train):
    model = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth = 30)
    trained_model = model.fit(X_train, y_train)

    return trained_model

def validate_model(model, X_val, y_val):
    y_hat_val = model.predict(X_val)
    X_val['y_hat'] = y_hat_val
    X_val['y'] = y_val
    results_conf_mat = X_val.groupby(['y', 'y_hat']).agg(pct_results = ('stem_width', 'count')).reset_index()
    results_conf_mat['pct_results'] = results_conf_mat['pct_results']/len(X_val)
    
    return results_conf_mat