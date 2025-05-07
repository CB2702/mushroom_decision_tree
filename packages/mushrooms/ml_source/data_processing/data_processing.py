import pandas as pd
import numpy as np

def cols_dash_to_underscore(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [x.replace('-', '_') for x in df.columns]
    df.columns = [str.lower(x) for x in df.columns]

    return df

def categorize_numerical_column(df: pd.DataFrame, col: str, dp: int) -> pd.DataFrame:
    df[col] = df[col].round(dp).astype(str).astype('category')

    return df

def enforce_dtype(df: pd.DataFrame, col: str, type):
    df[col] = df[col].astype(type)

    return df

def boxplot_clean(df: pd.DataFrame, cont_var_col: str) -> pd.DataFrame :

    # find quartiles and iqr
    var_lq = np.percentile(df[cont_var_col], 25)
    var_uq = np.percentile(df[cont_var_col], 75)
    var_iqr = var_uq - var_lq

    # find thresholds
    var_ut = var_uq + (1.5 * var_iqr)
    var_lt = var_lq - (1.5 * var_iqr)

    # filter df
    df = df.loc[(df[cont_var_col] >= var_lt) & (df[cont_var_col] <= var_ut)]

    return df

def calculate_volume(df: pd.DataFrame, width_col, height_col) -> pd.DataFrame:
    df['stem_volume'] = (3.14 * ((df[width_col] / 2) ** 2) * df[height_col])

    return df

def calculate_area(df: pd.DataFrame, width_col) -> pd.DataFrame:
    df['cap_area'] = 3.14 * ((df[width_col] / 2) ** 2)

    return df

def calculate_w_h_ratio(df: pd.DataFrame, width_col, height_col) -> pd.DataFrame:
    df['stem_ratio'] = df[width_col]/df[height_col]

    return df

def onehot_encode_cats(df: pd.DataFrame, cat_cols: list, dtype) -> pd.DataFrame:
    df = pd.get_dummies(df, columns = cat_cols, dtype = dtype)
    
    return df