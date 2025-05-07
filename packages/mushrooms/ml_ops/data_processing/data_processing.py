import sys
import os
sys.path.insert(1, os.getcwd())
from packages.mushrooms.ml_source.data_processing.data_processing import cols_dash_to_underscore, categorize_numerical_column, enforce_dtype, boxplot_clean, calculate_area, calculate_volume, calculate_w_h_ratio, onehot_encode_cats

def run_data_processing(df):
    print('Running data processing on dataframe.')
    print(f'Converting columns {df.columns} to lowercase_underscore format.')
    try:
        df = cols_dash_to_underscore(df)
    except:
        raise Exception('There was an issue with column underscore conversion.')
    print('Done.')
    
    print('Converting season to categorical variable.')
    try:
        df = categorize_numerical_column(df, 'season', 2)
    except:
        raise Exception('There was an issue with type conversion to category.')
    print('Done.')
    
    numerical_cols = ['cap_diameter', 'stem_width', 'stem_height']
    print(f'Cleaning outliers from numerical variables {numerical_cols}.')
    for col in numerical_cols:
        try:
            df = boxplot_clean(df, col)
        except:
            raise Exception(f'There was an issue with outlier cleaning for {col}.')
    print('Done')
    
    print('Calculating mushroom proportions.')
    try:
        df = calculate_volume(df, 'stem_width', 'stem_height')
    except:
        raise Exception('There was an issue with calculating the stem volume.')
    try:
        df = calculate_area(df, 'cap_diameter')
    except:
        raise Exception('There was an issue with calculating the cap area.')
    try:
        df = calculate_w_h_ratio(df, 'stem_width', 'stem_height')
    except:
        raise Exception('There was an issue with calculating the stem w/h ratio')
    print('Done.')
    
    print('Collecting categorical variables for one-hot-encoding')
    numerical_cols = ['stem_volume', 'cap_area', 'stem_ratio'] + ['cap_diameter', 'stem_height', 'stem_width'] + ['class']
    cat_cols = [col for col in list(df.columns) if col not in numerical_cols]
    for col in cat_cols:
        if col != 'class':
            df[col] = df[col].astype('category')
    df = onehot_encode_cats(df, cat_cols, dtype = int)
    print('Done.')
    

    return df