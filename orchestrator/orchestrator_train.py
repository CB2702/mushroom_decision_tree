import sys
import os
sys.path.insert(1, os.getcwd())
import pandas as pd
from packages.mushrooms.ml_ops.data_import.data_import import run_import_data
from packages.mushrooms.ml_ops.data_processing.data_processing import run_data_processing
from packages.mushrooms.ml_ops.training.train import run_model_training

df = run_import_data(path = 'data/mushroom_cleaned.csv', name = 'mushrooms')
print(df.head())

df = run_data_processing(df)
print(df)

run_model_training(df, 0.3, 0.5)