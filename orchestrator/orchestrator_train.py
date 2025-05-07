import pandas as pd
from packages.mushrooms.ml_ops.data_import.data_import import import_data
import sys
import os

# Force Python to recognize the project root (where 'packages' is)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Python path:")
print("\n".join(sys.path))  # Just for debugging, can remove later

df = import_data(path = 'data\mushroom_cleaned.csv', name = 'mushrooms')