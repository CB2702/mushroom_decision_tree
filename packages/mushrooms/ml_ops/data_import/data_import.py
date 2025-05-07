### DATA IMPORT MLOPS ###
from packages.mushrooms.ml_source.data_import.data_import import import_data
import logging
import pandas as pd

def import_data(path, name) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info(f'Importing {name} data from {path}')
    try:
        df = import_data(path)
    except:
        raise Exception('ERROR: There was an issue with data import. Please check file formatting.')
    logger.info(f'Success! {name} dataframe has been imported!')
    logger.info(f'{name} has {len(df)} rows and {len(df.columns)}')
    return df
