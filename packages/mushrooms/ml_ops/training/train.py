import sys
import os
sys.path.insert(1, os.getcwd())
from packages.mushrooms.ml_source.training.train import prepare_dataset, train_model, validate_model

def run_model_training(df, test_val_size, test_size):
    print('Splitting data into training, validation and testing sets.')
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(df, 'class', test_val_size, test_size)
    except:
        raise Exception('There was an issue with the splitting of the data.')
    print('Done.')
    print('Training model.')
    try:
        model = train_model(X_train, y_train)
    except:
        raise Exception('There was an issue with model training.')
    print('Done')
    print('Validating model')
    try:
        conf_matrix = validate_model(model, X_val, y_val)
    except:
        raise Exception('There was an issue with validating the model.')
    print('Done')
    print(conf_matrix)
    
    return model, conf_matrix