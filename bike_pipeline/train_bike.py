# Import libraries
from azureml.core import Run, Model
import argparse
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval

import xgboost as xgb

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--training-data", type=str, dest='training_data', help='training data')
args = parser.parse_args()
training_data = args.training_data

# Get the experiment run context
run = Run.get_context()

# load the prepared data file in the training folder
print("Loading Data...")
file_path = os.path.join(training_data, 'bike_data.csv')
bike_data = pd.read_csv(file_path)

# Separate features and labels
X, y = bike_data[['temp' ,'atemp' ,'humidity' ,'windspeed' ,'weather' ,'holiday' , 'workingday', 'season']].values, bike_data['count'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

space = {
    'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 1.00, 0.05)),
    'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 1, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'subsample':        hp.choice('subsample',        [0, 0.50, 0.8, 1]),
    'n_estimators':     hp.choice('n_estimators',     range(10, 500, 10)),
    'tree_method':      hp.choice('tree_method',      ['hist'])
}

# Objective function
def objective(params):
    # Instantiate model
    model = xgb.XGBRegressor(**params)

    # Fit and predict
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    y_hat = y_hat.clip(min=0)
    
    # Calculate the root mean squared error
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_hat))

    # Retrun loss, status and model
    return {'loss': rmsle, 'status': STATUS_OK, 'model': model}

# Trials to track progress
bayes_trials = Trials()

# Optimize
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=200, trials=bayes_trials)
params = space_eval(space, best)

# Train light gradient boosting model
print('Training a decision tree model...')
model = xgb.XGBRegressor(**params).fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
y_hat = y_hat.clip(min=0)
msle = mean_squared_log_error(y_test, y_hat)
print(msle)

# Save the trained model in the outputs folder
print("Saving model...")
os.makedirs('outputs', exist_ok=True)
model_file = os.path.join('outputs', 'bike_model.pkl')
joblib.dump(value=model, filename=model_file)

# Register the model
print('Registering model...')
Model.register(workspace=run.experiment.workspace,
               model_path = model_file,
               model_name = 'bike_model',
               tags={'Training context':'Pipeline'},
               properties={'MSE': np.float(msle)})

run.complete()
