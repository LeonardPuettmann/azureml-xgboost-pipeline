# Import libraries
import os 
import argparse 
import pandas as pd 
import pickle
from azureml.core import Run, Workspace
from sklearn.preprocessing import MinMaxScaler

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, dest='raw_dataset_id', help='raw dataset')
parser.add_argument('--prepped-data', type=str, dest='prepped_data', default='prepped_data', help='Folder for results')
args = parser.parse_args()
save_folder = args.prepped_data

# Get the experiment run context 
run = Run.get_context()
 
# load the data (passed as an input dataset)
print('Loading dataset...')
bike_data = run.input_datasets['raw_data'].to_pandas_dataframe()

# Log raw row count()
row_count = (len(bike_data))
run.log('raw_rows', row_count)

# remove all nulls
bike_data = bike_data.dropna()

# Normalize numeric collumns
scaler = MinMaxScaler()
num_cols = ['temp' ,'atemp' ,'humidity' ,'windspeed' ,'weather' ,'holiday' , 'workingday', 'season']
bike_data[num_cols] = scaler.fit_transform(bike_data[num_cols])

# Log processed rows
row_count = (len(bike_data))
run.log('processed_rows', row_count)

# Save the prepped data
print("Saving Data...")
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder,'bike_data.csv')
bike_data.to_csv(save_path, index=False, header=True)

# End the run
run.complete()
