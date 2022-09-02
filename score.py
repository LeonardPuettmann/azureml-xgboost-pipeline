# Deploy model as an endpoint
import json
import joblib 
import os 

# Called when a service is loaded
def init():
    global model
    # Get the path to the registered model file and load it
    model_path = os.path.join(os.getenv('AZURE_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)

# Called when a request is recieved
def run(raw_data):
    # Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])

    # Get a prediction from a model
    predictions = model.predict(data)

    # Return 
    return predictions.tolist()