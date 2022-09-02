## About

With the `.ipynb` file in this repository, you can deploy and run a pipeline in azureml. This pipeline creates an XGBoost regressor on the [bike sharing dataset](https://www.kaggle.com/c/bike-sharing-demand). However, you could modify it so that is works on other datasets as well.

# Requirements

- Python 3.7
- Active Azure subscription
- Active AzureML workspace
- config.json from your AzureML workspace (or edit the one in this repo with your credentials)

Please make sure that you use Python 3.7 for this, because the azureml-sdk is incompatible with higher versions of Python. I recommend creating a virtual env with the following command:
`py 3.7 -m venv azureml-env`

And then, inside the new environment:
`pip install -r requirements.txt`