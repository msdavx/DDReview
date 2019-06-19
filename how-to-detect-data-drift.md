---
title: How to detect data drift (Preview) on AKS deployments
titleSuffix: Azure Machine Learning service
description: Learn how to detect data drift on AKS deployed models in Azure Machine Learning service.
services: machine-learning
ms.service: machine-learning
ms.subservice: core
ms.topic: conceptual
ms.reviewer: jmartens
ms.author: copeters
author: cody-dkdc
ms.date: 06/17/2019
---
Machine learning models are generally only as good as the data they are trained with. Deploying a model to production without monitoring its performance can lead to undetected and detrimental impacts. However, directly calculating model performance may take time, resources, or otherwise be impractical. Instead, we can monitor for data drift. 

We define data drift as *a change in data that causes degradation in model performance*. We can compute data drift between the data used to inference/score a model and baseline data, most commonly the data used to train the model. See [Data Drift](./concept-data-drift.md) for details on how data drift is measured. 

> [!Note]
> This service is in (Preview) and limited in configuration options. Please see our [API Documentation](https://docs.microsoft.com/en-us/python/api/azureml-contrib-datadrift/?view=azure-ml-py) and [Release Notes](azure-machine-learning-release-notes) for details and updates. 

# How to detect data drift on models deployed to AKS (Preview)
With Azure Machine Learning service, you can monitor   the inputs to a model deployed on AKS and compare this data to a baseline dataset – typically, the training dataset for the model. At regular intervals, the inference data is [snapshot and profiled](./how-to-explore-prepare-data.md), then computed against the baseline dataset to produce a drift analysis that: 

* Measures the magnitude of data drift, called the [Drift Coefficient](./concept-data-drift.md)

* Measures the Drift Contribution by Feature, informing which features caused data drift

* Measures the Distance Metrics, currently Wasserstein and Energy Distance are computed 

* Measures the Distributions of Features, currently Kernel Density Estimation

* Send alerts to data drift by email 

In this article, we will show how to train, deploy, and monitor data drift on a model with the Azure ML service. 

## Prerequisites

- If you don’t have an Azure subscription, create a free account before you begin. Try the [free or paid version of Azure Machine Learning service](https://aka.ms/AMLFree) today.

- An Azure Machine Learning service Workspace and the Azure Machine Learning SDK for Python installed. Learn how to get these prerequisites using the [How to configure a development environment](how-to-configure-environment.md) document.

- [Set up your environment](how-to-configure-environment.md) and install the Data Drift SDK, Datasets SDK, and lightgbm package:

```
pip install azureml-contrib-datadrift
pip install azureml-contrib-datasets
pip install lightgbm
```

## Import Dependencies 
Import dependencies used in this guide:

```python
import json
import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests

# Azure ML service packages 
from azureml.contrib.datadrift import DataDriftDetector, AlertConfiguration
from azureml.contrib.opendatasets import NoaaIsdWeather
from azureml.core import Dataset, Workspace, Run
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.experiment import Experiment
from azureml.core.image import ContainerImage
from azureml.core.model import Model
from azureml.core.webservice import Webservice, AksWebservice
from azureml.widgets import RunDetails
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
``` 

## Set up 

Set up naming for this guide:

```python 
# set a prefix
prefix = 'dkdc'

# model, image, and service name to be used in the Azure Machine Learning service
model_name   = '{}DriftModel'.format(prefix)
image_name   = '{}DriftImage'.format(prefix)
service_name = '{}DriftAksService'.format(prefix)
dataset_name = '{}DriftTrainingDataset'.format(prefix)
aks_name     = '{}DriftAksCompute'.format(prefix)

# set email address that will recieve the alert of data drift
email_address = ''
```

## Get Azure Machine Learning service Workspace
Learn how to create this with the [How to configure a development environment](how-to-configure-environment.md) document.

```python
# load workspace object from config file
ws = Workspace.from_config()

# print workspace details
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')
```

## Generate Training and Test Data
For this guide, we will use NOAA ISD weather data from [Azure Open Datasets](https://azure.microsoft.com/en-us/services/open-datasets/) to demonstrate the Data Drift service.

First, create functions for getting and formatting the data.

```python
usaf_list = ['725724', '722149', '723090', '722159', '723910', '720279',
 '725513', '725254', '726430', '720381', '723074', '726682',
 '725486', '727883', '723177', '722075', '723086', '724053',
 '725070', '722073', '726060', '725224', '725260', '724520',
 '720305', '724020', '726510', '725126', '722523', '703333',
 '722249', '722728', '725483', '722972', '724975', '742079',
 '727468', '722193', '725624', '722030', '726380', '720309',
 '722071', '720326', '725415', '724504', '725665', '725424',
 '725066']

columns = ['usaf', 'wban', 'datetime', 'latitude', 'longitude', 'elevation', 'windAngle', 'windSpeed', 'temperature', 'stationName', 'p_k']


def enrich_weather_noaa_data(noaa_df):
hours_in_day = 23
week_in_year = 52

noaa_df["hour"] = noaa_df["datetime"].dt.hour
noaa_df["weekofyear"] = noaa_df["datetime"].dt.week

noaa_df["sine_weekofyear"] = noaa_df['datetime'].transform(lambda x: np.sin((2*np.pi*x.dt.week-1)/week_in_year))
noaa_df["cosine_weekofyear"] = noaa_df['datetime'].transform(lambda x: np.cos((2*np.pi*x.dt.week-1)/week_in_year))

noaa_df["sine_hourofday"] = noaa_df['datetime'].transform(lambda x: np.sin(2*np.pi*x.dt.hour/hours_in_day))
noaa_df["cosine_hourofday"] = noaa_df['datetime'].transform(lambda x: np.cos(2*np.pi*x.dt.hour/hours_in_day))

return noaa_df

def add_window_col(input_df):
shift_interval = pd.Timedelta('-7 days') # your X days interval
df_shifted = input_df.copy()
df_shifted['datetime'] = df_shifted['datetime'] - shift_interval
df_shifted.drop(list(input_df.columns.difference(['datetime', 'usaf', 'wban', 'sine_hourofday', 'temperature'])), axis=1, inplace=True)

# merge, keeping only observations where -1 lag is present
df2 = pd.merge(input_df,
   df_shifted,
   on=['datetime', 'usaf', 'wban', 'sine_hourofday'],
   how='inner',  # use 'left' to keep observations without lags
   suffixes=['', '-7'])
return df2

def get_noaa_data(start_time, end_time, cols, station_list):
isd = NoaaIsdWeather(start_time, end_time, cols=cols)
# Read into Pandas data frame.
noaa_df = isd.to_pandas_dataframe()
noaa_df = noaa_df.rename(columns={"stationName": "station_name"})

df_filtered = noaa_df[noaa_df["usaf"].isin(station_list)]
df_filtered.reset_index(drop=True)

# Enrich with time features
df_enriched = enrich_weather_noaa_data(df_filtered)

return df_enriched

def get_featurized_noaa_df(start_time, end_time, cols, station_list):
df_1 = get_noaa_data(start_time - timedelta(days=7), start_time - timedelta(seconds=1), cols, station_list)
df_2 = get_noaa_data(start_time, end_time, cols, station_list)
noaa_df = pd.concat([df_1, df_2])

print("Adding window feature")
df_window = add_window_col(noaa_df)

cat_columns = df_window.dtypes == object
cat_columns = cat_columns[cat_columns == True]

print("Encoding categorical columns")
df_encoded = pd.get_dummies(df_window, columns=cat_columns.keys().tolist())

print("Dropping unnecessary columns")
df_featurized = df_encoded.drop(['windAngle', 'windSpeed', 'datetime', 'elevation'], axis=1).dropna().drop_duplicates()

return df_featurized
```

For this guide, we will try to predict the temperature for each day of the next week, given the data from the past 2 weeks. Naively, we will only train on 2 weeks of data from January 2009.

```python 
# load Jan 1-14 2009 data as a dataframe
df = get_featurized_noaa_df(datetime(2009, 1, 1), datetime(2009, 1, 14, 23, 59, 59), columns, usaf_list)
df.head()
```

We will split the data for training the model and write the data to a CSV for uploading and registration as an Azure Machine Learning dataset. This training dataset will be associated with the model and used as the baseline dataset for the drift service. 

```python
# generate X and y dataframes for training and testing the model
label = "temperature"
x_df = df.drop(label, axis=1)
y_df = df[[label]]
x_train, x_test, y_train, y_test = train_test_split(df, y_df, test_size=0.2, random_state=223)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# create directory to write training data to
training_dir = 'outputs/training'
training_file = "training.csv"

# write dataframe to csv to we can upload and register it as a dataset
os.makedirs(training_dir, exist_ok=True)
training_df = pd.merge(x_train.drop(label, axis=1), y_train, left_index=True, right_index=True)
training_df.to_csv(training_dir + "/" + training_file)
```

## Upload, Create, Register, and Snapshot the Training Dataset

For the Data Drift Service (Preview) uses a baseline dataset for comparison against inference data. In this step we will create the training dataset which is automatically used as the baseline dataset by the service.


```python

name_suffix = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
snapshot_name = "snapshot-{}".format(name_suffix)

dstore = ws.get_default_datastore()
dstore.upload(training_dir, "data/training", show_progress=True)
dpath = dstore.path("data/training/training.csv")
trainingDataset = Dataset.auto_read_files(dpath, include_path=True)
trainingDataset = trainingDataset.register(workspace=ws, name=dataset_name, description="dset", exist_ok=True)

trainingDataSnapshot = trainingDataset.create_snapshot(snapshot_name=snapshot_name, compute_target=None, create_data_snapshot=True)
datasets = [(Dataset.Scenario.TRAINING, trainingDataSnapshot)]
print("dataset registration done.\n")
datasets
```

## Train a model

For this guide, we will train a lightgbm model to predict next week's temperature based on the past two weeks of weather data. 

```python
import lightgbm as lgb

train = lgb.Dataset(data=x_train, 
                    label=y_train)

test = lgb.Dataset(data=x_test, 
                   label=y_test,
                   reference=train)

params = {'learning_rate'    : 0.1,
          'boosting'         : 'gbdt',
          'metric'           : 'rmse',
          'feature_fraction' : 1,
          'bagging_fraction' : 1,
          'max_depth': 6,
          'num_leaves'       : 31,
          'objective'        : 'regression',
          'bagging_freq'     : 1,
          "verbose": -1,
          'min_data_per_leaf': 100}

model = lgb.train(params, 
                  num_boost_round=500,
                  train_set=train,
                  valid_sets=[train, test],
                  verbose_eval=50,
                  early_stopping_rounds=25)
                  
                  
# save model as a pickle file for use with Azure Machine Learning service
model_file = 'outputs/{}.pkl'.format(model_name)

os.makedirs('outputs', exist_ok=True)
joblib.dump(model, model_file)
```

## Register Model with Training Dataset

In this step, we register the model with the Azure Machine Learning service and specify the training dataset.

```python
model = Model.register(model_path=model_file,
                       model_name=model_name,
                       workspace=ws,
                       datasets=datasets)

print(model_name, image_name, service_name, model)
```

## Prepare Image Environment

In this step, we create a conda *myenv.yml* file with the required dependencies for running the model in AKS.

```python
myenv = CondaDependencies.create(conda_packages=['numpy','scikit-learn', 'joblib', 'lightgbm', 'pandas'],
                                 pip_packages=['azureml-monitoring', 'azureml-sdk[automl]'])

with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())
```

## Write score.py

In this step, we will write the score.py file. We need to use the [Model Data Collector](./how-to-enable-data-collection.md) to collect ata from the deployment.

```python
%%writefile score.py  # use this function in a Jupyter notebook to write the score.py file or copy the code directly into the file

import pickle
import json
import numpy
import azureml.train.automl
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from azureml.core.model import Model
from azureml.core.run import Run
from azureml.monitoring import ModelDataCollector
import time
import pandas as pd


def init():
    global model, inputs_dc, prediction_dc, feature_names, categorical_features

    print("Model is initialized" + time.strftime("%H:%M:%S"))
    model_path = Model.get_model_path(model_name=model_name)
    model = joblib.load(model_path)

    # specify feature names 
    feature_names = ["usaf", "wban", "latitude", "longitude", "station_name", "p_k",
                     "sine_weekofyear", "cosine_weekofyear", "sine_hourofday", "cosine_hourofday",
                     "temperature-7"]

    # drop these later
    categorical_features = ["usaf", "wban", "p_k", "station_name"]

    # setup model input data collection
    inputs_dc = ModelDataCollector(model_name=model_name,
                                   identifier="inputs",
                                   feature_names=feature_names)
    # setup model output data collection
    prediction_dc = ModelDataCollector(model_name=model_name,
                                       identifier="predictions",
                                       feature_names=["temperature"])


def run(raw_data):
    global inputs_dc, prediction_dc

    try:
        # load in the data
        data = json.loads(raw_data)["data"]
        data = pd.DataFrame(data)

        # remove the categorical features as the model expects OHE values
        input_data = data.drop(categorical_features, axis=1)

        result = model.predict(input_data)

        # Collect the non-OHE dataframe
        collected_df = data[feature_names]

        # collect model input data
        inputs_dc.collect(collected_df.values)
        
        # collect model output data
        prediction_dc.collect(result)
        
        # return model output 
        return result.tolist()
    except Exception as e:
        error = str(e)

        print(error + time.strftime("%H:%M:%S"))
        return error
```

## Create Image

In this step, we create the Docker Image containing the model, configuration details, and score.py for deployment to AKS. 

```python

# Image creation may take up to 15 minutes.

image_name = image_name + str(model.version)

if image_name not in ws.images:
    # Use the score.py defined in this directory as the execution script
    # NOTE: The Model Data Collector must be enabled in the execution script for DataDrift to run correctly
    image_config = ContainerImage.image_configuration(execution_script="score.py",
                                                      runtime="python",
                                                      conda_file="myenv.yml",
                                                      description="Image with weather dataset model")
                                                      
    image = ContainerImage.create(name=image_name,
                                  models=[model],
                                  image_config=image_config,
                                  workspace=ws)

    image.wait_for_creation(show_output=True)
else:
    image = ws.images[image_name]
```

## Create Compute Target

In this step, we create the AKS compute target.

```python

prov_config = AksCompute.provisioning_configuration()

if not aks_name in ws.compute_targets:
    aks_target = ComputeTarget.create(workspace=ws,
                                      name=aks_name,
                                      provisioning_configuration=prov_config)

    aks_target.wait_for_completion(show_output=True)
    print(aks_target.provisioning_state)
    print(aks_target.provisioning_errors)
else:
    aks_target=ws.compute_targets[aks_name]
```

## Deploy Model to Service

In this step, we deploy the image to AKS. 

```python

aks_service_name = service_name

if aks_service_name not in ws.webservices:
    aks_config = AksWebservice.deploy_configuration(collect_model_data=True, enable_app_insights=True)
    aks_service = Webservice.deploy_from_image(workspace=ws,
                                               name=aks_service_name,
                                               image=image,
                                               deployment_config=aks_config,
                                               deployment_target=aks_target)
    aks_service.wait_for_deployment(show_output=True)
    print(aks_service.state)
else:
    aks_service = ws.webservices[aks_service_name]
```

## Download Inference Data

In this step, we download additional weather data from 2016 to send to the deployed service for testing. 

```python

# Score Model on March 15, 2016 data
scoring_df = get_noaa_data(datetime(2016, 3, 15) - timedelta(days=7), datetime(2016, 3, 16),  columns, usaf_list)
# Add the window feature column
scoring_df = add_window_col(scoring_df)

# Drop features not used by the model
print("Dropping unnecessary columns")
scoring_df = scoring_df.drop(['windAngle', 'windSpeed', 'datetime', 'elevation'], axis=1).dropna()
scoring_df.head()


# One Hot Encode the scoring dataset to match the training dataset schema
columns_dict = model.datasets["training"][0].get_profile().columns
extra_cols = ('Path', 'Column1')
for k in extra_cols:
    columns_dict.pop(k, None)
training_columns = list(columns_dict.keys())

categorical_columns = scoring_df.dtypes == object
categorical_columns = categorical_columns[categorical_columns == True]

test_df = pd.get_dummies(scoring_df[categorical_columns.keys().tolist()])
encoded_df = scoring_df.join(test_df)

# Populate missing OHE columns with 0 values to match traning dataset schema
difference = list(set(training_columns) - set(encoded_df.columns.tolist()))
for col in difference:
    encoded_df[col] = 0
encoded_df.head()


# Serialize dataframe to list of row dictionaries
encoded_dict = encoded_df.to_dict('records')
```

## Send Inference Data to Model

In this step, we send the data to the service. It will be collected by the Model Data Collector and avaialble for data drift. 

```python

# retreive the API keys. AML generates two keys.
key1, key2 = aks_service.get_keys()

total_count = len(scoring_df)
i = 0
load = []
for row in encoded_dict:
    load.append(row)
    i = i + 1
    if i % 100 == 0:
        payload = json.dumps({"data": load})
        
        # construct raw HTTP request and send to the service
        payload_binary = bytes(payload,encoding = 'utf8')
        headers = {'Content-Type':'application/json', 'Authorization': 'Bearer ' + key1}
        resp = requests.post(aks_service.scoring_uri, payload_binary, headers=headers)
        
        print("prediction:", resp.content, "Progress: {}/{}".format(i, total_count))   

        load = []
        time.sleep(3)
```

## Configure DataDrift 

In this step, we configure the DataDrift Object. 

> [!Note]
> This service is in (Preview) and limited in configuration options. Please see our [API Documentation](https://docs.microsoft.com/en-us/python/api/azureml-contrib-datadrift/?view=azure-ml-py) and [Release Notes](azure-machine-learning-release-notes) for details and updates. 

```python
# list of services to detect data drift on for a version of a model
services = [service_name]

# whitelist of features to analyze drift
# NOTE: you should exclude features like time or auto-incrementing indices as drift will be detected on these features
feature_list = ['usaf', 'wban', 'latitude', 'longitude', 'station_name', 'p_k',  'sine_hourofday', 'cosine_hourofday', 'temperature-7']

# if email address is specified, setup AlertConfiguration
alert_config = AlertConfiguration([email_address]) if email_address else None

# there will be an exception indicating using get() method if DataDrift object already exist
try:
    datadrift = DataDriftDetector.create(ws, model.name, model.version, services, frequency="Day", alert_config=alert_config)
except KeyError:
    datadrift = DataDriftDetector.get(ws, model.name, model.version)
    
print("Details of DataDrift Object:\n{}".format(datadrift))
```

## Run an AdHoc DataDriftDetector Run

In this step, we run an AdHoc run to detect DataDrift. We will run this on today's data - in reality, 2016 weather data - which was sent to the service and collected above.

```python

# adhoc run today
target_date = datetime.today()
run = datadrift.run(target_date, services, feature_list=feature_list, create_compute_target=True)

# show details of the data drift run
exp = Experiment(ws, datadrift._id)
dd_run = Run(experiment=exp, run_id=run)
RunDetails(dd_run).show()
```



## Get Drift Analysis Results

In this step, we will wait for all data drift runs to complete and then print out and plot relevant data drift metrics. 

```python

# specify date range getting results

start = datetime.now() - timedelta(days=2)
end = datetime(year=2020, month=1, day=22, hour=15, minute=16)

children = list(dd_run.get_children())
for child in children:
    child.wait_for_completion()

drift_metrics = datadrift.get_output(start_time=start, end_time=end)
drift_metrics

# Show all drift figures, one per serivice.
# If setting with_details is False (by default), only drift will be shown; if it's True, all details will be shown.

drift_figures = datadrift.show(with_details=True)
```

## Enable DataDrift Schedule 

In this step, we enable the data drift schedule - it will run with the settings it is configured with above.

```python
datadrift.enable_schedule()
```

Disabling is similar.

```python
datadrift.disable_schedule()
```

# View Results in Azure ML Workspace UI

To view results in the Azure ML Workspace UI, navigate to the model page. On the details tab of the model, the data drift configuration is shown. A 'Data Drift (Preview)' tab is now available showing the data drift metrics. 

# Example notebook

The [how-to-use-azureml/data-drift/azure-ml-datadrift.ipynb](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/data-drift/azure-ml-datadrift.ipynb) notebook demonstrates concepts in this article. 

[!INCLUDE [aml-clone-in-azure-notebook](../../../includes/aml-clone-for-examples.md)]
