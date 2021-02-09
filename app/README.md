# Application

Here is our method contained in the Jupyter Notebook formatted as a Python application.
The data files must be placed in the data folder respecting the structure of the ENERTALK dataset.
Source folder contains every functions used grouped into categories.


## Initialize a time series

There is 3 way to initialize a time series

- complete initialization
- initialization using a past configuration
- initialization using a json object


### Complete initialization

This initialization allows to start a prediction from zero i.e. the optimal time step, the optimal number of clusters... aren't known.

```python
device = "05"
ts = TimeSeries(device_code=device)
```


### Initialization using a past configuration

Loads a past configuration saved in the /source/\_0\_time_series\_class/configuration/ folder. This allows to save time by not doing all the calculations to determine the parameters of the time series and models.

```python
filename = "conf_05_2021-02-02.txt"
ts = TimeSeries(load_fileconf=filename)
```


### Initialization using a json object

Uses a json object to get a personnalized configuration for the time series. If fields are missing, the corresponding parameters will be computed.

```python
conf = {
    'device': '03',
    'time_step': 2,
    'w_sizes': {
        'CNN_LSTM': 15, 
        'CNN': 25, 
        'LSTM': 10, 
        'MLP': 25
    },
    'nb_clust': 3
}

t = TimeSeries(load_dictconf=conf)
```

## Run the application

First, in the root folder, requirements can be installed in a new conda environnement using:

	$ conda create --name <env> --file requirements.txt
	$ conda activate <env>

The best way to use the application is to open and run app.py in an IDE like spyder.
However, the application can be launched in the shell using:

	$ python -m app

