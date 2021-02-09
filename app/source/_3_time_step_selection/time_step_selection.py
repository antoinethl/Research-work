import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from .._1_data_processing.data_processing import create_dataframe
from .._2_error_helping_functions.error_helping_functions import MAPE, print_error 


def check_timescale(df, kind="linear", graph=True, error=True):
    
    """
        Checks if the right timescale has been selected, computes the MAPE between the raw time series and
        a time series resampled at a given time step, interpolated to have the same length than the raw one
        
        @Params:
        df: DataFrame
            The DataFrame containing the time series information
        kind: str
            Kind of interpolation, default is linear but can be cubic..      
        graph: bool
            If True, plots the different graphs
        error: bool
            If True, prints the error
            
        @Returns
        float
            The MAPE between the raw time series and the interpolated one
    """
    
    raw = create_dataframe(device_type=df['device'].iloc[0][0:2], time=1, nbr_of_files=1, show=False)
    
    x = df[df["filename"].between(1, 4)].index
    y = df.iloc[x]["active_power"].to_numpy()
    
    step = len(raw) / len(x)
    x = (x*step).astype(int)
    
    f = interp1d(x, y, kind=kind)
    xnew = range(max(x))

    new = f(xnew)
    
    if graph:
        plt.scatter(x=range(0, 240, 2), y=y[:120], s=8, color="#4e3185")
        plt.title("Re-sampled time series")
        plt.show()

        plt.scatter(x=range(0, 240, 2), y=y[:120], s=8, color="#4e3185")
        plt.plot(new[:240], color="#5071ad", label="Interpolated")
        plt.legend()
        plt.title("Interpolated time series")
        plt.show()

        plt.plot(raw["active_power"][:240], color="#731135", label="Original")
        plt.scatter(x=range(0, 240, 2), y=y[:120], s=8, color="#4e3185")
        plt.plot(new[:240], color="#5071ad", label="Interpolated")
        plt.legend()
        plt.title("Re-sampled vs original")
        plt.show()

    if error: 
        print_error(raw["active_power"][0:max(x)], new)
    
    return MAPE(raw["active_power"][0:max(x)], new)
    del raw


def find_time_step(device_code, threshold=15):
    
    """
        Checks multiple time steps to find the optimal one, while the error (MAPE) is smaller than the threshold, 
        the time step to resample the time series increases. Time steps used are 1, 2, 5, 10, 15 minutes
        
        @Params:
        device_code: str
            String containing "0+int" between 0 and 8, which represents the code for the device type to be studied
        threshold: int
            The maximum value that the MAPE mustn't cross
            
        @Returns
        time_step: int
            The optimal computed time step
    """
    
    time_step = 1
    
    for i in [1, 2, 5, 10, 15]:
        
        print(f"Checking error with {i} min...")
        df = create_dataframe(device_code, i, 1, show=False)
        error = check_timescale(df, graph=False, error=False)
        print(f"\tMAPE : {error}\n")
        
        if error < threshold:
            time_step = i
        else:
            break
            
    print(f"Optimal time step is {time_step} min\n\n")  
    
    return time_step

