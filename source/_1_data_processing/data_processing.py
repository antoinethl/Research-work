import pandas as pd
import os
from datetime import datetime

def create_dataframe(device_type=None, time=0, nbr_of_files=10000, show=True):
    print(__name__)
    """
        Reads the Excel files and create a pandas DataFrame containing 
        the time series information for a given device
        
        @Params:
        device_type: str
            String containing "0+int" between 0 and 8, which represent the code for the device type to be
            studied, if None, the loop will read every files for a complete dataset
        time: int 
            The timescale to use when reading the files, represents a time step in minutes
        nbr_of_files: int
            The number of files to be read
        show: bool
            If True, will display the informations while reading the files
            
        @Returns:
        df: DataFrame
            The DataFrame containing the informations from files for the given device
    """

    path = "../"

    df = pd.DataFrame()
    i = 1

    for root, directories, filenames in os.walk(path):
        for filename in filenames:
            if (filename.split(".")[-1] == "gzip") and (filename[0:2] == device_type or device_type == None):

                df2 = pd.read_parquet(os.path.join(root, filename)) 
                
                df2["date"] = df2["timestamp"].apply(lambda x: datetime.fromtimestamp(x/1000))
                
                if time != 0:   
                    df2 = df2.set_index('date').resample(f'{time}T').mean().reset_index()
                
                df2["filename"] = i
                df2["device"] = filename
                df = df.append(df2)
                if show: print("File {}, {} | {} rows".format(i, filename, len(df2)))
                i += 1
                
        if i > nbr_of_files:
            break
            
    df = df.drop_duplicates(subset=['date'])
    df = df.reset_index()

    return df

