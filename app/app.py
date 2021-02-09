from source._0_time_series_class.time_series_class import TimeSeries

def main(full=False):
        
    if full:
        
        device = "05" 
        
        ts1 = TimeSeries(device) # Finds the optimal time step and read the files corresponding to the device number
        ts1.discretize() # Finds the optimal number of clusters and discretize the time series
        ts1.get_models_size() # Finds the best windows sizes for each model used in the hybrid prediction
        

    else:
        
        # Loads a text file with containing every informations that have 
        # already been computed for a given time series (nb of clusters, windows sizes...)
        
        ts1 = TimeSeries(load_fileconf="conf_05_2021-02-02.txt") 
            
        
    ts1.get_models()
    ts1.compute_error()
    ts1.compute_weights_init() 
    ts1.plot_weights() 
    ts1.plot_predictions()
    print(ts1.error.mean())
    

if __name__ == "__main__":
    main()
