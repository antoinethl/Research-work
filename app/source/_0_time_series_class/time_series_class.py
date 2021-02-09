import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

from .._1_data_processing.data_processing import create_dataframe
from .._2_error_helping_functions.error_helping_functions import MAPE, mae
from .._3_time_step_selection.time_step_selection import find_time_step
from .._4_clustering_discretization.clustering_discretization import clustering, plot_discretized
from .._5_data_windowing.data_windowing import create_X_Y
from .._6_deep_learning_models.deep_learning_models import get_model

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import rpy2.rinterface


rpy2.robjects.numpy2ri.activate()

base = importr('base')
Factoextra = importr("factoextra")


class TimeSeries():
    
    """
        A class representing the time series and every information needed to work with it
        Contains functions to discretize it, finds the optimal number of clusters, the optimal 
        time step and the models window size
        Contains the models used for the hybrid prediction, functions to calculate and adjust
        the weights that correspond to the importance of the models for the prediction of the final values
        
        @Attributes:
        conf: dict
            The configuration of the time series where every useful information is stored 
            like the optimal number of cluster, the time step, the device code, 
            the optimal window size for each model...
        df: DataFrame
            The DataFrame obtained after reading the data files
        ts: array of float
            The raw time series
        ts_dis: array of float
            The discretized time series
        trainX: array of float
            The training inputs
        trainY: array of float
            The training labels/output
        testX: array of float
            The testing inputs
        testY:
            The testing labels/output
        models: dict
            Dictionary containing the models used for predicting the time series, keys are the models names
        predict: DataFrame
            DataFrame that contains every prediction for each model and for each window in the test set
            Contains the final prediction computed with the weights
        error: DataFrame
            DataFrame that contains the MAE computed between the prediction of each model and the original test set
        weights: array of float
            The weights of the models that correspond to their importance for the prediction of the final values 
        last_values: DataFrame
            Last values used by the weights calculation function for the standard deviation 
    """
    
    
    def __init__(self, device_code=None, time_step=None, load_fileconf=None, load_dictconf=None, 
                 nbr_of_files=1000, threshold=15):
        
        """
            @Params:
            device_code: str
                The device code for the appliance to study, 
                useless if arguments load_fileconf or load_dictconf aren't None
            time_step: int
                The time step to use when reading the files, represents a time step in minutes, 
                if left None, the time step will be computed automatically
            load_fileconf: str
                The filename to load a configuration from a json stored locally in a text file
            load_dictconf: dict
                Load a configuration from a dictionary
            nbr_of_files: int
                The number of files to read
            threshold: int
                Threshold representing the limit while computing the MAPE 
                between raw time series and scaled time series
        """
        
        if load_fileconf:
            self.load_conf(load_fileconf)
          
        elif load_dictconf:
            self.conf = load_dictconf
            
        else:   
            self.conf = {
                "device": device_code
            }
        
            if time_step:
                self.conf["time_step"] = time_step  
                
            else:
                self.conf["time_step"] = find_time_step(device_code, threshold=threshold)
            
        self.df = create_dataframe(self.conf["device"], self.conf["time_step"], nbr_of_files=nbr_of_files)
        self.df = self.df.dropna()
        self.ts = self.df["active_power"]
        
    
    def set_nb_clusters(self):
        
        """
            Finds the optimal number of clusters needed to discretize the time series
            Uses the fviz_nbclust() from Factoextra in an embedded R code to compute it
        """
        
        print("Finding the optimal number of clusters...")
 
        sample = ro.r.matrix(self.df[self.df["filename"].between(1, 4)]["active_power"].to_numpy())
                
        r=ro.r("""
            check = function(matrix) {
            n_clust = fviz_nbclust(matrix, kmeans, k.max = 15)

            n_clust = n_clust$data

            max_cluster = as.numeric(n_clust$clusters[which.max(n_clust$y)])
            return(max_cluster)
            }
        """)

        result = r(sample)
        self.conf["nb_clust"] = int(result[0])
        
        print(f"Optimal number of clusters is {self.conf['nb_clust']}\n")
                    
            
    def discretize(self, threshold=100):
        
        """
            Discretizes the time series using the number of clusters in the configuration dictionary
            Computes the mape between the original time series and the discretized one
            If the mape is too high, the orginal time series will be used instead of the discretized one 
            
            @Params:
            threshold: int
                The maximum threshold not to be exceeded for the discretization

        """
        
        print("Starting discretization...")
        
        if not "nb_clust" in self.conf.keys():
            print("No number of clusters assigned\n")
            self.set_nb_clusters()
        
        self.ts_dis = clustering(self.ts, self.conf["nb_clust"])
        
        error = MAPE(self.ts, self.ts_dis)
        print(error)
        
        if error > threshold:
            self.ts_dis = self.ts
            print("\nError is too high, original time series will be used instead of disctretized one\n")
            
        print("Discretization done\n")
        
         
    def plot_series(self, t1=0, t2=100, t1p=None, t2p=None):
        
        """
            Plots the raw time series and the discretized one
            
            @Params:
            t1: int
                The lower bound index when plotting the time series
            t2: int
                The upper bound index when plotting the time series
            t1p: float between 0 and 1
                The lower bound index in percentage when plotting 
            t2p: float between 0 and 1
                The upper bound of time in percentage when plotting
        """
        
        plot_discretized(self.ts, self.ts_dis, t1=t1, t2=t2, t1p=t1p, t2p=t2p)
        
        
    def find_w_size(self, name, output_width=30, min_w=10, max_w=75, step=5, offset=0, 
                    sum_=False, window=None):
        
        """
            Finds the best input window size for a given model using an interval of values
            Computes the loss over windows size and keeps the size with the minimum error on the validation set
            
            @Params:
            name: str
                The name of the model used for the prediction, allowed names are 'CNN_LSTM', 'CNN', 'LSTM' and 'MLP'
            output_width: int
                The output window size
            min_w: int
                The minimum value of the interval to find the best window size
            max_w: int
                The maximum value of the interval to find the best window size
            step: int
                The step of values inside the interval
            offset: int
                The offset between the last value of the input window and the first value of the output window
            sum_: bool
                If true, the output will be the sum over the next 30 minutes
            window: int
                Optional input window size, if used, the function won't find the best window size and will use 
                this value instead
                
            @Returns:
            window_size: int
                The optimal window size for the given model
        """
        
        loss_arr = []
        val_loss_arr = []
        
        print("Finding best window size..")
        print(f"Model: {name}, output size: {output_width}\n")
        
        if window:
            
            self.create_train_test(name=name, f_size=window, offset=offset, output_width=output_width, sum_=sum_)
            model, loss, val_loss = get_model(name, self.trainX, self.trainY)
        
        else:
            
            for i in range(min_w, max_w, step):
            
                self.create_train_test(name=name, f_size=i, offset=offset, output_width=output_width, sum_=sum_)
                model, loss, val_loss = get_model(name, self.trainX, self.trainY)
                
                print(f"For window of {i} values, MAPE = {loss}")
                loss_arr.append(loss)
                val_loss_arr.append(val_loss)
                
                temp = np.insert(val_loss_arr, 0, val_loss_arr[0])
                temp = np.append(temp, val_loss_arr[-1])
            
                smooth = np.convolve(temp, [1, 2, 1], mode='valid')
     
                if (len(smooth)-np.argmin(smooth)) > 4:
                    break
                
            print("Done")
            
            val_loss_arr = np.insert(val_loss_arr, 0, val_loss_arr[0])
            val_loss_arr = np.append(val_loss_arr, val_loss_arr[-1])
            val_loss_arr_smooth = np.convolve(val_loss_arr, [1, 2, 1], mode='valid')     
            
            idx = np.argmin(val_loss_arr_smooth)
            
            window_size = range(min_w, max_w, step)[idx]
            
            range_ = range(min_w, max_w, step)[:len(loss_arr)]
            plt.plot(range_, loss_arr, label="loss", color="#33638DFF")
            plt.plot(range_, val_loss_arr[1:-1], label="val_loss", color="#3CBB75FF")
            plt.plot(range_, val_loss_arr_smooth/4, 
                     label="smooth_val_loss", color="#d18756")
            
            plt.axvline(x=window_size, linestyle="--", c="black", lw=1)
            plt.legend()
            plt.title(name + " model")
            plt.xlabel("window size")
            plt.ylabel("loss")
            plt.show()
            
            print(f"Best window size for {name} is {window_size}\n")

            return window_size
    
    
    def create_train_test(self, name, f_size=10, offset=0, output_width=1, train_size=0.8, sum_=False):
        
        """
            Creates a train and a test set for the deep learning models
            
            @Params:
            name: str
                The name of the model used for the prediction, allowed names are 'CNN_LSTM', 'CNN', 'LSTM' and 'MLP'
            f_size: int
                The input window size
            offset: int
                The offset between the last value of the input window and the first value of the output window
            output_width:
                The output window size
            train_size: float between 0 and 1
                The proportion of the dataset to include in the train set
            sum_: bool
                If true, the output will be the sum over the next 30 minutes   
        """
        
        if not hasattr(self, 'ts_dis'):
            print("Time series isn't discretized\n")
            self.discretize()
        
        n_train = int(train_size * len(self.ts_dis))

        if "CNN" in name:
            train = self.ts[:n_train]
            test = self.ts[n_train:]
        
        else:
            train = self.ts_dis[:n_train]
            test = self.ts_dis[n_train:]
            
        self.trainX, self.trainY = create_X_Y(train, f_size, offset, output_width, sum_)
        self.testX, self.testY = create_X_Y(test, f_size, offset, output_width, sum_)
        
        
    def get_models_size(self, models=["CNN_LSTM", "CNN", "LSTM", "MLP"], size=None):
        
        """
            Finds the optimal input window size for each model, stores the results in the conf dictionary
            
            @Params:
            models: array of string
                The names of the models used for the prediction, allowed names are 'CNN_LSTM', 'CNN', 'LSTM' and 'MLP'
            size: dict
                Optional dictionary containing the size for each model, keys correspond to the models names,
                if left None, the algorithm will find the optimal window size
        """
                
        self.conf["w_sizes"] = {}
        
        for model in models:
            
            print(f"Searching optimal window size for {model}:\n")
            
            if size:
                self.conf["w_sizes"][model] = size[model]
                
            else:
                self.conf["w_sizes"][model] = self.find_w_size(model, output_width=int(30/self.conf["time_step"]))
              
            
    def get_models(self, offset=0, sum_=False):
        
        """
            Creates and stores the models that will be used for predicting the time series
            according to their kind and their input window stored in the configuration dictionary
            Stores the models in a new dictionary
            For each model and for each predicted window, store the prediction in a new DataFrame named "predict"
            
            @Params:
            offset: int
                The offset between the last value of the input window and the first value of the output window
            sum_: bool
                If true, the output will be the sum over the next 30 minutes
        """
            
        self.models = {}
        self.predict = pd.DataFrame()
        min_value = min(self.conf["w_sizes"].values())
           
        output_width = int(30/self.conf["time_step"])
        
        
        for name in self.conf["w_sizes"].keys():
                
            size = self.conf["w_sizes"][name]
            self.create_train_test(name=name, f_size=size, offset=offset, output_width=output_width, sum_=sum_)
            model, loss, val_loss = get_model(name, self.trainX, self.trainY)
            
            pred = pd.DataFrame({name: model.predict(self.testX).tolist()},
                                index=range(size-min_value, len(self.testY)+(size-min_value)))
            
            pred[name] = pred[name].apply(lambda x: np.array(x))
            
            self.predict = pd.concat([self.predict, pred], axis=1)
                
            self.models[name] = model
            
            del model, pred
            
        self.create_train_test(name="CNN", f_size=min_value, offset=offset, output_width=output_width, sum_=sum_)
        self.predict["test"] = self.testY.tolist()
        self.create_train_test(name="MLP", f_size=min_value, offset=offset, output_width=output_width, sum_=sum_)
        self.predict["test_dis"] = self.testY.tolist()
        
        self.predict.dropna(inplace=True)
        
    
    def compute_error(self):
        
        """
            Computes the error between each model prediction for each window in the test set, stores the result
            in a new DataFrame. After this, creates a new column in the "predict" DataFrame with all errors in
            a numpy array
        """
        
        self.error = pd.DataFrame()
        
        for name in self.conf["w_sizes"].keys():
            
            self.error[f"mae {name}"] = self.predict[[name, "test"]].apply(lambda x: mae(x), axis=1)
            self.error[f"mape {name}"] = self.predict[[name, "test"]].apply(lambda x: MAPE(x[0], x[1]), axis=1)
            
        self.predict['error'] = self.error.filter(like='mae').apply(lambda r: tuple(r), axis=1).apply(np.array)
        
        
    def compute_weights_init(self, size_max=30, learning_rate=0.1):
        
        """
            Initializes the weights calculation and applies the function "compute_weights" 
            for each prediction in the "predict" DataFrame
            Stores the result in a new column in the "predict" DataFrame
            Computes the final time series with the model weights and their predictions
            Finally, computes the final error
            
            @Params:
            size_max: int
                The maximum size of the array containing the last errors to calculate the standard deviation
            learning_rate: float
                The value by which the new weights will be multiplied, it determines how
                newly acquired information will override old information, a value close to 1 
                will only consider new information while 0 will prevent it to learn
        
        """
        
        self.weights = [0] * len(self.models)
        self.last_values = pd.DataFrame()
        
        self.predict["weights"] = self.predict['error'].apply(lambda x: self.compute_weights(x))
        
        self.predict["final"] = self.predict[[*self.models.keys()]].apply(lambda x: self.compute_final_values(x), axis=1)
        
        self.error["mae final"] = self.predict[["final", "test"]].apply(lambda x: mae(x), axis=1)
        self.error["mape final"] = self.predict[["final", "test"]].apply(lambda x: MAPE(x[0], x[1]), axis=1)
        
    
    def compute_weights(self, error, size_max=30, learning_rate=0.1):
        
        """
            Computes the weight that each model will have for the prediction, a model which makes smaller errors and
            has a smaller standard deviation will have more weight for the prediction of the final values
            
            @Params:
            error: array of float
                The model errors used to update the weights
            size_max: int
                The maximum size of the array containing the last errors to calculate the standard deviation
            learning_rate: float
                The value by which the new weights will be multiplied, it determines how
                newly acquired information will override old information, a value close to 1 
                will only consider new information while 0 will prevent it to learn
                
            @Returns:
            weights: array of float
                The new updated weights
        """
        
        self.last_values = self.last_values.append([error], ignore_index=True)
        
        if len(self.last_values) > size_max:
            self.last_values = self.last_values.drop(self.last_values.index[0]).reset_index(drop=True)
    
        nw = error

        nw = [(max(nw)-elem+min(nw)) / sum(nw) for elem in nw]
        nw = [elem / sum(nw) for elem in nw]
    
        if np.array(self.weights).any():
            
            std = self.last_values.std().values
            std = [(max(std)-elem+min(std)) / sum(std) for elem in std]

            weights = np.add(self.weights, [learning_rate*a*b for a,b in zip(nw, std)])
            weights = [round((elem / sum(weights)), 2) for elem in weights]
            self.weights = weights
            
        else:
            self.weights = nw
            
        return np.array(self.weights)
    
    
    def compute_final_values(self, x):
        
        """
            Computes the final values with the model weights and their predictions
            
            @Params:
            x: Slice of DataFrame
                The DataFrame columns that contains the predictions of the models 
                
            @Returns:
            values: array of float
                The final window computed with the model weights and their predictions
        """
            
        values = np.zeros(len(x[0]))
    
        for i in range(len(x)):
            values = values + np.array(x.values[i] * self.weights[i])
        
        return values
    
    
    def plot_predictions(self, names=None, min_=1, max_=1000):
        
        """
            Plots the predicted windows for a given interval, some windows are skipped to avoid overlaps
            1 window displayed for 30/time_step predicted
            
            @Params:
            names: array of string
                If left None, will display every predictions + the original time series, "names" can be used
                to choose the predictions that must be plotted
            min_: int
                The lower bound index when plotting the time series
            max_: int
                The upper bound index when plotting the time series
        """
        
        if not names:
            names = [*self.models.keys()] + ["test", "final"]

        arr = range(min_, max_, int(30/self.conf["time_step"]))

        plt.figure(figsize=(16, 7), dpi=75)

        for name in names:
            plt.plot(np.concatenate(self.predict.iloc[arr][name].to_numpy()), label=name)

        plt.title("Predictions")
        plt.legend()
        plt.show()
    
    
    def plot_weights(self,):
        
        """
            Plots the weights evolution for each model used for the predictions
        """
        
        weights_evolution = pd.DataFrame(self.predict["weights"].values.tolist(), columns=[*self.models.keys()])

        plt.figure(figsize=(8, 5))

        for name in weights_evolution.columns:
            plt.plot(weights_evolution[name], label=name)

        plt.title("Weights evolution")
        plt.legend()
        plt.grid(axis="y", linestyle='--')
        plt.show()

        del weights_evolution
        

    def save_conf(self, name=None):
        
        """
            Saves a configuration as a json into a text file
            
            @Params:
            name: str
                The name that will be given to the file, if left None, the filename will be given according
                to the current date and the device code
        """
        
        if name:
            filename = name
            
        else:
            filename = "conf_" + str(self.conf["device"]) + "_" + datetime.today().strftime('%Y-%m-%d') + ".txt"
            
        path = "/"
        filename = path + filename
        
        with open(filename, "w") as file:
            json.dump(self.conf, file)
    
            
    def load_conf(self, filename):
        
        """
            Loads a configuration from a json stored in a text file
            
            @Params:
            filename: str
                The filename to load the configuration
        """

        path = "./source/_0_time_series_class/configuration/"
        filename = path + filename
        
        with open(filename) as file:
            self.conf = json.loads(file.read())

