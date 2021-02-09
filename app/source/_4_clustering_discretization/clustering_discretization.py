import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

from .._2_error_helping_functions.error_helping_functions import print_error


def discretize_signal(labels, centroids):
    
    """
        Discretizes a time series using the labels computed with a clustering method and the centroids
        
        @Params:  
        labels: array of int
            The labels found with the clustering method
        centroids: array of float
            The centroid for each cluster
            
        @Returns:
        ts: array of float
            The discretized time series
    """
    
    function = lambda x: centroids[x]
    ts = np.array([function(xi) for xi in labels]).reshape(-1)
    
    return ts


def plot_discretized(raw, discretized, t1=0, t2=100, t1p=None, t2p=None):
    
    """
        Plots the discretized time series compared to the raw one
        
        @Params:
        raw: array of float
            The raw time series
        discretized: array of float 
            The discretized time series
        t1: int
            The lower bound index when plotting the time series
        t2: int
            The upper bound index when plotting the time series
        t1p: float between 0 and 1
            The lower bound index in percentage when plotting 
        t2p: float between 0 and 1
            The upper bound of time in percentage when plotting
    """
    
    if t1p and t2p:
        t1 = int(len(raw)*(t1p/100))
        t2 = int(len(raw)*(t2p/100))
    
    plt.plot(raw[t1:t2], color="#731135")
    plt.title("Original time series")
    plt.show()
    
    plt.plot(discretized[t1:t2], color="#5071ad")
    plt.title("Discretized time series")
    plt.show()
    
    plt.plot(raw[t1:t2], label="Original", color="#731135")
    plt.plot(discretized[t1:t2], label="Discretized", color="#5071ad")
    plt.title("Discretization")
    plt.legend()
    plt.show()

    
def clustering(x, n_clusters=3, error=True, plot=True):
    
    """
        Applies K-means clustering method to the time series according to the number of clusters given
        
        @Params:
        df: DataFrame
            The DataFrame containing the time series information
        n_clusters: int
            The number of clusters
        error: bool
            If True, prints the error
        plot: bool
            If True, plots the different graphs
            
        @Returns:
        ts: array of float
            The discretized time series
    """

    X = x.to_numpy().reshape(-1, 1)

    clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    ts = discretize_signal(clustering.labels_, clustering.cluster_centers_)

    if plot: plot_discretized(X, ts, t1=0, t2=200)
    if error: print_error(X, ts)
    
    return ts

