from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.metrics import MeanAbsolutePercentageError


def MAPE(y_true, y_pred):
    
    """
        Computes the MAPE between 2 time series
        
        @Params:
        y_true: array of float
            The original time series
        y_pred: array of float
            The predicted, interpolated or discretized time series
            
        @Returns:
        float
            The MAPE between the 2 time series
    """
    
    m = MeanAbsolutePercentageError()
    m.update_state(y_true, y_pred)
    
    return m.result().numpy()


def print_error(y_true, y_pred):
    
    """
        Prints the MSE, the MAE and the MAPE between 2 time series
        
        @Params:
        y_true: array of float
            The original time series
        y_pred: array of float
            The predicted, interpolated or discretized time series
    """
     
    print("\tMAPE: " + str(MAPE(y_true, y_pred)))
    print("\tMSE: " + str(mean_squared_error(y_true, y_pred)))
    print("\tMAE: " + str(mean_absolute_error(y_true, y_pred)))


def mae(x):
    
    """
        Computes and returns the MAE between a prediction window and the original window
            
        @Params:
        x: object
            Slice of a DataFrame with 2 columns, the one containing every predicted windows and the other with
            the original windows
            
        @Returns:
        float
            The MAE between the 2 value windows
    """

    return mean_absolute_error(x[0], x[1])

