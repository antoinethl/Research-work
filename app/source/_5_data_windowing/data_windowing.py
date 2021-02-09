import numpy as np


def create_X_Y(ts, f_size=10, offset=0, output_width=1, sum_=False):
    
    """
        From a given time series, creates an input array containing the input windows and a label array containing 
        the output windows for machine/deep learning tasks
        
        @Params:
        ts: array of float
            The time series
        f_size: int
            The input window size
        offset: int
            The offset between the last value of the input window and the first value of the output window
        output_width:
            The output window size
        sum_: bool
            If true, the label to predict will be the sum of the values contained in the output window
            
        @Returns:
        X: array of float
            The input array containing the input windows
        Y: array of float
            The label array containing the output windows
    """

    X, Y = [], []

    for i in range(len(ts) - f_size - offset - output_width):
        if (output_width == 1):
            Y.append(ts[i + f_size + offset])
        elif not sum_:
            Y.append(ts[i + f_size + offset:i + f_size + offset + output_width])
        else:
            Y.append(np.sum(ts[i + f_size + offset:i + f_size + offset + output_width]))
        X.append(ts[i:(i + f_size)])
    
    X, Y = np.array(X), np.array(Y)
    
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    if sum_: Y = np.reshape(Y, (Y.shape[0], 1))

    return X, Y

