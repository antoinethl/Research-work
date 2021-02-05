from _1_data_processing.data_processing import create_dataframe

#from _3_time_step_selection.time_step_selection import check_timescale
import app

import os
print (os.getcwd())

if __name__ == '__main__':
    app.run()


x = create_dataframe("05")
print(x)

#check_timescale(x, kind="linear", graph=True, error=True)

