from utils import append_to_report
import numpy as np 

def read_labeled_timeseries(state, reset_time = False, time_unit = 1):
    """
    Load a CSV file that contains timeseries data delineated by labels 
    """

    df = state['df']

    labels = df.iloc[:,0].to_numpy()
    indices =  np.where(np.logical_not(np.equal(labels[1:], labels[:-1])))[0] + 1

    time_column = np.split(df.iloc[:,1].to_numpy(dtype=np.float32) / time_unit, indices)
    data_columns = np.split(df.iloc[:,2:].to_numpy(dtype=np.float32), indices, axis=0) 

    if reset_time:
        time_column = [segment - segment[0] for segment in time_column]

    append_to_report(state, [f"Reset Time: {reset_time}, Time Unit: {time_unit:.2f}"])
    
    state['labeled_timeseries'] = (time_column, data_columns) 


def apply_pca(state, verbose=False):

    if 'df' not in state:
        raise Exception("Input dataframe needs to be in state")

    
    append_to_report(state, [f"--- Applied PCA --- "])