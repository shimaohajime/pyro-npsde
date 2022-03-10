import math
from numpy.core.fromnumeric import shape
from utils import append_to_report
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def read_labeled_timeseries(state, reset_time = False, time_unit = 1, data_dim=None):
    """
    Load a CSV file that contains timeseries data delineated by labels 
    """

    df = state['df']

    labels = df.iloc[:,0].to_numpy()
    indices =  np.where(np.logical_not(np.equal(labels[1:], labels[:-1])))[0] + 1
    if not data_dim:
        data_dim = df.shape[1]-2

    time_column = np.split(df.iloc[:,1].to_numpy(dtype=np.float64) / time_unit, indices)
    data_columns = np.split(df.iloc[:,2:data_dim+2].to_numpy(dtype=np.float64), indices, axis=0) 

    if reset_time:
        time_column = [segment - segment[0] for segment in time_column]

    append_to_report(state, [f"Reset Time: {reset_time}, Time Unit: {time_unit:.2f}"])
    
    state['labeled_timeseries'] = (time_column, data_columns) 

def apply_standardscaling(state):

    if 'df' not in state:
        raise Exception("Input dataframe needs to be in state")

    df = state['df']
    scaler = StandardScaler()
    df.iloc[:,2:] = scaler.fit_transform(df.iloc[:,2:])

    append_to_report(state, [f"Applied standard scaling (normalization) to data entries"])

def apply_pca(state, num_components=2):

    if 'df' not in state:
        raise Exception("Input dataframe needs to be in state")

    df = state['df']
    pca = PCA()
    df.iloc[:,2:] = pca.fit_transform(df.iloc[:,2:])
    df.drop(inplace=True, columns=df.columns[list(range(2 + num_components,df.shape[1]))])
    append_to_report(state, [f"Applied PCA to entries. Total explained variance:{ int(100*(np.sum(pca.explained_variance_ratio_[:num_components])))/100}"])