from multiprocessing import Process, Lock, Pool
import importlib
from src.yildiz.npde_helper import build_model, fit_model 
from src.pyro.npsde_pyro import pyro_npsde_run 
import pandas as pd 
import numpy as np 
import datetime
import argparse 
import json
import os
import time 
from copy import copy 
import utils
import taskmanager
import preprocessing
from sklearn.model_selection import KFold

            


def macro_cross_validation(state, n_splits = 5):

    l = Lock() 
    hyperparams = fetch_task_from_tasklist(l, state["tasklist"]) 
    kf = KFold(n_splits = n_splits, shuffle=True)

    # apply_standardscaling(state)
    # apply_pca(state)

    # Split trajectories into train/test set
    read_labeled_timeseries(state, reset_time = True, time_unit = state["metadata"]["time_unit"] if "time_unit" in state["metadata"] else 1)
    workers = [] 

    for train_index, test_index in kf.split(state["labeled_timeseries"]):
        train_set, test_set = state["labeled_timeseries"][train_index], state["labeled_timeseries"][test_index]
        workers += [Process(target=_cross_validate, args=(l, state, train_set, test_set,))]
    
    for worker in workers:
        worker.join() 


    def _cross_validate(state, train_set, test_set):
        state = copy(state) # Shallow copy 
        append_to_report(state, [f"[{datetime.datetime.now()}, pid {os.getpid()}] Cross-validation child process started"])

def _subprocess(state):
    print(f"[{datetime.datetime.now()}, pid {os.getpid()}] Child process started")

    state = copy(state) # Shallow copy 
    task = fetch_task_from_tasklist(state.tasklist_lock, state["tasklist"]) 
    append_to_report(state, [f"[{datetime.datetime.now()}, pid {os.getpid()}] Child process started"])
    time.sleep(np.random.randint(3,7))
    task.update_status(Task.TaskStatus.COMPLETED)
    append_to_report(state, [f"[{datetime.datetime.now()}, pid {os.getpid()}] Child process ended"])

def macro_parallel(state, n_processes = 5):
    read_labeled_timeseries(state, reset_time = True, time_unit = state["metadata"]["time_unit"] if "time_unit" in state["metadata"] else 1)
    workers = [] 

    for i in range(n_processes):
        worker = Process(target=_subprocess, args=(state))
        worker.start()
        workers += [worker]

    for worker in workers:
        worker.join() 


def begin_simulation(state_init, input_path, tasklist_path, report_path = None):

    state = copy(state_init)
    df = pd.read_csv(input_path)
    tl = pd.read_csv(tasklist_path)


    validate_input_dataframe(df)
    validate_tasklist(tl)


    state['df'] = df 
    state['tasklist'] = tasklist_path 
    state['tasklist_lock'] = Lock()

    if report_path:
        f = open(report_path, 'w')
        f.write(f"Simulation began at {datetime.datetime.now()}\n")
        f.close()
        state['report'] = report_path 
        state['report_lock'] = Lock() 

    return state 



if __name__ == "__main__":

    parser = argparse.ArgumentParser('Macros for Seshat SDE inference research project')
    parser.add_argument("subroutine", type=str, help="[cross-validate]")
    parser.add_argument("algorithm", type=str, help="[yildiz/pyro]")
    parser.add_argument("data", type=str, help="Path to input data")
    parser.add_argument("--report", type=str, help="Path to report output")
    parser.add_argument("metadata", type=str, help="Path to metadata that describes the data")
    parser.add_argument("tasklist", type=str, help="Path to tasklist")
    parser.add_argument("n_process", type=int, help="Number of child processes to spawn")


    args = parser.parse_args() 

    metf = open(args.metadata, "r")
    metadata = json.load(metf)
    metf.close() 

    state = {
        "algorithm" : args.algorithm, 
        "metadata" : metadata,
    }

    state = begin_simulation(state, args.data, args.tasklist, report_path = args.report)

    macro_parallel(state, args.n_process)
    # if args.subroutine == "cross-validate":
    #     macro_cross_validate(state)
    

        


