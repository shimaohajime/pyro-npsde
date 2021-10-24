from multiprocessing import Process, Lock, Pool
import importlib
import multiprocessing
import tensorflow as tf 
from src.yildiz.npde_helper import build_model, fit_model 
# from src.pyro.npsde_pyro import pyro_npsde_run 
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

    while True:
        
        task = taskmanager.fetch_task_from_tasklist(l, state["tasklist"])

        if not task:
            utils.append_to_report(state, ["No remaining tasks."])
            break
        
        try:
            workers = [] 
            workers_queue = multiprocessing.Queue

            preprocessing.apply_standardscaling(state)
            preprocessing.apply_pca(state)
            preprocessing.read_labeled_timeseries(state, reset_time = True, time_unit = state["metadata"]["time_unit"] if "time_unit" in state["metadata"] else 1)
            

            # Split trajectories into train/test set
            kf = KFold(n_splits = n_splits, shuffle=True)

            (time_column, data_columns) = state['labeled_timeseries'] 
            time_column = np.array(time_column)
            data_columns = np.array(data_columns)

            for train_index, test_index in kf.split(time_column):
                train_set_time, test_set_time = time_column[train_index], time_column[test_index]
                train_set_data, test_set_data = data_columns[train_index], data_columns[test_index]

                worker = Process(target=_cross_validate, args=(l, state, task, (train_set_time, train_set_data), (test_set_time, test_set_data), workers_queue))
                worker.start() 
                workers += [worker]
            
            for worker in workers:
                worker.join() 

            cv_results = []
            while True:
                try:
                    cv_results += [workers_queue.get()]
                except:
                    break 
            
            cv = np.mean(cv_results)
            utils.append_to_report(state, [f"Cross-validation completed: {cv}"])
            task.update(columns={"cv_score" : cv})
        
        except Exception as e:
            task.update(to=taskmanager.Task.TaskStatus.ERROR)
            utils.append_to_report(state, [f"Cross-validation error encountered: {str(e)}"])

        
def _cross_validate(lock, state, hyperparams, train_set, test_set, q):
    state = copy(state) # Shallow copy 
    index, hyperparams = hyperparams.index, hyperparams.args 

    build_model_hyperparams = {
        'sf0' : hyperparams['sf0'],
        'sfg0' : hyperparams['sfg0'],
        'ell0' : hyperparams['ell0'],
        'ellg0' : hyperparams['ellg0'],
        'W' : hyperparams['W'],
        'fix_Z' : hyperparams['fix_Z'],
        's' : hyperparams['s'],
    }

    fit_model_hyperparams = {
        'Nw' : hyperparams['Nw'],
        'eta' : hyperparams['eta'],
        'dec_step' : hyperparams['dec_step'],
        'dec_rate' : hyperparams['dec_rate'],
        'num_iter' : hyperparams['num_iter'],
        "plot_" : False
    }

    utils.append_to_report(state, [f"[{datetime.datetime.now()}, pid {os.getpid()}] Cross-validation child process started for Task {index}"])

    # Run simulation
    train_set_time, train_set_data = train_set 
    test_set_time, test_set_data = test_set

    with tf.Session() as sess:
        model = build_model(sess, train_set_time, train_set_data, **build_model_hyperparams)
        model = fit_model(sess, model, test_set_time, test_set_data, **fit_model_hyperparams)

        def cross_validation(npde, t_val, Y_val):
            # Set s to be as large as possible for validation purpose
            # npde.integrator.s = 100
            
            Nt_val = len(Y_val)

            if Nt_val == 0:
                return "NA"

            D = 2
            x0_val = np.zeros((Nt_val,D))
            Ys_val = np.zeros((0,D))
            for i in range(Nt_val):
                x0_val[i,:] = Y_val[i][0,:]
                Ys_val = np.vstack((Ys_val,Y_val[i]))

            with tf.name_scope("cost_val"):
                Xs_val = npde.forward(Nw=50, x0=x0_val,ts=t_val)
                ll_val = 0
                for i in range(len(Y_val)):
                    mvn_val = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=Y_val[i],covariance_matrix=tf.diag(npde.sn))
                    ll_val_i = tf.stack([mvn_val.log_prob(Xs_val[i][j,:,:]) for j in range(Xs_val[i].shape[0])]) # Nw x D
                    ll_val_i = tf.reduce_sum(tf.log(tf.reduce_mean(tf.exp(ll_val_i),axis=0)))
                    ll_val += ll_val_i
                sde_prior = npde.build_prior()
                cost_val = -ll_val

            val_cost = cost_val.eval()
            return val_cost

        # Run cross-validation
        q.put(cross_validation(model, test_set_time, test_set_data))


def begin_simulation(state_init, input_path, tasklist_path, report_path = None):

    state = copy(state_init)
    df = pd.read_csv(input_path)
    tl = pd.read_csv(tasklist_path)


    utils.validate_input_dataframe(df)
    utils.validate_tasklist(tl)


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

    tf.get_logger().setLevel('INFO')
    parser = argparse.ArgumentParser('Macros for Seshat SDE inference research project')
    parser.add_argument("subroutine", type=str, help="[cross-validate]")
    parser.add_argument("algorithm", type=str, help="[yildiz/pyro]")
    parser.add_argument("data", type=str, help="Path to input data")
    parser.add_argument("--report", type=str, help="Path to report output")
    parser.add_argument("metadata", type=str, help="Path to metadata that describes the data")
    parser.add_argument("tasklist", type=str, help="Path to tasklist")
    parser.add_argument("n_process", type=int, help="Number of child processes to spawn")


    args = vars(parser.parse_args()) 

    # args = {
    #     "algorithm" : "yildiz",
    #     "report" : "task1.out", 
    #     "tasklist" : "tasks/yildiz_seshat_task1.csv",
    #     "data" : "data/seshat_old_formatted.csv",
    #     "metadata" : "data/seshat_old_metadata.json",
    #     "n_process" : 5
    # }

    metf = open(args['metadata'], "r")
    metadata = json.load(metf)
    metf.close() 

    state = {
        "algorithm" : args['algorithm'], 
        "metadata" : metadata,
    }

    state = begin_simulation(state, args['data'], args['tasklist'], report_path = args['report'])

    if args['subroutine'] == "cross-validate":
        macro_cross_validation(state)
    

        


