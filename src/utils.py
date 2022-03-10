import numpy as np 
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc

def validate_input_dataframe(df):
    assert df.columns[0] == 'Label' and df.columns[1] == 'Time', "Input data has invalid format"

def validate_tasklist(df):
    assert df.columns[0] == 'Status', "Tasklist has invalid format"

def append_to_report(state, lines):

    for line in lines:
        print(line + "\n")

    if "report" not in state:
        return 

    state["report_lock"].acquire() 

    with open(state["report"], "a") as f:
        for line in lines:
            f.write(line + "\n")

    f.close() 
    state["report_lock"].release() 

def vdp(x,t=0):
    dx = np.asarray([x[1],
                     (1-x[0]**2)*x[1]-x[0]])
    return dx

# Credit: https://github.com/cagatayyildiz/npde/blob/master/utils.py 
def em_int(f,g,x0,t):
    """ Euler-Maruyama integration
    
    """
    ts = np.linspace(0,np.max(t),(len(t)-1)*5)
    ts = np.unique(np.sort(np.hstack((ts,t))))
    idx = np.where( np.isin(ts,t) )[0]
    ts = np.reshape(ts,[-1,1])
    dt = ts[1:] - ts[:-1]
    T = len(ts)
    D = len(x0)
    Xs = np.zeros((T,D),dtype=np.float64)
    Xs[0,:] = x0
    for i in range(0,T-1):
        fdt = f(Xs[i,:],ts[i])*dt[i]
        gdt = g(Xs[i,:],ts[i])*np.sqrt(dt[i])
        Xs[i+1,:] = Xs[i,:] + fdt + gdt.flatten()
    X = Xs[idx,:]
    return X

def gen_data(model='vdp',Ny=[30],tend=8,x0=np.asarray([2.0,-3.0]),nstd=0.1):
    x0in = x0
    Nt = len(Ny)
    x0 = np.zeros((Nt,2))
    t = np.zeros((Nt,), dtype=np.object)
    Y = np.zeros((Nt,), dtype=np.object)

    mean_ = np.array([-2,1])
    var_ = np.eye(2)*0.5
    if model == 'vdp':
        gtrue = lambda x,t : g(x,t,mean_,var_,0,0)
    elif model == 'vdp-cdiff':
        gtrue = lambda x,t : g(x,t,mean_,var_,0,0.2)
    elif model == 'vdp-sdiff':
        gtrue = lambda x,t : g(x,t,mean_,var_,3.0,0.0)
    else:
        raise NotImplementedError('Only stochastic/deterministic Van der Pol supported')
    diff = lambda x,t: ss.norm.rvs(size=[2,1]) * gtrue(x,t)
    for i in range(Nt):
        tspan = np.linspace(0,tend,Ny[i])
        t[i] = tspan
        X = em_int(vdp, diff, x0in, tspan)
        Y[i] = X + ss.norm.rvs(size=X.shape) * nstd
        x0[i,:] = Y[i][0,:]

    return x0,t,Y,X,2,vdp,gtrue



            
