import pandas as pd
import numpy as np
import pyro
import torch
import pyro.distributions as dist
import torch.distributions.constraints as constraints
from pyro.infer import SVI, Trace_ELBO

class Kernel:
    '''
    Defines the RBF kernel used in Yildiz algorithm.
    '''
    def __init__(self,sf,ell):
        self.sf = sf
        self.ell = ell
    def square_dist(self,X,X2=None):
        X = X / self.ell
        Xs = torch.sum(torch.square(X), dim=1)
        if X2 is None:
            return -2 * torch.mm(X, X.t()) + Xs.reshape([-1,1]) + Xs.reshape([1,-1])
        else:
            X2 = X2 / self.ell
            X2s = torch.sum(torch.square(X2), dim=1)
            return -2 * torch.mm(X, X2.t()) + Xs.reshape([-1,1]) + X2s.reshape([1,-1])
    def RBF(self,X,X2=None):
        if X2 is None:
            return self.sf**2 * torch.exp(-self.square_dist(X) / 2)
        else:
            return self.sf**2 * torch.exp(-self.square_dist(X, X2) / 2)
    def K(self,X,X2=None):
        if X2 is None:
            rbf_term = self.RBF(X)
        else:
            rbf_term = self.RBF(X,X2)
        return rbf_term

class NPSDE():
    '''
    Implementation of Yildiz NPSDE algorithm.
    '''
    def __init__(self,vars,sf_f,sf_g,ell_f,ell_g,Z,fix_sf,fix_ell,fix_Z,delta_t,jitter):

        self.vars = vars #list of variable names to be estimated. (ex) ['PCA0','PCA1']
        self.n_vars = len(vars)

        ##Hyperparameters, either fixed or learned.
        self.sf_f = sf_f if fix_sf else pyro.param('sf_f', sf_f)
        self.sf_g = sf_g if fix_sf else pyro.param('sf_g', sf_g)
        self.ell_f = ell_f if fix_ell else pyro.param('ell_f', ell_f)
        self.ell_g = ell_g if fix_ell else pyro.param('ell_g', ell_g)
        self.Z = Z if fix_Z else pyro.param('Z', Z)
        self.Zg = self.Z #Inducing location for drift and diffusion are set same.

        ##Define kernels
        self.kernel_f = Kernel(self.sf_f, self.ell_f)
        self.kernel_g = Kernel(self.sf_g, self.ell_g)

        self.n_grid = Z.shape[0]
        self.delta_t = delta_t #Euler-Maruyama time discretization.
        self.jitter = jitter

    def compute_f(self, X, U, Z, kernel):
        N = X.shape[0]
        M = Z.shape[0]
        D = Z.shape[1] # dim of state
        Kzz = kernel.K(Z) + torch.eye(M) * self.jitter
        Kzx = kernel.K(Z, X)
        Lz = torch.cholesky(Kzz)
        A = torch.triangular_solve(Kzx, Lz, upper=False)[0]
        #Note U is whitened. Visualization requires unwhitening.
        f = torch.mm(A.t(), U)
        return f

    def compute_g(self, X, Ug, Zg, kernel):
        N = X.shape[0]
        M = Zg.shape[0]
        D = Zg.shape[1] # dim of state
        Kzz = kernel.K(Zg) + torch.eye(M) * self.jitter
        Kzx = kernel.K(Zg, X)
        Lz = torch.cholesky(Kzz)
        A = torch.triangular_solve(Kzx, Lz, upper=False)[0]
        #Note Ug is whitened. Visualization requires unwhitening.
        g = torch.mm(A.t(), Ug)
        return torch.abs(g) #Since we are not generating Euler-Maruyama explicitly by sampling Gaussian noise, g needs to be positive.

    def calc_drift_diffusion(self, X, U, Ug, Z, Zg, kernel_f, kernel_g):
        f = self.compute_f(X, U, Z, kernel_f)
        g = self.compute_g(X, Ug, Zg,  kernel_g)
        return f, g

    def unwhiten_U(self, U_whitened, Z, kernel):
        ##The estimated U and Ug are in whitened space, and requires un-whitening to get the original vectors.
        M = Z.shape[0]
        Kzz = kernel.K(Z) + torch.eye(M) * self.jitter
        Lz = torch.cholesky(Kzz)
        U = torch.mm(Lz,U_whitened)
        return U

    def model(self,df,balanced_df):
        '''
        NPSDE model.
        df is a Pandas DataFrame whose keys include variables in vars (such as ['PC0','PC1']) plus "time".
        balanced_df is the same data as a balanced panel, where all the time steps are filled (potentially with missing values).
        '''
        t_max = df['time'].max()
        t_grid = np.arange(1,t_max+1)

        ##Inducing vectors, which are the main parameters to be estimated
        U = pyro.sample('U', dist.Normal(torch.zeros([self.n_grid,self.n_vars]), torch.ones([self.n_grid,self.n_vars]) ).to_event(1).to_event(1) ) #Prior should be matched to Yildiz?
        Ug = pyro.sample('Ug', dist.Normal(torch.ones([self.n_grid,self.n_vars]), torch.ones([self.n_grid,self.n_vars] )).to_event(1).to_event(1) )#,constraint=constraints.positive #Prior should be matched to Yildiz?
        # U = pyro.sample('U', dist.MultivariateNormal(torch.zeros([n_grid**2,2]), torch.eye(2) ) ) #Prior should be matched to Yildiz?
        # Ug = pyro.sample('Ug', dist.MultivariateNormal(torch.zeros([n_grid**2,2]), torch.eye(2) ) ) #Prior should be matched to Yildiz?
        #Ug = pyro.sample('Ug', dist.HalfNormal(torch.ones([n_grid**2,2]) ).to_event(1).to_event(1) )#,constraint=constraints.positive #Prior should be matched to Yildiz?

        ##Euler-Maruyama sampling
        X = torch.tensor(df[df.time==0][self.vars].values.astype(np.float32))
        for t in np.arange(self.delta_t, t_max+self.delta_t, self.delta_t):
            f,g = self.calc_drift_diffusion(X, U, Ug, self.Z, self.Zg, self.kernel_f, self.kernel_g)
            X = pyro.sample('Xseq_{}'.format(t), dist.Normal(X + f * self.delta_t, g * torch.sqrt(torch.tensor([self.delta_t])) ).to_event(1).to_event(1)   )#Needs to be MultiVariate and iterate over sample to allow covariance.
            ##For t in the observed time step, find the observed variables and condition on the data.
            if t in t_grid:
                idx = (~balanced_df[balanced_df['time']==t][self.vars].isna()).values
                if np.sum(idx)!=0:
                    df_t  = balanced_df[balanced_df['time']==t][self.vars]
                    X_obs = torch.tensor( df_t.dropna().values.flatten().astype(np.float32) )
                    X_sample = X[[idx]]
                    ##Note that this flattens all the observed variable into a flat vector.
                    pyro.sample('Xobs_{}'.format(t), dist.Normal(X_sample, 1  ).to_event(1), obs=X_obs )


    def guide_map(self,df,balanced_df):
        '''
        The "guide" for MAP estimation of NPSDE model.
        '''

        t_max = df['time'].max()
        ##MAP estimate of parameters
        U_map = pyro.param('U_map', torch.zeros([self.n_grid,self.n_vars]) )
        #Ug_map = pyro.param('Ug_map', torch.ones([n_grid**2,2])*torch.sqrt(torch.tensor(2) )/torch.tensor(np.pi)  )#, constraint=constraints.positive
        Ug_map = pyro.param('Ug_map', torch.ones([self.n_grid,self.n_vars])  )#, constraint=constraints.positive
        U = pyro.sample("U", dist.Delta(U_map).to_event(1).to_event(1))
        Ug = pyro.sample("Ug", dist.Delta(Ug_map).to_event(1).to_event(1))

        ##Euler-Maruyama sampling
        X = torch.tensor(df[df.time==0][self.vars].values.astype(np.float32))
        for t in np.arange(self.delta_t, t_max+self.delta_t, self.delta_t):
            f,g = self.calc_drift_diffusion(X, U, Ug, self.Z, self.Zg, self.kernel_f, self.kernel_g)
            X = pyro.sample('Xseq_{}'.format(t), dist.Normal(X + f * self.delta_t, g * torch.sqrt(torch.tensor([self.delta_t])) ).to_event(1).to_event(1)  )#Needs to be MultiVariate and iterate over sample to allow covariance.


    def train(self, df,balanced_df, n_steps=1001, lr=0.01):
        pyro.clear_param_store()
        adam = pyro.optim.Adam({"lr": lr})
        self.svi = SVI(self.model, self.guide_map, adam, loss=Trace_ELBO())
        for step in range(n_steps):
            loss = self.svi.step(df,balanced_df)
            if step % 50 == 0:
                print('[iter {}]  loss: {:.4f}'.format(step, loss))

if __name__ == '__main__':
    df = pd.read_csv('data/seshat/Seshat_old_pca.csv')
    df['time'] = df[['NGA','Time']].groupby('NGA').transform(lambda x: (x - x.min()) / 100.)['Time']
    t_max = df['time'].max()
    t_grid = np.arange(1,t_max+1)

    unit_list = df.NGA.unique()
    nga_grid = np.array( [ [i]*len(t_grid) for i in  unit_list ] ).flatten()
    balanced_df = pd.DataFrame( {'time':np.tile(t_grid, len(unit_list)), 'NGA':nga_grid} )
    balanced_df = balanced_df.merge(df,how='left')

    pyro.clear_param_store()

    Zx_, Zy_ = np.meshgrid( np.linspace(df['PCA0'].min(), df['PCA0'].max(),5), np.linspace(df['PCA1'].min(), df['PCA1'].max(),5) )
    Z = torch.tensor( np.c_[Zx_.flatten(), Zy_.flatten()].astype(np.float32) )

    npsde = NPSDE(vars=['PCA0','PCA1'],sf_f=torch.tensor(1,dtype=torch.float32),sf_g=torch.tensor(1,dtype=torch.float32),ell_f=torch.tensor([1,1],dtype=torch.float32),ell_g=torch.tensor([1,1],dtype=torch.float32),Z=Z,fix_sf=False,fix_ell=True,fix_Z=False,delta_t=.1,jitter=1e-6)

    npsde.train(df,balanced_df)
