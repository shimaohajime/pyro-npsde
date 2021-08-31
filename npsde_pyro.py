import pandas as pd
import numpy as np
import pyro
import torch
import os 
import pyro.distributions as dist
import torch.distributions.constraints as constraints
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import math 
from copy import deepcopy 
from pyro.poutine import trace 
from pprint import pprint 
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

    hyperparameters = ['vars','sf_f','sf_g','ell_f','ell_g','Z','fix_sf','fix_ell','fix_Z','delta_t','jitter']
    diffusion_dimensions = 1 

    def __init__(self,vars,sf_f,sf_g,ell_f,ell_g,Z,fix_sf,fix_ell,fix_Z,delta_t,jitter):

        self.vars = vars #list of variable names to be estimated. (ex) ['PCA0','PCA1']
        self.n_vars = len(vars)

        ##Hyperparameters, either fixed or learned.
        self.sf_f = sf_f 
        self.sf_g = sf_g 
        self.ell_f = ell_f 
        self.ell_g = ell_g 
        self.Z = Z 
        self.Zg = self.Z #Inducing location for drift and diffusion are set same.

        ##For save_model
        self.fix_sf = fix_sf 
        self.fix_ell = fix_ell 
        self.fix_Z = fix_Z 

        self.n_grid = Z.shape[0]
        self.delta_t = delta_t #Euler-Maruyama time discretization.
        self.jitter = jitter

    def compute_f(self, X, U, Z, kernel):
        N = X.shape[0]
        M = Z.shape[0]
        D = Z.shape[1] # dim of state
        Kzz = kernel.K(Z) + torch.rand(M) * self.jitter
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

    def model(self,X, guided=True):
        '''
        NPSDE model.
        X : 2D array, first column for timestamps, the rest for components of timeseries 
        '''
        t_max = X[:,0].max()
        t_grid = np.arange(t_max)

        sf_f = self.sf_f if self.fix_sf else pyro.param('sf_f', self.sf_f)
        sf_g = self.sf_g if self.fix_sf else pyro.param('sf_g', self.sf_g)
        ell_f = self.ell_f if self.fix_ell else pyro.param('ell_f', self.ell_f)
        ell_g = self.ell_g if self.fix_ell else pyro.param('ell_g', self.ell_g)
        Z = self.Z if self.fix_Z else pyro.param('Z', self.Z)
        Zg = self.Zg if self.fix_Z else pyro.param('Z', self.Z)
        noise = pyro.param('noise', torch.tensor(1.0))

        ##Define kernels
        kernel_f = Kernel(sf_f, ell_f)
        kernel_g = Kernel(sf_g, ell_g)
        ##Inducing vectors, which are the main parameters to be estimated
        U = pyro.sample('U', dist.Normal(torch.zeros([self.n_grid,self.n_vars]), torch.ones([self.n_grid,self.n_vars])+1 ).to_event(1).to_event(1) ) #Prior should be matched to Yildiz?
        Ug = pyro.sample('Ug', dist.Normal(torch.ones([self.n_grid,1]), torch.ones([self.n_grid,1] )).to_event(1).to_event(1) )#,constraint=constraints.positive #Prior should be matched to Yildiz?

        ##Euler-Maruyama sampling
        Xt = torch.tensor(X[X[:,0]==0][:, 1:])
        timestamps = np.arange(self.delta_t, t_max+self.delta_t, self.delta_t)
        for i, t in enumerate(timestamps):
            f,g = self.calc_drift_diffusion(Xt, U, Ug, Z, Zg, kernel_f, kernel_g)
 
            Xt = pyro.sample('Xseq_{}'.format(i), dist.Normal(Xt + f * self.delta_t, g * torch.sqrt(torch.tensor([self.delta_t])) + torch.rand(g.shape) * self.jitter ).to_event(1).to_event(1)   )#Needs to be MultiVariate and iterate over sample to allow covariance.
            ##For t in the observed time step, find the observed variables and condition on the data.
            if t in t_grid:
                idx = (~np.isnan(X[X[:,0]==t][:, 1:]))
                if np.sum(idx)!=0:
                    df_t  = X[X[:,0]==t][:, 1:]
                    Xt_obs = torch.tensor( df_t[tuple([idx])] )
                    Xt_sample = Xt[tuple([idx])]
                    ##Note that this flattens all the observed variable into a flat vector.
                    pyro.sample('Xobs_{}'.format(i), dist.Normal(Xt_sample, noise ).to_event(1), obs=Xt_obs )

                    # Piecewise
                    if guided: 
                      Xt_new = torch.tensor(df_t)
                      Xt_new[[~idx]] = Xt[[~idx]]
                      Xt = Xt_new

    def guide_map(self,X, guided=True):
        '''
        The "guide" for MAP estimation of NPSDE model.
        '''

        t_max = X[:,0].max()
        t_grid = np.arange(t_max)

        sf_f = self.sf_f if self.fix_sf else pyro.param('sf_f', self.sf_f)
        sf_g = self.sf_g if self.fix_sf else pyro.param('sf_g', self.sf_g)
        ell_f = self.ell_f if self.fix_ell else pyro.param('ell_f', self.ell_f)
        ell_g = self.ell_g if self.fix_ell else pyro.param('ell_g', self.ell_g)
        Z = self.Z if self.fix_Z else pyro.param('Z', self.Z)
        Zg = self.Zg if self.fix_Z else pyro.param('Z', self.Z)

        ##Define kernels
        kernel_f = Kernel(sf_f, ell_f)
        kernel_g = Kernel(sf_g, ell_g)


        ##MAP estimate of parameters
        U_map = pyro.param('U_map', torch.zeros([self.n_grid,self.n_vars]) )
        # U_cov_matrix = pyro.param('U_cov_matrix', torch.stack([torch.eye(self.n_vars) for _ in range(self.n_grid)]) , constraint=constraints.positive_definite)
        Ug_map = pyro.param('Ug_map', torch.ones([self.n_grid,NPSDE.diffusion_dimensions])  , constraint=constraints.positive)
        # Ug_cov_matrix = pyro.param('Ug_cov_matrix', torch.stack([torch.eye(NPSDE.diffusion_dimensions) for _ in range(self.n_grid)]), constraint=constraints.positive_definite)
        
        # U = pyro.sample("U", dist.MultivariateNormal(U_map, U_cov_matrix).to_event(1))
        # Ug = pyro.sample("Ug", dist.MultivariateNormal(Ug_map, Ug_cov_matrix).to_event(1))
        U = pyro.sample("U", dist.Delta(U_map).to_event(1).to_event(1))
        Ug = pyro.sample("Ug", dist.Delta(Ug_map).to_event(1).to_event(1))

        ##Euler-Maruyama sampling
        timestamps = np.arange(self.delta_t, t_max+self.delta_t, self.delta_t) 

        Xt = torch.tensor(X[X[:,0]==0][:, 1:]) # N x D 

        for i,t in enumerate(timestamps):
            f,g = self.calc_drift_diffusion(Xt, U, Ug, Z, Zg, kernel_f, kernel_g)
            Xt = pyro.sample('Xseq_{}'.format(i), dist.Normal(Xt + f * self.delta_t, g * torch.sqrt(torch.tensor([self.delta_t])) ).to_event(1).to_event(1)  )#Needs to be MultiVariate and iterate over sample to allow covariance.
            # piecewise
            if guided:
              if t in t_grid:
                  idx = (~np.isnan(X[X[:,0]==t][:, 1:]))

                  if np.sum(idx)!=0:
                      df_t  = X[X[:,0]==t][:, 1:]
                      Xt_new = torch.tensor(df_t)
                      Xt_new[[~idx]] = Xt[[~idx]]
                      Xt = Xt_new 

    def train(self, X, n_steps=1001, lr=0.01, Nw=50):
        # pyro.clear_param_store()
        def dist(p1, p2):
          return np.linalg.norm(p1-p2)

        adam = pyro.optim.Adam({"lr": lr})
        Z = self.Z if self.fix_Z else pyro.param('Z', self.Z)

        self.svi = SVI(self.model, self.guide_map, adam, loss=Trace_ELBO(num_particles=Nw))
        try:
            for step in range(n_steps):
              
                  loss = self.svi.step(X)
                 
                  if step % 5 == 0:
                    log_evidence = []
                    with torch.no_grad():
                      for _ in range(20):
                        tr = trace(self.model).get_trace(X)
                        log_evidence += [tr.log_prob_sum()]
                    print('[iter {}]  loss: {:.4f}'.format(step, loss))
                    print('log evidence: {:.4f}'.format(np.mean(log_evidence)))


        except Exception as e :
          print(e)
          with torch.no_grad():
            sf_f = self.sf_f if self.fix_sf else pyro.param('sf_f', self.sf_f)
            ell_f = self.ell_f if self.fix_ell else pyro.param('ell_f', self.ell_f)
            Z = self.Z if self.fix_Z else pyro.param('Z', self.Z)

            ##Define kernels
            kernel_f = Kernel(sf_f, ell_f)
            Kzz = kernel_f.K(Z) + torch.eye(Z.shape[0]) * self.jitter

    def mc_samples(self, X, Nw=1):
        t_max = X[:,0].max()
        timeseries_count = len(X[X[:,0]==0])
        timestamps = np.arange(self.delta_t, t_max+self.delta_t, self.delta_t) 
        return_sites = [] 
        for i,t in enumerate(timestamps):
            if t in X[:,0]:
                return_sites += ['Xseq_{}'.format(i)]
        predictive = pyro.infer.Predictive(self.model, guide=self.guide_map, num_samples=Nw, return_sites=return_sites)
        Xs = np.zeros((Nw, timeseries_count, X.shape[1]-1, len(return_sites) + 1)) # Nw x N x D x T 
        for i in range(Nw):
            Xs[i, :, :, 0] = X[X[:,0]==0,1:]
        pred = predictive.forward(X, guided=False)
        for i, time in enumerate(return_sites):
            Xs[:, :, :,  i+1] = pred[time].detach()

        
        return Xs 

    def save_model(self, path):
        filename = os.path.basename(path)
        pyro.get_param_store().save(os.path.join(os.path.dirname(path), os.path.splitext(filename)[0] + '_params' + os.path.splitext(filename)[1]))
        constant_hyperparameters = list(set(NPSDE.hyperparameters).difference(set(pyro.get_param_store().get_all_param_names()))) # Get list of constant hyperparameters 
        torch.save({
            'constants' : { param : getattr(self, param) for param in constant_hyperparameters}
        }, path) 

    @staticmethod
    def load_model(path):
        filename = os.path.basename(path)

        pyro.get_param_store().clear()
        pyro.get_param_store().load(os.path.join(os.path.dirname(path), os.path.splitext(filename)[0] + '_params' + os.path.splitext(filename)[1]))

        metadata = torch.load(path)

        constr_args = [
            metadata['constants'][param] if param in metadata['constants'] else pyro.get_param_store().get_param(param).detach() for param in NPSDE.hyperparameters
        ]

        # Create new npsde object and attach fields to params 
        return NPSDE(*constr_args) 

    def export_params(self):
        sf_f = self.sf_f if self.fix_sf else pyro.param('sf_f', self.sf_f).detach().numpy()
        sf_g = self.sf_g if self.fix_sf else pyro.param('sf_g', self.sf_g).detach().numpy()
        ell_f = self.ell_f if self.fix_ell else pyro.param('ell_f', self.ell_f).detach().numpy()
        ell_g = self.ell_g if self.fix_ell else pyro.param('ell_g', self.ell_g).detach().numpy()

        return {
          'sf_f' : sf_f, 
          'sf_g' : sf_g, 
          'ell_f' : ell_f, 
          'ell_g' : ell_g 
        }

    def plot_model(self, X, prefix="",Nw=1):
        mpl.rc('text', usetex=False)

        # X0 = torch.tensor(df[df.time==0][self.vars].values.astype(np.float32))

        with torch.no_grad():
            Y = self.mc_samples(X, Nw=1) # Nw x N x D x T 
            Z = self.Z.detach() 
            Zg = self.Zg.detach() 
            U = pyro.get_param_store().get_param('U_map').detach() 
            Ug = pyro.get_param_store().get_param('Ug_map').detach() 

            sf_f = self.sf_f if self.fix_sf else pyro.param('sf_f', self.sf_f).detach() 
            sf_g = self.sf_g if self.fix_sf else pyro.param('sf_g', self.sf_g).detach() 
            ell_f = self.ell_f if self.fix_ell else pyro.param('ell_f', self.ell_f).detach() 
            ell_g = self.ell_g if self.fix_ell else pyro.param('ell_g', self.ell_g).detach() 
            kernel_f = Kernel(sf_f, ell_f)
            kernel_g = Kernel(sf_g, ell_g)

        U = self.unwhiten_U(U, Z, kernel_f)
        Ug = self.unwhiten_U(Ug, Zg, kernel_g)




        complete_data = X
        complete_data = complete_data[~np.isnan(complete_data).any(axis=1), :]
        breakoff = list(np.where(complete_data[:, 0] == 0)[0])
        breakoff += [len(complete_data)]
        X_timeseries = [complete_data[breakoff[i] : breakoff[i+1], 1:] for i in range(len(breakoff) - 1)]
        plt.figure(1,figsize=(20,12))
        gs = mpl.gridspec.GridSpec(1, 1)
        ax1 = plt.subplot(gs[0,0])
        for j in range(len(X_timeseries)):
            for i in range(Y.shape[0]):
                pathh, = ax1.plot(Y[i,j,0,:],Y[i,j,1,:],'b-',linewidth=0.5,label='samples')
            dh, = ax1.plot(X_timeseries[j][:,0],X_timeseries[j][:,1],'-ro',markersize=4,linewidth=0.3,label='data points')
            
        
        
        ilh = ax1.scatter(Z[:,0],Z[:,1],100, facecolors='none', edgecolors='k',label='inducing locations')
        ivh = ax1.quiver(Z[:,0],Z[:,1],U[:,0],U[:,1],units='height',width=0.006,color='k',label='inducing vectors')
        ax1.set_xlabel('PC1', fontsize=30)
        ax1.set_ylabel('PC2', fontsize=30)
        # Commented out because it is causing image size issue 
        ax1.legend(handles=[pathh,ilh,ivh,dh],loc=2) 
        ax1.set_title('Vector Field',fontsize=30)
        plt.savefig('%sdrift_sde.png' % prefix, dpi=400)
        # plt.show()


        # flattened_Y = np.asarray([Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))])
        extents = [Z[:,0].min(), Z[:,0].max(), Z[:,1].min(), Z[:,1].max()]

        
        W = 50
        # Fixed boundaries
        xv = np.linspace(extents[0], extents[1], W)
        yv = np.linspace(extents[2], extents[3], W)
        xvv,yvv = np.meshgrid(xv,yv, indexing='ij')


        Zs = np.array([xvv.T.flatten(),yvv.T.flatten()], dtype=np.float32).T

        f,g = self.calc_drift_diffusion(Zs, U, Ug, Z, Zg, kernel_f, kernel_g)

        Us = f.detach().numpy()
        Ugs = g.detach().numpy()
        
        fig = plt.figure(2, figsize=(15,12))
        gs = mpl.gridspec.GridSpec(nrows=1, ncols=1)
        ax1 = plt.subplot(gs[0,0])
        Uc = np.sqrt(Us[:,0] ** 2 + Us[:,1] ** 2)
        strm = ax1.streamplot(np.unique(list(Zs[:,0])), np.unique(list(Zs[:,1])), Us[:,0].reshape(W,W), Us[:,1].reshape(W,W), color=Uc.reshape(W,W), cmap='autumn')
        ax1.set_title('Drift Stream')
        fig.colorbar(strm.lines)
        plt.savefig('%sdrift_stream.png' % prefix, dpi=200)

        fig = plt.figure(3, figsize=(15,12))
        gs = mpl.gridspec.GridSpec(nrows=1, ncols=20)
        ax1 = plt.subplot(gs[0,0:19])
        ax2 = plt.subplot(gs[0,19])


        Ugs = Ugs.reshape(W, W, -1)
        x_delta, y_delta = (extents[1] - extents[0]) / W, (extents[3] - extents[2]) / W

        if Ugs.shape[2] == 1:
          ax1.imshow(Ugs[:,:,0], extent=extents, origin='lower', interpolation='nearest') 
        elif Ugs.shape[2] == 2: 
          mag = (Ugs[:,:,0] ** 2 + Ugs[:,:,1] ** 2) ** 0.5
          x_max, y_max, mag_max, mag_min = Ugs[:,:,0].max(), Ugs[:,:,1].max() , mag.max(), mag.min()

          Zs_grid = Zs.reshape(W, W, 2)
          ellipses = [] 
          cmap = mpl.cm.get_cmap('viridis')
          for r in range(W):
              for c in range(W):
                  ellipses += [ax1.add_patch(mpl.patches.Ellipse(Zs_grid[r, c], Ugs[r,c,0]*x_delta/x_max, Ugs[r,c,1]*y_delta/y_max, color=cmap(((Ugs[r,c,0] ** 2 + Ugs[r,c,1] ** 2) ** 0.5 - mag_min) / (mag_max - mag_min))))]

        # Display locations and scales of diffusion inducing points 
        # diff_Z = Zg
        # diff_U = Ug
        # diff_U = diff_U / np.max(diff_U) * (100 ** 2)
        # ax1.scatter(diff_Z[:,0], diff_Z[:,1], s=diff_U, facecolors='none', edgecolors='white')
        ax1.set_title('estimated diffusion')
        ax1.set_xlabel('$PC_1$', fontsize=12)
        ax1.set_ylabel('$PC_2$', fontsize=12)
        ax1.set_xlim(extents[0], extents[1])
        ax1.set_ylim(extents[2], extents[3])
        # mpl.colorbar.ColorbarBase(ax2, cmap = cmap, norm = mpl.colors.Normalize(vmin=mag_min, vmax=mag_max), orientation='vertical')
        plt.savefig('%sdiff.png' % prefix, dpi=200)

       
def pyro_npsde_run(df, components, steps, lr, Nw, sf_f,sf_g, ell_f, ell_g, W, fix_sf, fix_ell, fix_Z, delta_t, prefix):
    
    assert(len(components) == 2)
    
    df['time'] = df[['entity','time']].groupby('entity').transform(lambda x: (x - x.min()))['time']
    min_diff = min(abs(np.diff(df['time'])))
    df['time'] = df['time']/min_diff
    t_max = df['time'].max()
    t_grid = np.arange(t_max)

    unit_list = df.entity.unique()
    nga_grid = np.array( [ [i]*len(t_grid) for i in  unit_list ] ).flatten()
    balanced_df = pd.DataFrame( {'time':np.tile(t_grid, len(unit_list)), 'entity':nga_grid} )
    balanced_df = balanced_df.merge(df,how='left')
    X = balanced_df[['time'] + components].to_numpy(dtype=np.float32)


    # X = df[['time'] + components].to_numpy(dtype=np.float32)
    pyro.clear_param_store()

    Zx_, Zy_ = np.meshgrid( np.linspace(df[components[0]].min(), df[components[0]].max(),W), np.linspace(df[components[1]].min(), df[components[1]].max(),W) )
    Z = torch.tensor( np.c_[Zx_.flatten(), Zy_.flatten()].astype(np.float32) )

    npsde = NPSDE(vars=components,sf_f=torch.tensor(sf_f,dtype=torch.float32),sf_g=torch.tensor(sf_g,dtype=torch.float32),ell_f=torch.tensor((ell_f),dtype=torch.float32),ell_g=torch.tensor((ell_g),dtype=torch.float32),Z=Z,fix_sf=int(fix_sf),fix_ell=int(fix_ell),fix_Z=int(fix_Z),delta_t=float(delta_t),jitter=1e-6)

    npsde.train(X, n_steps=steps, lr=lr, Nw=Nw)

    npsde.save_model('%s.pt' % prefix)
    npsde.plot_model(X, '%s' % prefix, Nw=1)

    return npsde.export_params() 

if __name__ == "__main__":
    pyro_npsde_run(pd.read_csv('data/seshat/Seshat_old_pca.csv'), ['PCA0', 'PCA1'], 150, 0.02, 20, 1, 1, [1,1], [1,1], 5, False, False, False, 0.1, "model1")


