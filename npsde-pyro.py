import pandas as pd
import numpy as np
import pyro
import torch
import os 
import pyro.distributions as dist
import torch.distributions.constraints as constraints
import matplotlib.pyplot as plt 
import matplotlib as mpl 

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

    def model(self,X):
        '''
        NPSDE model.
        X : 2D array, first column for timestamps, the rest for components of timeseries 
        '''
        t_max = X[:,0].max()
        t_grid = np.arange(1,t_max+1)

        sf_f = self.sf_f if self.fix_sf else pyro.param('sf_f', self.sf_f)
        sf_g = self.sf_g if self.fix_sf else pyro.param('sf_g', self.sf_g)
        ell_f = self.ell_f if self.fix_ell else pyro.param('ell_f', self.ell_f)
        ell_g = self.ell_g if self.fix_ell else pyro.param('ell_g', self.ell_g)
        Z = self.Z if self.fix_Z else pyro.param('Z', self.Z)
        Zg = self.Zg if self.fix_Z else pyro.param('Z', self.Z)

        ##Define kernels
        kernel_f = Kernel(sf_f, ell_f)
        kernel_g = Kernel(sf_g, ell_g)

        ##Inducing vectors, which are the main parameters to be estimated
        U = pyro.sample('U', dist.Normal(torch.zeros([self.n_grid,self.n_vars]), torch.ones([self.n_grid,self.n_vars]) ).to_event(1).to_event(1) ) #Prior should be matched to Yildiz?
        Ug = pyro.sample('Ug', dist.Normal(torch.ones([self.n_grid,self.n_vars]), torch.ones([self.n_grid,self.n_vars] )).to_event(1).to_event(1) )#,constraint=constraints.positive #Prior should be matched to Yildiz?
        # U = pyro.sample('U', dist.MultivariateNormal(torch.zeros([n_grid**2,2]), torch.eye(2) ) ) #Prior should be matched to Yildiz?
        # Ug = pyro.sample('Ug', dist.MultivariateNormal(torch.zeros([n_grid**2,2]), torch.eye(2) ) ) #Prior should be matched to Yildiz?
        #Ug = pyro.sample('Ug', dist.HalfNormal(torch.ones([n_grid**2,2]) ).to_event(1).to_event(1) )#,constraint=constraints.positive #Prior should be matched to Yildiz?

        ##Euler-Maruyama sampling
        Xt = torch.tensor(X[X[:,0]==1][:, 1:])
        timestamps = np.arange(self.delta_t, t_max+self.delta_t, self.delta_t) 
        for i, t in enumerate(timestamps):
            f,g = self.calc_drift_diffusion(Xt, U, Ug, Z, Zg, kernel_f, kernel_g)
            Xt = pyro.sample('Xseq_{}'.format(i), dist.Normal(Xt + f * self.delta_t, g * torch.sqrt(torch.tensor([self.delta_t])) ).to_event(1).to_event(1)   )#Needs to be MultiVariate and iterate over sample to allow covariance.
            ##For t in the observed time step, find the observed variables and condition on the data.
            if t in t_grid:
                idx = (~np.isnan(X[X[:,0]==t][:, 1:]))
                if np.sum(idx)!=0:
                    df_t  = X[X[:,0]==t][:, 1:]
                    Xt_obs = torch.tensor( df_t[~np.isnan(df_t)].flatten() )
                    Xt_sample = Xt[[idx]]
                    ##Note that this flattens all the observed variable into a flat vector.
                    pyro.sample('Xobs_{}'.format(i), dist.Normal(Xt_sample, 1  ).to_event(1), obs=Xt_obs )


    def guide_map(self,X):
        '''
        The "guide" for MAP estimation of NPSDE model.
        '''

        t_max = X[:,0].max()

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
        #Ug_map = pyro.param('Ug_map', torch.ones([n_grid**2,2])*torch.sqrt(torch.tensor(2) )/torch.tensor(np.pi)  )#, constraint=constraints.positive
        Ug_map = pyro.param('Ug_map', torch.ones([self.n_grid,self.n_vars])  )#, constraint=constraints.positive
        U = pyro.sample("U", dist.Delta(U_map).to_event(1).to_event(1))
        Ug = pyro.sample("Ug", dist.Delta(Ug_map).to_event(1).to_event(1))

        ##Euler-Maruyama sampling
        timestamps = np.arange(self.delta_t, t_max+self.delta_t, self.delta_t) 

        Xt = torch.tensor(X[X[:,0]==1][:, 1:]) # N x D 

        # Xs = torch.zeros([X.shape[0], len(timestamps), X.shape[1]]) # N x t x D 
        for i,t in enumerate(timestamps):
            f,g = self.calc_drift_diffusion(Xt, U, Ug, Z, Zg, kernel_f, kernel_g)
            Xt = pyro.sample('Xseq_{}'.format(i), dist.Normal(Xt + f * self.delta_t, g * torch.sqrt(torch.tensor([self.delta_t])) ).to_event(1).to_event(1)  )#Needs to be MultiVariate and iterate over sample to allow covariance.
        #     Xs[:, i, :] = X 

        # return Xs 


    def train(self, X, n_steps=1001, lr=0.01, Nw=50):
        # pyro.clear_param_store()
        adam = pyro.optim.Adam({"lr": lr})
        self.svi = SVI(self.model, self.guide_map, adam, loss=Trace_ELBO(num_particles=Nw))
        for step in range(n_steps):
            loss = self.svi.step(X)
            if step % 50 == 0:
                print('[iter {}]  loss: {:.4f}'.format(step, loss))

    def mc_samples(self, X, Nw=1):
        t_max = X[:,0].max()
        timeseries_count = len(X[X[:,0]==1])
        timestamps = np.arange(self.delta_t, t_max+self.delta_t, self.delta_t) 
        return_sites = [] 
        for i,t in enumerate(timestamps):
            if t in X[:,0]:
                return_sites += ['Xseq_{}'.format(i)]
        predictive = pyro.infer.Predictive(self.model, guide=self.guide_map, num_samples=Nw, return_sites=return_sites)
        Xs = np.zeros((Nw, timeseries_count, X.shape[1]-1, len(return_sites) + 1)) # Nw x N x D x T 
        for i in range(Nw):
            Xs[i, :, :, 0] = X[X[:,0]==1,1:]
        pred = predictive.forward(X)
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


    def plot_model(self, X, prefix="",Nw=1):
        mpl.rc('text', usetex=True)

        t_max = X[:, 0].max()
        t_grid = np.arange(1,t_max+1)

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

        complete_data = X
        complete_data = complete_data[~np.isnan(complete_data).any(axis=1), :]
        breakoff = list(np.where(complete_data[:, 0] == 1)[0])
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

        # f = self.unwhiten_U(f, Zs, self.kernel_f)
        # g = self.unwhiten_U(g, Zs, self.kernel_g)

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

        fig = plt.figure(2, figsize=(15,12))
        gs = mpl.gridspec.GridSpec(nrows=1, ncols=20)
        ax1 = plt.subplot(gs[0,0:19])
        ax2 = plt.subplot(gs[0,19])

        Ugs = Ugs.reshape(W, W, 2) 
        mag = (Ugs[:,:,0] ** 2 + Ugs[:,:,1] ** 2) ** 0.5
        x_max, y_max, mag_max, mag_min = Ugs[:,:,0].max(), Ugs[:,:,1].max() , mag.max(), mag.min()
        x_delta, y_delta = (extents[1] - extents[0]) / W, (extents[3] - extents[2]) / W

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
        ax1.set_xlabel('$PC_1$ x%.3f' % (x_max / x_delta), fontsize=12)
        ax1.set_ylabel('$PC_2$ x%.3f' % (y_max / y_delta), fontsize=12)
        ax1.set_xlim(extents[0], extents[1])
        ax1.set_ylim(extents[2], extents[3])
        mpl.colorbar.ColorbarBase(ax2, cmap = cmap, norm = mpl.colors.Normalize(vmin=mag_min, vmax=mag_max), orientation='vertical')
        plt.savefig('%sdiff.png' % prefix, dpi=200)

       

if __name__ == '__main__':
    df = pd.read_csv('data/seshat/Seshat_old_pca.csv')
    components =['PCA0','PCA1']
    df['time'] = df[['NGA','Time']].groupby('NGA').transform(lambda x: (x - x.min()) / 100. + 1)['Time']
    t_max = df['time'].max()
    t_grid = np.arange(1,t_max+1)

    unit_list = df.NGA.unique()
    nga_grid = np.array( [ [i]*len(t_grid) for i in  unit_list ] ).flatten()
    balanced_df = pd.DataFrame( {'time':np.tile(t_grid, len(unit_list)), 'NGA':nga_grid} )
    balanced_df = balanced_df.merge(df,how='left')

    X = balanced_df[['time'] + components].to_numpy(dtype=np.float32)

    # pyro.clear_param_store()

    # Zx_, Zy_ = np.meshgrid( np.linspace(df['PCA0'].min(), df['PCA0'].max(),3), np.linspace(df['PCA1'].min(), df['PCA1'].max(),3) )
    # Z = torch.tensor( np.c_[Zx_.flatten(), Zy_.flatten()].astype(np.float32) )

    # npsde = NPSDE(vars=components,sf_f=torch.tensor(1,dtype=torch.float32),sf_g=torch.tensor(1,dtype=torch.float32),ell_f=torch.tensor([1,1],dtype=torch.float32),ell_g=torch.tensor([1,1],dtype=torch.float32),Z=Z,fix_sf=False,fix_ell=True,fix_Z=False,delta_t=.1,jitter=1e-6)

    # npsde.train(X, n_steps=3, Nw=1)

    # npsde.save_model('model1.pt')

    npsde = NPSDE.load_model('model1.pt')

    npsde.plot_model(X, 'test')


