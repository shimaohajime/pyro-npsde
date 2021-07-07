import pandas as pd
import numpy as np
import pyro
import torch
import os 
import pyro.distributions as dist
import torch.distributions.constraints as constraints
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import time 
from copy import deepcopy 

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

    def model(self,X, delta_t, projection=0):
        '''
        NPSDE model.
        X : 2D array, first column for timestamps, the rest for components of timeseries 
        '''
        t_max = X[:,0].max() + projection
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

        ##Inducing vectors, which are the main parameters to be estimated
        U = pyro.sample('U', dist.Normal(torch.zeros([self.n_grid,self.n_vars]), torch.ones([self.n_grid,self.n_vars]) ).to_event(1).to_event(1) ) #Prior should be matched to Yildiz?
        Ug = pyro.sample('Ug', dist.Normal(torch.ones([self.n_grid,self.n_vars]), torch.ones([self.n_grid,self.n_vars] )).to_event(1).to_event(1) )#,constraint=constraints.positive #Prior should be matched to Yildiz?
        # U = pyro.sample('U', dist.MultivariateNormal(torch.zeros([n_grid**2,2]), torch.eye(2) ) ) #Prior should be matched to Yildiz?
        # Ug = pyro.sample('Ug', dist.MultivariateNormal(torch.zeros([n_grid**2,2]), torch.eye(2) ) ) #Prior should be matched to Yildiz?
        #Ug = pyro.sample('Ug', dist.HalfNormal(torch.ones([n_grid**2,2]) ).to_event(1).to_event(1) )#,constraint=constraints.positive #Prior should be matched to Yildiz?

        ##Euler-Maruyama sampling
        Xt = torch.tensor(X[X[:,0]==0][:, 1:])
        timestamps = np.arange(delta_t, t_max+delta_t, delta_t) 
        for i, t in enumerate(timestamps):
            f,g = self.calc_drift_diffusion(Xt, U, Ug, Z, Zg, kernel_f, kernel_g)
            Xt = pyro.sample('Xseq_{}'.format(i), dist.Normal(Xt + f * delta_t, g * torch.sqrt(torch.tensor([delta_t])) ).to_event(1).to_event(1)   )#Needs to be MultiVariate and iterate over sample to allow covariance.
            ##For t in the observed time step, find the observed variables and condition on the data.
            if t in t_grid:
                idx = (~np.isnan(X[X[:,0]==t][:, 1:]))
                if np.sum(idx)!=0:
                    df_t  = X[X[:,0]==t][:, 1:]
                    Xt_obs = torch.tensor( df_t[~np.isnan(df_t)].flatten() )
                    Xt_sample = Xt[[idx]]
                    ##Note that this flattens all the observed variable into a flat vector.
                    pyro.sample('Xobs_{}'.format(i), dist.Normal(Xt_sample, 1  ).to_event(1), obs=Xt_obs )


    def guide_map(self,X, delta_t, projection=0):
        '''
        The "guide" for MAP estimation of NPSDE model.
        '''

        t_max = X[:,0].max() + projection

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
        timestamps = np.arange(delta_t, t_max+delta_t, delta_t) 

        Xt = torch.tensor(X[X[:,0]==0][:, 1:]) # N x D 

        # Xs = torch.zeros([X.shape[0], len(timestamps), X.shape[1]]) # N x t x D 
        for i,t in enumerate(timestamps):
            f,g = self.calc_drift_diffusion(Xt, U, Ug, Z, Zg, kernel_f, kernel_g)
            Xt = pyro.sample('Xseq_{}'.format(i), dist.Normal(Xt + f * delta_t, g * torch.sqrt(torch.tensor([delta_t])) ).to_event(1).to_event(1)  )#Needs to be MultiVariate and iterate over sample to allow covariance.
        #     Xs[:, i, :] = X 

        # return Xs 


    def train(self, X, n_steps=1001, lr=0.01, Nw=50):
        # pyro.clear_param_store()
        adam = pyro.optim.Adam({"lr": lr})
        self.svi = SVI(self.model, self.guide_map, adam, loss=Trace_ELBO(num_particles=Nw))
        for step in range(n_steps):
            loss = self.svi.step(X, self.delta_t)
            if step % 50 == 0:
                print('[iter {}]  loss: {:.4f}'.format(step, loss))

    # takes as input dataset with missing data 
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
        pred = predictive.forward(X, self.delta_t)
        for i, time in enumerate(return_sites):
            Xs[:, :, :,  i+1] = pred[time].detach()

        
        return Xs 

    def forward(self, start_point, delta_t, timesteps, Nw):

        forward = torch.zeros((Nw, timesteps+1 , len(start_point)))  #  Nw , timestep,  dim 
        rsites = ['Xseq_%d' % i for i in range(timesteps)]
        predictive = pyro.infer.Predictive(self.model, guide=self.guide_map, num_samples=Nw, return_sites=rsites)
        X = np.atleast_2d(np.concatenate((np.zeros((1,1)), [start_point]), axis=1, dtype=np.float32))
        pred = predictive.forward(X, delta_t, projection=timesteps)

        for i, key in enumerate(rsites):
            forward[:,i+1,:] = pred[key][:,0,:]

        forward = forward.detach().numpy() 
        forward[:,0,:] = start_point

        return forward

    def forward_conditional_batch(self, points_batch, Nw, tolerance, dim=2, min_perc=0.2, max_steps=10):

        
        max_timestep = 0
        for n in range(len(points_batch)-1, -1, -1):
            points = np.array(points_batch[n])
            timestep = points.shape[0]-1
            if timestep > max_timestep:
                max_timestep = timestep 
        max_timestep = min(max_steps, max_timestep)

        forward = torch.zeros((len(points_batch), Nw, max_timestep , dim))  # Batch_entry, Nw , timestep,  dim 

        first_points = np.zeros((len(points_batch), dim))
        for n in range(len(points_batch)):
            points = np.array(points_batch[n])
            first_points[n,:] = points[0,:]

        rsites = ['Xseq_%d' % i for i in range(int(max_timestep/self.delta_t))]
        predictive = pyro.infer.Predictive(self.model, guide=self.guide_map, num_samples=Nw, return_sites=rsites)
        X = np.atleast_2d(np.concatenate((np.zeros((len(first_points),1)), first_points), axis=1, dtype=np.float32))
        pred = predictive.forward(X, projection=max_timestep)

        for n in range(len(points_batch)):
            for i, key in enumerate(rsites):
                if not (i+1) % (1/self.delta_t)==0:
                    continue
                forward[n, :,int((i+1) / (1/self.delta_t))-1,:] = pred[key][:,n,:]

        forward = forward.detach().numpy() 
        output = list() 
        imputations = 0

        for n in range(len(points_batch)):
            points = np.array(points_batch[n])
            forward_n = forward[n]
            within = []
            if points.shape[0]-1 > max_timestep:
                output.append(points)
                continue 
            for i in range(Nw):
                consistent = True
                has_values = False 
                for j in range(points.shape[0]-1):
                    target = points[j+1, ~np.isnan(points[j+1])]
            
                    proj = forward_n[i, j, ~np.isnan(points[j+1])]
                    if (not target.shape[0] == 0):
                        has_values = True
                    if (not target.shape[0] == 0) and np.linalg.norm(proj - target) > tolerance:
                       
                        consistent = False 
   

                if consistent:
                    within += [forward_n[i,:len(points)-1,:]]
    
            within = np.array(within)
            if len(within) >= int(Nw * min_perc):
                projected = np.concatenate((np.atleast_2d(points[0,:]), np.mean(within, axis=0)), axis=0)
                if len(projected) != len(points):
                    print('error!')
                
                nonempty_indices = np.array([[i,j] for i in range(points.shape[0]) for j in range(points.shape[1]) if ~np.isnan(points[i,j])]) 
                projected[nonempty_indices[:,0], nonempty_indices[:,1]] = points[nonempty_indices[:,0], nonempty_indices[:,1]]
                output.append(projected)
                imputations += np.isnan(points).sum()
            else:
                output.append(points) 

        return output, imputations 

    def fgAt(self, points):
        with torch.no_grad():
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

        f,g = self.calc_drift_diffusion(points, U, Ug, Z, Zg, kernel_f, kernel_g)

        return f,g 
        
    def impute(self, X, tolerance, threshold_u, threshold_ug, Nw=50,max_steps=10):

        def pairwise(l):
            if len(l) == 0:
                return []
            return pairwise(l[1:]) + [[l[0], l[i]] for i in range(1, len(l))]

        def check_viable_region_batch(start_points, end_points, threshold_u=threshold_u, threshold_ug=threshold_ug):

            with torch.no_grad():
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

            points = np.zeros((4*len(start_points), 2), dtype=np.float32)
            for n in range(len(start_points)):
                start_point = start_points[n]
                end_point = end_points[n]

                diff = (end_point - start_point)/2
                midpoint = (start_point + end_point)/2
                point1 = midpoint + np.array([-midpoint[1], midpoint[0]])
                point2 = midpoint + np.array([midpoint[1], -midpoint[0]])
                points[n*4: n*4+4] = [start_point, end_point, point1, point2]


            f,g = self.calc_drift_diffusion(points, U, Ug, Z, Zg, kernel_f, kernel_g)


            Us = f.detach().numpy()
            Ugs = g.detach().numpy()
            output = list()
            averages = list()
            for n in range(0, len(start_points)*4, 4):
                U = Us[n:n+4]
                Ug = Ugs[n:n+4]
                output += [np.max(list(map(lambda x: np.linalg.norm(x[1]-x[0]), pairwise(U)))) <= threshold_u and  np.max(list(map(lambda x: abs(x[1]-x[0]), pairwise(Ug)))) <= threshold_ug]
                averages.append((np.mean(U, axis=0), np.mean(Ug, axis=0)))

            return np.array(output), np.array(averages)

        def gp_regression(points, u, ug):
            offset = np.outer(np.arange(points.shape[0]), u) + points[0,:]
            adjusted_points = deepcopy(points)
            for i in range(adjusted_points.shape[0]):
                for j in range(adjusted_points.shape[1]):
                    if not np.isnan(adjusted_points[i,j]):
                        adjusted_points[i,j] -= offset[i,j] 

            # Assuming on-diagonal diffusion
            for d in range(adjusted_points.shape[1]):
                dimension_points = adjusted_points[1:,d]
                cov_matrix = np.array([[ (i+1) if i==j else min(i,j)+1 for j in range(len(dimension_points))] for i in range(len(dimension_points))]) * ug[d]
                filled_index = ~np.isnan(dimension_points)
                empty_index = np.isnan(dimension_points)
                m22 = cov_matrix[np.ix_(filled_index, filled_index)]
                m21 = cov_matrix[np.ix_(filled_index, empty_index)]
                m12 = cov_matrix[np.ix_(empty_index, filled_index)]
                m11 = cov_matrix[np.ix_(empty_index, empty_index)]
                
                adjusted_points[np.concatenate(([False],empty_index)), d] = np.matmul(np.atleast_1d(np.matmul(m12, np.linalg.inv(m22) if m22.shape[0]>1 else 1/m22)), dimension_points[filled_index])

            return adjusted_points + offset, np.isnan(points).sum() 

        print('Start..')
        breakoff = list(np.where(X[:, 0] == 0)[0])
        breakoff += [len(X)]
        print(breakoff)
        X = [X[breakoff[i] : breakoff[i+1], 1:] for i in range(len(breakoff) - 1)]

        indices = [] 
        snippets = []
        # Identify indexes that need filling 
        for i in range(len(X)):
            j = 1 
            while j < len(X[i])-1:
                if any(np.isnan(X[i][j,:])):
                    prevj = j-1 
                    nextj = -1 
                    
                    for next_j in range(j+1, len(X[i]), 1):
                        nextj = next_j
                        if not any(np.isnan(X[i][next_j,:])):
                            break 
                        
                    arr = X[i][prevj:nextj+1, :]

                    if nextj != -1 and not np.isnan(arr[-1, :]).any():
                        indices += [(i, prevj, nextj)]
                        snippets += [X[i][prevj:nextj+1, :]]
                        j = nextj + 1 
                    else:
                        break 
                        
                    
                else:
                    j += 1


        print('Matching missing data with suitable algorithm...')
        is_viable, averages = check_viable_region_batch(np.array([x[0,:] for x in snippets]), np.array([x[-1,:] for x in snippets]))
        
        forward_snippets = [snippets[i] for i in range(len(is_viable)) if ~is_viable[i]]
        forward_indices = [indices[i] for i in range(len(is_viable)) if ~is_viable[i]]
        missing_data = np.sum([np.isnan(X[i]).sum() for i in range(len(X))]) 

        print('Applying GP regression...')
        averages = averages[is_viable]
        imputations = 0
        for n,(i, prev_j, next_j) in enumerate([indices[i] for i in range(len(is_viable)) if is_viable[i]]):
            X[i][prev_j:next_j+1, :], count = gp_regression(X[i][prev_j:next_j+1, :], averages[n, 0], averages[n, 1])
            imputations += count
        print('%d imputations made' % imputations)

        print('Applying forward sampling...')
        forward_imputations, count = self.forward_conditional_batch(forward_snippets, Nw, tolerance, max_steps=max_steps)
        print('%d imputations made' % count)
        imputations += count

        print()
        print('%d/%d missing data filled' % (imputations, missing_data))

        for n, (i, prev_j, next_j) in enumerate(forward_indices):
            if (next_j - prev_j + 1 != len(forward_imputations[n])):
                print(prev_j, next_j, forward_imputations[n])
            X[i][prev_j:next_j+1, :] = forward_imputations[n]

        return X 

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

    def plot_model(self, X, prefix="",Nw=1, graphs=[1,2,3]):
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
        breakoff = list(np.where(complete_data[:, 0] == 0)[0])
        breakoff += [len(complete_data)]
        X_timeseries = [complete_data[breakoff[i] : breakoff[i+1], 1:] for i in range(len(breakoff) - 1)]

        if (1 in graphs):
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


        if 2 in graphs:
            # flattened_Y = np.asarray([Y[i][j] for i in range(len(Y)) for j in range(len(Y[i]))])
            extents = [Z[:,0].min(), Z[:,0].max(), Z[:,1].min(), Z[:,1].max()]

            
            W = 50
            # Fixed boundaries
            xv = np.linspace(extents[0], extents[1], W)
            yv = np.linspace(extents[2], extents[3], W)
            xvv,yvv = np.meshgrid(xv,yv, indexing='ij')


            Zs = np.array([xvv.T.flatten(),yvv.T.flatten()], dtype=np.float32).T

            U = self.unwhiten_U(U, Z, kernel_f)
            Ug = self.unwhiten_U(Ug, Zg, kernel_g)

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

        if 3 in graphs:
            fig = plt.figure(3, figsize=(15,12))
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

       

def pyro_npsde_run(csv_input, columns, steps, lr, Nw, sf_f,sf_g, ell_f, ell_g, W, fix_sf, fix_ell, fix_Z, delta_t, prefix):
    df = pd.read_csv(csv_input)
    components = columns

    assert(len(components) == 2)
    df['time'] = df[['entity','time']].groupby('entity').transform(lambda x: (x - x.min()) / 100. )['time']
    t_max = df['time'].max()
    t_grid = np.arange(t_max)

    unit_list = df.entity.unique()
    nga_grid = np.array( [ [i]*len(t_grid) for i in  unit_list ] ).flatten()
    balanced_df = pd.DataFrame( {'time':np.tile(t_grid, len(unit_list)), 'entity':nga_grid} )
    balanced_df = balanced_df.merge(df,how='left')

    X = balanced_df[['time'] + components].to_numpy(dtype=np.float32)

    pyro.clear_param_store()

    Zx_, Zy_ = np.meshgrid( np.linspace(df[components[0]].min(), df[components[0]].max(),W), np.linspace(df[components[1]].min(), df[components[1]].max(),W) )
    Z = torch.tensor( np.c_[Zx_.flatten(), Zy_.flatten()].astype(np.float32) )

    npsde = NPSDE(vars=components,sf_f=torch.tensor(sf_f,dtype=torch.float32),sf_g=torch.tensor(sf_g,dtype=torch.float32),ell_f=torch.tensor((ell_f),dtype=torch.float32),ell_g=torch.tensor((ell_g),dtype=torch.float32),Z=Z,fix_sf=int(fix_sf),fix_ell=int(fix_ell),fix_Z=int(fix_Z),delta_t=float(delta_t),jitter=1e-6)

    npsde.train(X, n_steps=steps, lr=lr, Nw=Nw)

    npsde.save_model('%s.pt' % prefix)
    npsde.plot_model(X, '%s' % prefix, Nw=1)

    return npsde.export_params() 


if __name__ == '__main__':
    df = pd.read_csv('data/seshat/Seshat_old_pca.csv')
    components =['PCA0','PCA1']
    df['time'] = df[['NGA','Time']].groupby('NGA').transform(lambda x: (x - x.min()) / 100. )['Time']

    nga_tmax=df[['NGA', 'time']].groupby('NGA')['time'].max()
    filled_nga = np.concatenate([[nga_tmax.index[i]] * int(nga_tmax.values[i]) for i in range(nga_tmax.shape[0])])
    filled_t = np.concatenate([np.arange(int(nga_tmax.values[i])).tolist() for i in range(nga_tmax.shape[0])])

    filled_df = pd.DataFrame({'time' : filled_t, 'NGA': filled_nga})
    filled_df = filled_df.merge(df, how='left')

    t_max = df['time'].max()
    t_grid = np.arange(t_max)

    unit_list = df.NGA.unique()
    nga_grid = np.array( [ [i]*len(t_grid) for i in  unit_list ] ).flatten()
    balanced_df = pd.DataFrame( {'time':np.tile(t_grid, len(unit_list)), 'NGA':nga_grid} )
    balanced_df = balanced_df.merge(df,how='left')

    X = balanced_df[['time'] + components].to_numpy(dtype=np.float32)
    X2 = filled_df[['time'] + components].to_numpy(dtype=np.float32)
    # # pyro.clear_param_store()

    # # Zx_, Zy_ = np.meshgrid( np.linspace(df['PCA0'].min(), df['PCA0'].max(),3), np.linspace(df['PCA1'].min(), df['PCA1'].max(),3) )
    # # Z = torch.tensor( np.c_[Zx_.flatten(), Zy_.flatten()].astype(np.float32) )

    # # npsde = NPSDE(vars=components,sf_f=torch.tensor(1,dtype=torch.float32),sf_g=torch.tensor(1,dtype=torch.float32),ell_f=torch.tensor([1,1],dtype=torch.float32),ell_g=torch.tensor([1,1],dtype=torch.float32),Z=Z,fix_sf=False,fix_ell=True,fix_Z=False,delta_t=.1,jitter=1e-6)

    # # npsde.train(X, n_steps=3, Nw=1)

    # # npsde.save_model('model1.pt')

    npsde = NPSDE.load_model('model1.pt')

    imputed_X = npsde.impute(X2, 0.5, 0.1, 0.1, max_steps=5)
    # flatten list 
    imputed_X = np.array([ imputed_X[i][j,:] for i in range(len(imputed_X)) for j in range(len(imputed_X[i]))])
    imputed_X = imputed_X[~np.isnan(imputed_X).any(axis=1), :]

    pd.DataFrame(imputed_X).to_csv('imputed.csv')

