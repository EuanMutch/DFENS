# numerical functions
import numpy as np

import sys

import pymultinest

from mpi4py import MPI

import scipy

import scipy.stats

# SETUP MPI variables-------------------------------------------------
rank = MPI.COMM_WORLD.Get_rank()

#------------------------------------------------------------------------
# PMC class for packaging multi nest run
#------------------------------------------------------------------------
class Pmc:

    # valid prior structures
    valid_ps = ["U", "LU", "G", "MG"]

    def __init__(self, ndim, p_type, p_bounds, cov_s, pti,
                 logLike, yModel, y_data, y_err,
                 n_fix = None, n_params = None,
                 evidence_tolerance = 0.5, sampling_efficiency = 0.8, n_live_points = 400):

        # number of fixed parameters (n_params >= ndim)
        self.ndim = ndim
        self.n_params = n_params
        if n_params == None and n_fix == None:
            self.n_params = ndim
        elif n_params == None and n_fix != None:
            self.n_params = ndim + n_fix

        self.ptype = p_type
        self.p_bounds = np.array(p_bounds)

        self.n_fix = n_fix

        self.logLike = logLike
        self.yModel = yModel
        self.y_data = y_data
        self.y_err = y_err

        self.evidence_tolerance = evidence_tolerance
        self.sampling_efficiency = sampling_efficiency
        self.n_live_points = n_live_points
        
        self.cov_s = cov_s # This is a list of the covariance matrices for different sets of parameters e.g. cov_s = np.array([cov0, cov1, cov2]). Note the pythonic numbering (starting from 0).
        self.pti = pti # This is a list of indices linking parameters to the relevant covariance matrix. If they are not assigned to a covariance matrix (e.g are uniform prior) they are assigned nan.
        # E.g. 8 parameters: 2 are independent, 6 are related to 3 covariance matrices pti = np.array([nan, nan, 0, 0, 1, 1, 2, 2]). The first 2 are independent, the second 2 are related to cov0 etc. 

        # sanitise
        if n_fix and ndim != n_params-n_fix :
            raise ValueError

        if ndim != len(p_type):
            raise ValueError

        if set(p_type).difference(set(self.valid_ps)):
            raise ValueError

        if self.p_bounds.shape[0] != n_params:
            raise ValueError

    # FUNCTIONS : MCMC-------------------------------------------------
    def run_mc(self, prefix="chains/1-"):

        def prior(cube, ndim, n_params):
            p = 0
            
            for ps in self.ptype:
                if self.ptype[p] == "U":
                    cube[p] = self.p_bounds[p,0]\
                              + cube[p]*(self.p_bounds[p,1]-self.p_bounds[p,0])
                elif self.ptype[p] == "LU":
                    cube[p] = 10**(self.p_bounds[p,1]*cube[p] - self.p_bounds[p,0])
                elif self.ptype[p] == "G":
                    mu, std = self.p_bounds[p, 0], self.p_bounds[p, 1]
                    rv = scipy.stats.norm(mu, std) 
                    cube[p] = rv.ppf(cube[p])
                elif self.ptype[p] == "MG":   # For multivariate gaussians
                    rv2 = scipy.stats.norm()  # set up a regular gaussian distribution in prior cube then stretch and rotate later
                    cube[p] = rv2.ppf(cube[p]) 
                p += 1

            a = 0
            while a < len(self.cov_s):  # cycle through list of covariance matrices
                x = np.empty([0])
                mu2 = np.empty([0])
                mk = []
                cov = self.cov_s[a] #select covariance matrix
                [eival, eivec] = np.linalg.eig(cov) # obtain eigenvalue and eigen vector of covariance matrix
                l = np.matrix(np.diag(np.sqrt(eival)))
                Q = np.matrix(eivec)*l 
                p1 = 0
                for z in self.pti: # cycle through list of parameter indices
                    if z == a: # Link parameters to the corresponding covariance matrix using the indices
                        x = np.append(x, cube[p1])
                        mu2 = np.append(mu2, self.p_bounds[p1,0]) # make vector of mean values 
                        mk.append(p1)
                    p1 += 1
   
                y = np.dot(Q, x) + mu2 #stretch and rotate initial gaussian distribution here
            
                q2 = 0
                for z in mk:
                    cube[z] = y[0, q2] #Adjust relevant parameters in prior cube according to relevent covariance matrix
                    q2 += 1
                a += 1

        def loglike(cube, ndim, n_params):
            y_calc = self.yModel(cube, n_params)
            loglikelihood = self.logLike(y_calc, self.y_data, self.y_err)

            return(loglikelihood)

        MPI.COMM_WORLD.Barrier()

        pymultinest.run(loglike, prior, self.ndim, n_params = self.n_params,
                        outputfiles_basename = prefix,
                        resume = True, verbose = True,
                        importance_nested_sampling = False,
                        evidence_tolerance = self.evidence_tolerance,
                        sampling_efficiency = self.sampling_efficiency,
                        n_live_points = self.n_live_points,
                        log_zero = -1e90, multimodal = False, init_MPI = False)

        MPI.COMM_WORLD.Barrier()

    def result(self, prefix):
        
        #if rank == 1:
        a = pymultinest.Analyzer(n_params = self.n_params, outputfiles_basename = prefix)
        
        return(a)

























