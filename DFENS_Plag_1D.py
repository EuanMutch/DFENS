# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:56:36 2017

@author: ejfm2
"""
#FENICS
# from dolfin import *

# May need to rewrite so that intensive parameters are required for classes?
# read command line arguments
import os
import sys
import time

# numerical functions
import numpy as np
import scipy.integrate as integrate
import math as m
from scipy import special as sp
from scipy.interpolate import interp1d
from dolfin import *


# data
import pandas as pd

# MC
import pymultinest
import json

# plotting
import matplotlib.pyplot as plt
#import seaborn as sb

# multi thread
from mpi4py import MPI
from petsc4py import PETSc



# custom
import pmc
#import KC_fO2 as kc

set_log_active(False)

# SETUP MPI variables-------------------------------------------------
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
#comm = PETSc.Comm(MPI.COMM_SELF)
comm = mpi_comm_self()

# FUNCTIONS : DIFFUSION-------------------------------------------------
# Diffusion Functions
       
def D_plag(T_i, asio2, Xan, lnD0, cXan, caSiO2, cT_i):
    lnD = lnD0 + cXan*Xan + caSiO2*asio2 + cT_i*T_i
    
    D = exp(lnD)*Constant(1e12)
    
    return D        
        
def modc(model, fcoords, dist):
    mod_i = model[::-1]
    intp = interp1d(fcoords[:,0], mod_i,bounds_error=False, fill_value= "extrapolate")
    modx = intp(dist)
    return modx
        

def ICs_import(dist_init, IC, xs, rc, Q):
    i_int = interp1d(dist_init, IC ,bounds_error=False, fill_value=rc)
    ic = i_int(xs)
    Cinit = ic[dof_to_vertex_map(Q)] 
    return Cinit
        
# Model function

def mod_diff(cube, nparams):
    
    #t, T, fe3, P, aSiO2, DFo, DNi, DMn, A_PlMg, DMgpl = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5:11], cube[11:17], cube[17:23], cube[23], cube[24:nparams]
    t, T, aSiO2, B_PlMg, A_PlMg, DMgpl = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5:nparams]
    t *= 86400.0 # convert time into seconds
    dt = t/300.0 # only 50 time steps
    dT = Constant(dt)
    #P *= 1.0e8
    T += 273.0
    T_i = 1/T
    
    L_pl = max(dist_pl)
    n_pl = 299
    R = Constant(0.008314)
    A_Mg = Constant(A_PlMg)
    T_i = Constant(T_i)
    aSiO2 = Constant(aSiO2)
    #B_Mg = Constant(B_PlMg)
    
    
    # Adjust mesh depending on numerical stability
    Dd = m.exp(DMgpl[0] + DMgpl[1]*0.7 + DMgpl[2]*aSiO2 + DMgpl[3]*T_i)*1e12
    
    ms = L_pl/n_pl
    #print(ms)
    CFL = (dt*Dd)/(ms**2)
    
    if CFL > 0.5:
        Dx = m.sqrt((1/0.49)*dt*Dd)
        n = int(max(dist_pl)/Dx)
        n_pl = n
        #L = max(dist)
        L_pl = max(dist_pl)
        #olM = Mesh(n1, L1)  
        
    if n_pl < 1:
        n_pl = 2
        
    #Plagioclase Diffusion Models
    mesh_pl = IntervalMesh(comm, n_pl, 0.0, L_pl)
    xs_pl = np.linspace(0, L_pl, n_pl+1)
    
    #print(len(mesh_pl))
    Qpl = FunctionSpace(mesh_pl, "CG", 1)
    # Define boundaries
    def left_boundary_pl(x):
        return near(x[0], 0.0)

    def right_boundary_pl(x):
        return near(x[0], L_pl)
        
        
    # Define An Content using imported data (EPMA or BSE)
    An = Function(Qpl)

    # Interpolate Xan at data point
    An_dist, Xan = obs_pl['Distance'], obs_pl['Xan']
    An_int = interp1d(An_dist, Xan,bounds_error=False, fill_value= Xan[0])
    
    # Need to translate An data to mesh points using 1D interpolation
    An_mesh = An_int(xs_pl)
    An.vector()[:] = An_mesh[dof_to_vertex_map(Qpl)]
        
    # Construct weak form
    C0_Mgpl = Function(Qpl)       # Composition at current time step
    C1_Mgpl = TrialFunction(Qpl)  # Composition at next time step
    S_Mgpl = TestFunction(Qpl)
    
    theta = Constant(0.5)
    C_mid_Mgpl = theta*C1_Mgpl + Constant((1.0-theta))*C0_Mgpl

    DMgpl = D_plag(T_i, aSiO2, An, Constant(DMgpl[0]), Constant(DMgpl[1]), Constant(DMgpl[2]), Constant(DMgpl[3]))
    F1_Mgpl = (DMgpl)/(R*Constant(T))*(Constant(A_Mg))*grad(An)
    F_Mgpl = S_Mgpl*(C1_Mgpl-C0_Mgpl)*dx + dT*(inner(grad(S_Mgpl), DMgpl*grad(C_mid_Mgpl) - F1_Mgpl*(C_mid_Mgpl)))*dx

    a_Mgpl = lhs(F_Mgpl)
    L_Mgpl = rhs(F_Mgpl)

    Cbc0_Mgpl = DirichletBC(Qpl, Constant(rim_comp_Mgpl), left_boundary_pl) 
    Cbcs_Mgpl = [Cbc0_Mgpl]

    Cinit_Mgpl = ICs_import(dist_init_pl, inicon_pl['Mg'].values, xs_pl, rim_comp_Mgpl, Qpl)
    C0_Mgpl.vector()[:] = Cinit_Mgpl
        
    
    # Timestepping
    i = 0
    
    while i < t:

        solve(a_Mgpl==L_Mgpl, C0_Mgpl, Cbcs_Mgpl)
        
        i += dt # Need to decide on time increment
        
        
    rcoords_pl = mesh_pl.coordinates()
        
    Mgpl_mod = modc(C0_Mgpl.vector().get_local(), rcoords_pl, dist_pl)
    
    gl_mod = Mgpl_mod 
       
    return gl_mod
    
# loglikelihood function
    
# ---------------- log likelihood function
def loglike(model, data, data_err):
    return np.sum( -0.5*((model - data)/data_err)**2 )
    

#------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------    
if __name__ == "__main__":
    
    
    ## DATA-------------------------------------------------
    
    # import data from files
    
    
    f_dat_pl = sys.argv[1]
    f_modpar = sys.argv[2]
    f_inicon_pl = sys.argv[3]
    f_mk_pl = sys.argv[4]
    
    f_out = sys.argv[5]
    
    obs_pl = pd.read_csv(f_dat_pl)
    
    obs_mpar = np.genfromtxt(f_modpar,delimiter=',',dtype=None, names=True)
    inicon_pl = pd.read_csv(f_inicon_pl)
    df_mk_pl = pd.read_csv(f_mk_pl)
    
    # Import vcov files
    # Mg in plagioclase
    Mgpl_vcov = pd.read_csv('./D_vcov/Mgpl_FVOw_vcov.csv')
    MgKd_vcov = pd.read_csv('./D_vcov/Mg_Plag_partition_vcov1.csv')
    
    # Consider combining uncertainties and observations into a single file
    # Also for plag, probably convenient to combine all data sources into a single file with uncertainties as well.
    
    # combine different elements into a single global array
    
    gl_obs = obs_pl['Mg'].values  
    gl_err = obs_pl['Mg_sd'].values
    
    dist_pl = obs_pl['Distance'].values
    dist_init_pl = inicon_pl['Distance'].values
    
    #rim_comp_Mgpl, core_comp_Mgpl = df_mk['Fo_markers'][0], df_mk['Fo_markers'][1]
    
    rim_comp_Mgpl = df_mk_pl['Mg_markers'][0]
    
    #n, L = 299, max(dist)
    n_pl, L_pl = 299, max(dist_pl)
    #olM = Mesh(n, L)  # olivine mesh
    
    # check for output
    dir = '/'.join(f_out.split('/')[:-1])
    if not os.path.exists(dir):
        print("Making directory for output:", dir)
        os.makedirs(dir)
        
    # PARAMETER SETUP-------------------------------------------------
    # number of weight parameters (melt region sections = N(weights) + 1)
    parameters = ["t", "T", "aSiO2", "B_PlMg", "A_PlMg", "lnD0_Mgpl", "cXan_Mgpl", "caSiO2_Mpl", "cT_i_Mgpl"]
    pti = np.empty([len(parameters)])
    pti[:] = np.nan
    pti[3:5] = 0
    pti[5:] = 1
    
    
    cov_MgKd = MgKd_vcov.as_matrix(columns=['(Intercept)', 'Xan'])
    cov_Mgpl = Mgpl_vcov.as_matrix(columns=['(Intercept)', 'Xan', 'aSiO2', 'T_i'])
    
    cov_s = np.array([cov_MgKd, cov_Mgpl])
    
    #==============================================================================
    # for i in Ds.index.values:
    #     parameters.append(i)
    #==============================================================================
    n_dim = len(parameters)
    n_params = n_dim
    
    # MCMC-------------------------------------------------
    # setup prior cube
    ptype = ["MG"] * n_dim
    ptype[0:1] = ["LU"]   #type of distribution LU = ln uniform, U = uniform
    ptype[1:3] = ["G"]*2

    #ptype[23:24] = ["G"]
        
    pcube = np.full((n_dim,2), np.nan)
    pcube[0,:] = [2.0, 6.0]     # time (days)
    pcube[1,:] = [1190.0, 30.0]                           #float(obs_mpar['T'][0]), float(obs_mpar['T'][1])]     # T 
    pcube[2,:] = [0.62, 0.04]                     #float(obs_mpar['aSiO2'][0]), float(obs_mpar['aSiO2'][1])] # aSiO2 (kbar)
    pcube[3:5, 0] = np.array([-17.407, -34.099]) #Mg in plagioclase partitioning
    pcube[5:,0] = np.array([-1.424e1, -5.398, 3.854, -3.154e4]) #DMgpl
    
    invMC = pmc.Pmc(n_dim, ptype, pcube, cov_s, pti, loglike, mod_diff,
                    gl_obs, gl_err,
                    evidence_tolerance = 0.5, sampling_efficiency = 0.8,
                    n_live_points = 400, n_params= n_params)
    
    json.dump(parameters, open(f_out + 'params.json', 'w')) # save parameter names
    
    invMC.run_mc(f_out)
    result = invMC.result(f_out)
        
        # fiddle around with this to gaussian with variance and covariance, manually change prior cube
        
    if rank == 1 :
            # PLOT-------------------------------------------------
    # Prevents having to generate 4 plots when weaving on 4 separate processors
    #==============================================================================
        bf_params = result.get_best_fit()['parameters']
        C_bf = mod_diff(bf_params, len(bf_params))
        pd.DataFrame({'Mg_bf': C_bf}).to_csv(f_out + 'mod_cv.csv', sep=',')
            
        # data fit
        #fig, axes = plt.subplots(3,1)
        
        plt.errorbar(dist_pl, obs_pl['Mg'].values, yerr = obs_pl['Mg_sd'].values,fmt='o', color='red', label='data')
        plt.plot(dist_pl, inicon_pl['Mg'].values, color = 'black')
        plt.plot(dist_pl, C_bf, color = 'blue')
        plt.ylabel('Mg (ppm)')
        plt.xlabel('Distance (um)')
        #plt.errorbar(dist, obs['XFo'].values, yerr = obs['Fo_std'].values,fmt='o', color='red', label='data')
        #plt.plot(dist, C_Fo_bf, color = 'blue')
        
        
        plt.savefig(f_out + 'fit.png')
        plt.close()
        
        



