# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:56:36 2017

@author: ejfm2
"""

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
#from mpi4py import MPI
#from petsc4py import PETSc



# custom
import pmc
import KC_fO2 as kc # Calculates oxygen fugacity from melt compositions

set_log_active(False)

# SETUP MPI variables-------------------------------------------------
size = MPI.comm_world.Get_size()
rank = MPI.comm_world.Get_rank()
comm = MPI.comm_self

# FUNCTIONS : DIFFUSION-------------------------------------------------
# Diffusion Functions


def D(C, T_i, P, lnfO2, lnD0, clnfO2, cXFo, cT_i, cP, cT_iP, aniso):
    # Calculate diffusion coefficient given parameters generated in each realisation
    lnD = lnD0 + clnfO2*lnfO2 + cXFo*C + cT_i*T_i + cP*P + cT_iP*T_i*P
    
    D_001 = exp(lnD)
    D_100 = (Constant(1.0)/Constant(aniso))*D_001
    D_010 = (Constant(1.0)/Constant(aniso))*D_001

    D = (D_100*Constant(m.cos(psi)**2.0) + D_010*Constant(m.cos(phi)**2.0) + D_001*Constant(m.cos(gamma)**2.0))*Constant(1e12)
    return D        
        
        
        
def modc(model, fcoords, dist):
    # Interpolate model points onto observation distances
    mod_i = model[::-1]
    intp = interp1d(fcoords[:,0], mod_i,bounds_error=False, fill_value= mod_i[0])
    modx = intp(dist)
    return modx
        

def ICs_import(dist_init, IC, xs, rc, Q):
    # Interpolate imported initial conditions onto 1D mesh
    i_int = interp1d(dist_init, IC ,bounds_error=False, fill_value=rc)
    ic = i_int(xs)
    Cinit = ic[dof_to_vertex_map(Q)] 
    return Cinit
        
##################################################################################################################################################################

def mod_diff(cube, nparams):
    # Diffusion model function that is run in pymultinest multiple times
    t, T, fe3, P, DFo, DNi, DMn = cube[0], cube[1], cube[2], cube[3], cube[4:10], cube[10:16], cube[16:nparams] #parameters taken from prior distributions
    t *= 86400.0 # convert time into seconds
    dt = t/300.0 # only 300 time steps
    dT = Constant(dt)
    P *= 1.0e8 # Convert pressure to Pa
    T += 273.0 # Temperature to K
    T_i = 1/T # Inverse Temperature
    L = max(dist) # Maximum observed distance along profile
    n = 299 # Inital number of mesh points
    #calculate fO2 
    lnfO2 = kc.fO2calc_eq7(melt_comp, T, P, fe3)  # fO2 in bars from Kress and Carmichael
    
    # Adjust mesh depending on numerical stability (CFL condition)
    Dd = m.exp(DFo[0] + DFo[1]*lnfO2 + DFo[2]*0.9 + DFo[3]*T_i + DFo[4]*P + DFo[5]*T_i*P)*1e12
    ms = L/n
    CFL = (dt*Dd)/(ms**2)
    
    if CFL > 0.5:
        Dx = m.sqrt((1/0.49)*dt*Dd)
        n = int(max(dist)/Dx)
        L = max(dist) 
        
    if n < 1:
        n = 2
    
    # Create mesh with spacings defined by CFL condition above: use PETSc comm so that separate meshes can be used on separate processes
    mesh = IntervalMesh(comm, n, 0.0, L) 
    xs = np.linspace(0, L, n+1)

    Q = FunctionSpace(mesh, "CG", 1)
    # Define boundaries
    def left_boundary(x):
        return near(x[0], 0.0)

    def right_boundary(x):
        return near(x[0], L)
        
    # Construct weak form
    C0_fo = Function(Q)       # Composition at current time step Forsterite
    C1_fo = Function(Q)  # Composition at next time step
    S_fo = TestFunction(Q)
     
    C0_ni = Function(Q)       # Composition at current time step Ni
    C1_ni = TrialFunction(Q)  # Composition at next time step
    S_ni = TestFunction(Q)
    
    C0_mn = Function(Q)       # Composition at current time step Ni
    C1_mn = TrialFunction(Q)  # Composition at next time step
    S_mn = TestFunction(Q)
    
    theta = Constant(0.5)
    C_mid_fo = theta*C1_fo + Constant((1.0-theta))*C0_fo
    C_mid_ni = theta*C1_ni + Constant((1.0-theta))*C0_ni
    C_mid_mn = theta*C1_mn + Constant((1.0-theta))*C0_mn

    T_i = Constant(T_i) # Make parameters that go into D Constants
    P = Constant(P)
    lnfO2 = Constant(lnfO2)
    
    F_fo = S_fo*(C1_fo-C0_fo)*dx + dT*(inner(D(C_mid_fo, T_i, P, lnfO2, Constant(DFo[0]), Constant(DFo[1]), Constant(DFo[2]), Constant(DFo[3]), Constant(DFo[4]), Constant(DFo[5]), 6.0)*grad(S_fo), grad(C_mid_fo)))*dx
    F_ni = S_ni*(C1_ni-C0_ni)*dx + dT*(inner(D(C_mid_fo, T_i, P, lnfO2, Constant(DNi[0]), Constant(DNi[1]), Constant(DNi[2]), Constant(DNi[3]), Constant(DNi[4]), Constant(DNi[5]), 10.7)*grad(S_ni), grad(C_mid_ni)))*dx
    F_mn = S_mn*(C1_mn-C0_mn)*dx + dT*(inner(D(C_mid_fo, T_i, P, lnfO2, Constant(DMn[0]), Constant(DMn[1]), Constant(DMn[2]), Constant(DMn[3]), Constant(DMn[4]), Constant(DMn[5]), 6.0)*grad(S_ni), grad(C_mid_mn)))*dx

    a_ni = lhs(F_ni)
    L_ni = rhs(F_ni)
    a_mn = lhs(F_mn)
    L_mn = rhs(F_mn)
    
    # Define boundary condtions - Dirichlet at left boundary Neumann at right
    Cbc0_fo = DirichletBC(Q, Constant(rim_comp_fo), left_boundary) 
    Cbcs_fo = [Cbc0_fo]
    
    Cbc0_ni = DirichletBC(Q, Constant(rim_comp_ni), left_boundary) 
    Cbcs_ni = [Cbc0_ni]

    Cbc0_mn = DirichletBC(Q, Constant(rim_comp_mn), left_boundary) 
    Cbcs_mn = [Cbc0_mn]

    # Initial Conditions    
    if IC_i == True:
        # Use imported initial conditions
        Cinit_fo = ICs_import(dist_init, inicon['Fo'].values, xs, rim_comp_fo, Q)
        C0_fo.vector()[:] = Cinit_fo
        Cinit_ni = ICs_import(dist_init, inicon['Ni'].values, xs, rim_comp_ni, Q)
        C0_ni.vector()[:] = Cinit_ni
        Cinit_mn = ICs_import(dist_init, inicon['Mn'].values, xs, rim_comp_mn, Q)
        C0_mn.vector()[:] = Cinit_mn
    else:
        # Use constant initial conditions
        Cinit_fo = Constant(core_comp_fo)
        C0_fo.interpolate(Cinit_fo)
        Cinit_ni = Constant(core_comp_ni)
        C0_ni.interpolate(Cinit_ni)
        Cinit_mn = Constant(core_comp_mn)
        C0_mn.interpolate(Cinit_mn)

    # Timestepping
    i = 0
    
    while i < t:
        solve(a_ni==L_ni, C0_ni, Cbcs_ni)
        solve(a_mn==L_mn, C0_mn, Cbcs_mn)
        solve(F_fo==0, C1_fo, Cbcs_fo)
        C0_fo.assign(C1_fo)
        
        
        i += dt # Need to decide on time increment
        
# Convert FEniCS output into numpy array interpolated at observation distances        
    rcoords = mesh.coordinates()
        
    Fo_mod = modc(C0_fo.vector().get_local(), rcoords, dist)
    Ni_mod = modc(C0_ni.vector().get_local(), rcoords, dist)
    Mn_mod = modc(C0_mn.vector().get_local(), rcoords, dist)
    
    
# Create Global model output by combining individual elements
    gl_mod = np.append(Fo_mod, [Ni_mod, Mn_mod])
    
    return gl_mod
    
########################################################################################################################################
    
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
    
    f_dat = sys.argv[1]
    f_melt = sys.argv[2]
    f_modpar = sys.argv[3]
    f_angles = sys.argv[4]
    #f_err = sys.argv[6]
    f_mk = sys.argv[5]
    f_out = sys.argv[6]
    
    if len(sys.argv) == 8:
        f_inicon = sys.argv[7]
        inicon = pd.read_csv(f_inicon) #np.genfromtxt(f_inicon,delimiter=',',dtype=None, names=True)
        dist_init = inicon['Distance'].values
        IC_i = True
    else:
        IC_i = False
        
    
    obs = pd.read_csv(f_dat) # Import analytical model comps
    
    mc = np.genfromtxt(f_melt,delimiter=',',dtype=None, names=True) #Melt composition
    obs_mpar = np.genfromtxt(f_modpar,delimiter=',',dtype=None, names=True) # Model parameters
    obs_angle = np.genfromtxt(f_angles,delimiter=',',dtype=None, names=True) # Angles
    
    #ele_err = np.genfromtxt(f_err,delimiter=',',dtype=None, names=True)
    df_mk = pd.read_csv(f_mk) # Model markers
    
    # Import vcov files
    
    # Import covariance matrices for elements
    fo_all_vcov = pd.read_csv('./D_vcov/FeMg_all_001_vcov.csv') 
    fo_TAMED_vcov = pd.read_csv('./D_vcov/FeMg_TAMED_001_vcov.csv')
    ni_vcov = pd.read_csv('./D_vcov/Ni_wNi73ss_vcov.csv')
    mn_vcov = pd.read_csv('./D_vcov/Mn_all_vcov.csv')
    ca100_vcov = pd.read_csv('./D_vcov/Ca_100_vcov.csv')
    ca010_vcov = pd.read_csv('./D_vcov/Ca_010_vcov.csv')
    ca001_vcov = pd.read_csv('./D_vcov/Ca_001_vcov.csv')
    

    # Create global arrays for observations and errors
    gl_obs = np.append(obs['XFo'].values, [obs['Ni'].values, obs['Mn'].values]) 
    gl_err = np.append(obs['Fo_stdev'].values, [obs['Ni_stdev'].values, obs['Mn_stdev'].values])
    
    # Profile distances
    dist = obs['Distance'].values
    
    melt_comp = mc['Composition']
    
    psi, phi, gamma = m.radians(obs_angle['angle100P']), m.radians(obs_angle['angle010P']), m.radians(obs_angle['angle001P'])
    
    rim_comp_fo, core_comp_fo = df_mk['Fo_markers'][0], df_mk['Fo_markers'][1]
    rim_comp_ni, core_comp_ni = df_mk['Ni_markers'][0], df_mk['Ni_markers'][1]
    rim_comp_mn, core_comp_mn = df_mk['Mn_markers'][0], df_mk['Mn_markers'][1]
    
    n, L = 299, max(dist)
    # check for output
    dir = '/'.join(f_out.split('/')[:-1])
    if not os.path.exists(dir):
        print("Making directory for output:", dir)
        os.makedirs(dir)
        
    # PARAMETER SETUP-------------------------------------------------
    # number of weight parameters (melt region sections = N(weights) + 1)
    
    parameters = ["t", "T", "fe_3", "P", "lnD0_Fo", "clnfO2_Fo", "cXFo_Fo", "cT_i_Fo", "cP_Fo", "cT_iP_Fo", "lnD0_Ni", "clnfO2_Ni", "cXFo_Ni", "cT_i_Ni", "cP_Ni", "cT_iP_Ni","lnD0_Mn", "clnfO2_Mn", "cXFo_Mn", "cT_i_Mn", "cP_Mn", "cT_iP_Mn"]
    pti = np.empty([len(parameters)])
    pti[:] = np.nan
    pti[4:10], pti[10:16], pti[16:] = 0, 1 , 2

    
    cov_fo = fo_TAMED_vcov.as_matrix(columns=['(Intercept)', 'lnfO2', 'XFo', 'T_i', 'P_Pa', 'T_i:P_Pa'])
    cov_ni = ni_vcov.as_matrix(columns=['(Intercept)', 'lnfO2', 'XFo', 'T_i', 'P_Pa', 'T_i:P_Pa'])
    cov_mn = mn_vcov.as_matrix(columns=['(Intercept)', 'lnfO2', 'XFo', 'T_i', 'P_Pa', 'T_i:P_Pa'])
    
    cov_s = np.array([cov_fo, cov_ni, cov_mn])
    #==============================================================================
    # for i in Ds.index.values:
    #     parameters.append(i)
    #==============================================================================
    n_dim = len(parameters)
    n_params = n_dim
        
    # MCMC-------------------------------------------------
    # setup prior cube
    tprior = str(obs_mpar['tprior'][0])
    #type of distribution LU = ln uniform, U = uniform, MG= multivariate gaussian
    ptype = ["MG"] * n_dim
    ptype[0:1] = ["LU"]   
    ptype[1:4] = ["G"]*3
        
    pcube = np.full((n_dim,2), np.nan)
    pcube[0,:] = [2.0, 5.0]  # time (days)
    pcube[1,:] = [1190.0, 30.0] # Temperature 
    pcube[2,:] = [0.15, 0.02] # fe3/fet
    pcube[3,:] = [3.5, 1.4] # P (kbar)
    pcube[4:10,0] = np.array([-6.755, 2.244e-1, -7.181, -2.674e4, -5.213e-10, -1.028e-7]) # DFo
    pcube[10:16,0] = np.array([-1.109e1, 2.769e-1, -2.185, -2.508e4, -1.246e-9, 9.967e-7]) # DNi 
    pcube[16:,0] = np.array([-7.548, 1.963e-1, -7.153, -2.672e4, -9.504e-10, 7.195e-7]) #DMn
    
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
        bf_params = result.get_best_fit()['parameters']
        #print(bf_params)
        
        # extract best fit parameter for time and rerun in model, then plot up
        
        C_bf = mod_diff(bf_params, len(bf_params))
        C_bf_spl  = np.split(C_bf, 3) 
        C_Fo_bf, C_Ni_bf, C_Mn_bf = C_bf_spl[0], C_bf_spl[1], C_bf_spl[2]
        #C_Fo_bf = C_bf
        
        if IC_i == True:
            inicon_fo = inicon['Fo'].values
            inicon_ni = inicon['Ni'].values
            inicon_mn = inicon['Mn'].values
            disti = dist_init
        
        else:
            inicon_fo = np.ones(len(obs['XFo'].values))*core_comp_fo
            inicon_ni = np.ones(len(obs['Ni'].values))*core_comp_ni
            inicon_mn = np.ones(len(obs['Mn'].values))*core_comp_mn
            disti = dist
        
        # data fit
        fig, axes = plt.subplots(3,1)
        
        axes[0].errorbar(dist, obs['XFo'].values, yerr = obs['Fo_stdev'].values,fmt='o', color='red', label='data')
        axes[0].plot(disti, inicon_fo, color = 'black')
        axes[0].plot(dist, C_Fo_bf, color = 'blue')
        axes[0].set_ylabel('XFo')
        axes[1].errorbar(dist, obs['Ni'].values, yerr = obs['Ni_stdev'].values,fmt='o', color='red', label='data')
        axes[1].plot(disti, inicon_ni, color = 'black')
        axes[1].plot(dist, C_Ni_bf, color = 'blue')
        axes[1].set_ylabel('Ni (ppm)')
        axes[2].errorbar(dist, obs['Mn'].values, yerr = obs['Mn_stdev'].values,fmt='o', color='red', label='data')
        axes[2].plot(disti, inicon_mn, color = 'black')
        axes[2].plot(dist, C_Mn_bf, color = 'blue')
        axes[2].set_ylabel('Mn (ppm)')
        #plt.errorbar(dist, obs['XFo'].values, yerr = obs['Fo_std'].values,fmt='o', color='red', label='data')
        #plt.plot(dist, C_Fo_bf, color = 'blue')
        
        
        plt.savefig(f_out + 'fit.png')
        plt.close()

        # WRITE-------------------------------------------------

        pd.DataFrame({'Fo_bf': C_Fo_bf,'Ni_bf': C_Ni_bf,'Mn_bf': C_Mn_bf}).to_csv(f_out + 'mod_cv.csv', sep=',')

        
        
        



