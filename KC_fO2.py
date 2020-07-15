#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:29:37 2017

@author: ejfm2
"""
import math as m
import numpy as np


# Function to calculate oxygen fugacity using eqn 7 of Kress and Carmichael (1991)

def fO2calc_eq7(comp, T, P, fe3_fet):
    # Composition array in oxide wt% - need to convert to mol fraction
    # Comp array in order SiO2 TiO2 Al2O3 FeOt, MnO, MgO, CaO, Na2O, K2O, P2O5
    
    ele_mass = np.array([60.09, 79.9, 101.96128, 71.8464, 70.938, 40.3, 56.0794, 61.97894, 94.196, 141.948]) 
 

    moles = (comp/ele_mass)
    mol_sum = np.sum(moles)
    mol_frac = moles/mol_sum 
        
    X_Al2O3 = mol_frac[2]
    X_FeOt = mol_frac[3]
    X_CaO = mol_frac[6]
    X_Na2O = mol_frac[7]
    X_K2O = mol_frac[8]

    X_FeO1_5 = X_FeOt*fe3_fet
    X_FeO = X_FeOt - X_FeO1_5
    X_Fe2O3 = X_FeO1_5/2.0


   # Regression parameters from Kress and Carmichael 
    a = 0.196
    b = 1.1492e4
    c = -6.675
    d_Al2O3 = -2.243
    d_FeOt = -1.828
    d_CaO = 3.201
    d_Na2O = 5.854
    d_K2O = 6.215
    e = -3.36
    f = -7.01e-7 # K Pa-1
    g = -1.54e-10 # Pa-1
    h = 3.85e-17 # K Pa-2
    
    T0 = float(1673) #K
    
    Sum_ele = (d_Al2O3*X_Al2O3) + (d_FeOt*X_FeOt) + (d_CaO*X_CaO) + (d_Na2O*X_Na2O) + (d_K2O*X_K2O)
    
    lnfO2 = (m.log((X_Fe2O3/X_FeO)) - ((b/T) + c + Sum_ele + (e*(1-(T0/T)-m.log(T/T0))) + f*(P/T) + g*(((T-T0)*P)/T) + h*((P**2)/T)))/a
    
    return lnfO2
    
    
def fO2calc_eq6CAFS(comp, T, fe3_fet):
    # Composition array in oxide wt% - need to convert to mol fraction
    # Comp array in order SiO2 TiO2 Al2O3 FeOt, MnO, MgO, CaO, Na2O, K2O, P2O5
    
         
    ele_mass = np.array([60.09, 79.9, 101.96128, 71.8464, 70.938, 40.3, 56.0794, 61.97894, 94.196, 141.948]) 

    moles = (comp/ele_mass)
    mol_sum = np.sum(moles)
    mol_frac = moles/mol_sum  
        
    X_SiO2 = mol_frac[0]
    X_Al2O3 = mol_frac[2]
    X_CaO = mol_frac[6]
    X_FeOt = mol_frac[3] 

    X_FeO1_5 = X_FeOt*fe3_fet
    X_FeO = X_FeOt - X_FeO1_5
    X_Fe2O3 = X_FeO1_5/2.0 
 
    # Regression parameters from Kress and Carmichael 
    a = 0.207
    b = 1.298e4
    c = -6.115
    d_SiO2 = -2.368
    d_Al2O3 = -1.622
    d_CaO = 2.073

    Sum_ele = (d_SiO2*X_SiO2) + (d_Al2O3*X_Al2O3) + (d_CaO*X_CaO) 
    
    lnfO2 = (m.log((X_Fe2O3/X_FeO)) - ((b/T) + c + Sum_ele))/a
    
    return lnfO2  
    

def fO2calc_eq6NAFS(comp, T, fe3_fet):
    # Composition array in oxide wt% - need to convert to mol fraction
    # Comp array in order SiO2 TiO2 Al2O3 FeOt, MnO, MgO, CaO, Na2O, K2O, P2O5
    
        
    ele_mass = np.array([60.09, 79.9, 101.96128, 71.8464, 70.938, 40.3, 56.0794, 61.97894, 94.196, 141.948]) 
    X_SiO2 = mol_frac[0]
    X_Al2O3 = mol_frac[2]
    X_Na2O = mol_frac[7]
    X_FeOt = mol_frac[3]
 
    X_FeO1_5 = X_FeOt*fe3_fet
    X_FeO = X_FeOt - X_FeO1_5
    X_Fe2O3 = X_FeO1_5/2.0
 
    # Regression parameters from Kress and Carmichael 
    b = 1.678e4
    c = -9.266
    d_SiO2 = 0.312
    d_Al2O3 = -0.698
    d_Na2O = 2.065
    
    
    Sum_ele = (d_SiO2*X_SiO2) + (d_Al2O3*X_Al2O3) + (d_Na2O*X_Na2O) 
    
    lnfO2 = (m.log((X_Fe2O3/X_FeO)) - ((b/T) + c + Sum_ele))
    
    return lnfO2 

    
