#!/bin/bash

Profiles="HOR_2_OL_C28_P1 HOR_3_OL_C3_P2 HOR_3_OL_C5_P2 HOR_3_OL_C10_P2 HOR_3_OL_C11_P2 HOR_3_OL_C12_P2 HOR_3_OL_C13_P2 HOR_3_OL_C15_P2 HOR_3_OL_C16_P2 SKU_1_OL_C1_P4 SKU_1_OL_C2_P3 SKU_1_OL_C3_1_P4 SKU_1_OL_C3_2_P2 SKU_1_OL_C3_3_P3 SKU_1_OL_C3_4_P3 SKU_1_OL_C4_1_P4 SKU_2_OL_C8_P1 SKU_2_OL_C19_P1 SKU_4_C1_1_OL_P2 SKU_4_C3_1_OL_P2 HOR_1_OL_C1_P3 HOR_1_OL_C2_P3 HOR_1_OL_C3_P3 HOR_1_OL_C4_P3 HOR_2_OL_C6_P1 HOR_2_OL_C12_P1 HOR_2_OL_C15_P1 HOR_2_OL_C18_P1 HOR_2_OL_C19_P1 HOR_2_OL_C25_P1"

for x in ${Profiles} ; do

# copy files

cp ./KC_fO2.py ./${x}_pyMN/KC_fO2.py

cp ./1D_ol_diffusion_MCMC_v8_fenics_ME_MPI_adj.py ./${x}_pyMN/1D_ol_diffusion_MCMC_v8_fenics_ME_MPI_adj.py 

cp ./multinest_marginals.py ./${x}_pyMN/multinest_marginals.py 

cp ./pmc.py ./${x}_pyMN/pmc.py

cd ./${x}_pyMN

echo "${x}"

# CHAINS_4
f_dat="${x}_XFo_filt_err.csv"
f_melt="${x}_meltcomp.csv"
f_modpar="SKU_modpar_lu.csv"
f_angles="${x}_angles.csv"
f_inicon="${x}_inicon_all.csv"
f_mk="${x}_IC_markers_all.csv"
f_out='chains_Al_lu_MPI_redo'
#mv "$f_out"/* ~/.trash
 
mpiexec -n 20 python3 1D_ol_diffusion_MCMC_v8_fenics_ME_MPI_adj.py $f_dat $f_melt $f_modpar $f_angles $f_mk $f_out"/vars_1" $f_inicon

python3 multinest_marginals.py $f_out/vars_1 

cd ..

done




