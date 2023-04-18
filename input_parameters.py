#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 23:32:02 2021

input file that includes all paramters to be read 
by the main program 'cyrstal form factor'

parameters can be defined by user

v1.0

@author: chengzhen
"""
import os
##==== Build the Periodic System ====
system_name = 'Si' #name of atom
myatomnum = 14 #atomic number of the atom
lattice_vectors = [[2.715, 2.715, 0. ], [0., 2.715, 2.715], [2.715, 0., 2.715]]
#primitive lattice vectors of the unit cell, [[ax,ay,az],[bx,by,bz],[cx,cy,cz]], in unit Å
atomloc =  '''Si	0.	0.	0.     
		Si	1.3575	1.3575	1.3575'''
#positions of the atoms in the unit cell, in Å
mybasis = 'TZP' #name of basis set in PySCF
#mybasis = os.getcwd() + '/cc-pvtz.dat'
#if using an input file, use: mybasis = os.getcwd()+'/'+<filename>
effective_core_potential = None #name of ECP in PySCF
#if using an input file, use: effective_core_potential = os.getcwd()+'/'+<filename>

##==== Scissor Correction ============
do_scissor = True
expt_BG = 1.11 #in eV

##==== Will you repeat the calculation with a different XC or density fit? 
# If yes, do you have quite a bunch of storage (in multiples of TBs) available to speed up the next calculation?
save_Smat = False	# Switch to False to lower storage use, however will have to redo the slowest part of calculation if repeating with other XC functional.
save_temp_f2 = True # High storage usage, however lower RAM usage.
numerical = True	# Numerical calculation of scattering matrix in AO basis

##==== Create q and Ee bins ==========
qmin_ame = 0.0 #minimum value of q, in alpha*me
dqbin_ame = 0.02 #width of q bins, in alpha*me
nqbins = 500 #number of q bins

emin = 0.0 #minimum value of Ee, in eV
debin = 0.1 #width of Ee bins, in eV
nebins = 500 #number of Ee bins

##==== Number of bands to include =====
numval = 'all' #number of valence bands to include in the calculation, use 'all' for all available valence bands
numcon = 'all' #number of conduction bands to include in the calculation, use 'all' for all available conduction bands

##==== Settings for the all-electron calculation====
method = 'kptsdft' 
#method of the all-electron calculation, 
#'gpdft' = gamma-point dft, 'kptsdft' = k-points dft
densityfit = 'MDF' #density fitting method, 'MDF' = mixed density fitting, 'test' for gaussian density fitting.
xcfunc = 'pbe' #exchange-correlation functional, in format 'exchange,correlation'

##==== k-points grid ==================
#if method is 'gpdft', then should put nkx = 1, nky = 1, nkz = 1
nkx = 4 #number of k-points for x-axis
nky = 4 #number of k-points for y-axis
nkz = 4 #number of k-points for z-axis

##==== Cutoff of R vector ============
Rvec_cut = 4 #in unit of |a|, 1 is nearest neighbor

##==== Computation parameters ========
max_cores = 40
max_memory = 24

##==== crystal form factor calculation
##==== parameters and specifications. 
##==== No need to change anything. ===
logname = system_name + '_dark.log'
ff_file = system_name + '_f2.h5py'
mid_step_folder = system_name + '_temp'	# Need this: allows us to keep heavy objects off memory, as well as allows us to resume if calculation stopped at any point.
multibasis = False

##==== pyscf & logging parameters ====
outlev = 4 #output level
rcut = 42.5
precision = 5e-9
pyscf_outfile = system_name + '_pyscf.log' #output file for pyscf