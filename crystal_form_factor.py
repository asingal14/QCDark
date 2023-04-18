#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Details: crystal form factor based on DFT using atomic centered gaussian orbitals with PySCF, v4.0
Created: Tue July 19, 2022 15:45:00 (GMT)

v0.0, v1.0, v2.0 by chengzhen
v3.0, v4.0 (current) by amansingal

relevant citation: <to be added>
"""

"""
Parallelization details:
    Loop over initial and final k-points.
        Loop over G points - save data as tmp_file.h5py
            Parallelize over atomic orbitals            
"""

#%% Part 0: Import packages, create log file & patch (for python versions 3.3 - 3.7) bpo-17560 connection send/receive max size update

import os
import time
import math
import numpy as np
import itertools
import pdb
import logging
import h5py

from pyscf import gto
import pyscf.lib
import pyscf.pbc.gto as pbcgto

import input_parameters as parmt
import routines as routines

pyscf.lib.misc.num_threads(n = parmt.max_cores)

##==== Create log file, start counting time ============
logging.basicConfig(filename=parmt.logname, level=logging.INFO, format='%(message)s') 
#filemode='w' to start new log file
start_time = time.time()


##==== Apply PR-10305 / bpo-17560 connection ===========
##==== send/receive max size update for python =========
##==== versions 3.3-3.7 ================================
routines.patch()

#%% Part 1: Constants, conversion factors & build the periodic system
##==== Constants in the whole calculation ==============
c = 299792458 #c in m/s
me = 0.51099895000*10**6 #eV/c^2
alpha = 1/137

##==== Conversion factors ==============================
har2ev = 27.211386245988 #convert energy from hartree to eV
p2ev = 1.99285191410*10**(-24)/(5.344286*10**(-28)) #convert momentum from a.u. to eV/c
bohr2m = 5.29177210903*10**(-11) #convert Bohr radius to meter
a2bohr = 1.8897259886 #convert Å to Bohr radius
hbarc = 0.1973269804*10**(-6) #hbarc in eV*m

##==== Build cell with chosen basis set ================
cell = pbcgto.M(
    a = np.asarray(parmt.lattice_vectors),
    atom = parmt.atomloc,
#    basis = {parmt.myatom: gto.parse(bset.basis_sets[parmt.mybasis])},
    basis = parmt.mybasis,
    cart = True,
    verbose = parmt.outlev,
    output = parmt.pyscf_outfile,
    ecp = parmt.effective_core_potential,
    rcut = parmt.rcut,
    precision = parmt.precision
)

cart_labs = cell.cart_labels() #list of contracted Cartesian Gaussian orbitals
nbandstot = len(cart_labs)

logging.info('Built periodic system {}, check PySCF output for lattice vectors and atom locations.'\
             .format(parmt.system_name))
logging.info('Number of atomic orbitals(Cartesian) = {}.'.format(nbandstot))

#%% Part 2: Preparation of the calculation of crystal form factor, and perform DFT

#Build temporary directory & check if stored parameters are consistent with current input
routines.makedir(parmt.mid_step_folder)
routines.check_stored_data(cell)

##==== Total valence and conduction bands ==============
nvaltot = routines.getbandindices(cell)

##==== Create k-points grid ============================
kpts, kweights, nktot = routines.gen_kpts(cell)

##==== Specify parameters of the calculation ===========
'''
runsettings: dictionary of the parameters of the all-electron calculation
keys: "method": 'gpdft' = gamma-point dft, 'kptsdft' = k-points dft
      "cell": pyscf pbc Cell object
      "kpts": k-points in 1/Bohr, 2d array of shape (nktot,3)
      "df": density fitting method, 'test' or 'MDF'
      "xcfunc": exchange-correlation functional, 'exchange,correlation'
'''
runsettings = {"method":parmt.method, "cell":cell, "kpts":kpts, \
               "densityfit":parmt.densityfit, "xcfunc": parmt.xcfunc}

##==== Load band energy levels =========================
#energies: band energy levels, 2d array of shape (nktot,nbandstot)
#Eik = energies[ik1,ival], Ei'k' = energies[ik2,icon]
energies = routines.loadenergy(runsettings)

##==== Convert energies from hartree to eV and apply scissor correction
energies = routines.scissorcorrection(energies, nvaltot)

#%% Part 3: Preparation for calculation of crystal form factor

##==== Create R vectors and Hermite Gaussian Polynomials ======
routines.create_Rvecs_Etab(cell)

##==== Create G-vectors that satisfies the cutoff ======
routines.calculate_Gvectors(cell)

##==== Create and store prefactor term =================
routines.getprefactor(cell)

##==== Calculate the crystal form factor ===============
f2 = routines.calculate_parallel(cell, kweights)

##==== Store all results ===============================
routines.final_file(cell, f2)
logging.info('Crystal form factor calculation saved to ' + parmt.ff_file + '. Total time taken = {:.2f} s.'.format(time.time() - start_time))