#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QCDark branch to calculate the dielectric function:
     RPA Dielectric function based on atomic centered cartesian gaussian orbitals using PySCF as DFT code.
     version: 0.01

Implementation details:
     Current facilities available: 
          None.
     Implementation procedure:
          1. Switch from a 3-dimensional Etab formulation to 1-dimensional E-tab formulation
               1.1 Build all atomic orbitals such that one can construct each orbital from just the relevant primitive gaussians.
               1.2 Use primitive gaussians as indexed items. 
               1.3 

          . Switch from contracted gaussians to primitive gaussian-based calculations
               .1 Change cartesian_moments.py code to build 1-dimensional integrals over primitive gaussians.
               .2 Test direct input times versus interpolation time-scales. 
"""