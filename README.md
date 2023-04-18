# QCDark
QCDark can compute dark matter--electron scattering rates in crystals, and is built on top of the pyscf library.

## Requirements
1. python 3.6+
2. pyscf 2.0+
3. h5py
4. multiprocessing

## Using QCDark
Input parameters in ```input_parameters.py``` and run
```
python crystal_form_factor.py
```
This will produce a file containing the crystal form factor, eq. (13). Next, run an ipython notebook or jupyter notebook for post-processing.
```
from dark_matter_rates import *
cff = read_output(file_name)
Ee, dR = d_rate(mass_darkmatter, cff, FDM_exp = 0, screening = default_screening, astro_model = default_astro)
```
Inbuilt functions for secondary ionization: e-h pair production energy with input energy, Ramanthan & Kurinsky (2020) ionization with support for input file.
