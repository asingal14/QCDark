

import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as clb
import matplotlib.cm as cm
import h5py
import matplotlib.ticker
import pdb

"""Global Constants & Conversion"""
amu2eV = 9.315e8                                             # eV/u
lightSpeed = 299792.458                                     # In km/s
pi = np.pi
alpha = 1.0/137.03599908                                    # EM fine-structure constant at low E
mElectron = 5.1099894e5                                     # In eV
BohrInv2eV = alpha*mElectron                                # 1 Bohr^-1 to eV
Ryd2eV = 0.5*mElectron*alpha**2                             # In eV/Ryd
cm2sec = 1/lightSpeed*1e-5                                  # 1 cm in s
sec2yr = 1/(60.*60.*24*365.25)                              # 1 s in years
cmInv2eV = 50677.3093773                                    # 1 eV in cm^-1

"""Dark Matter astrophysical parameters"""
default_astro = {
     'v0': 238.,
     'vEarth': 250.2,
     'vEscape': 544.0,
     'rhoX': 0.3e9,
     'sigma_e': 1e-39
}
old_astro = {
     'v0': 230.,
     'vEarth': 240.,
     'vEscape': 600.0,
     'rhoX': 0.4e9,
     'sigma_e': 1e-39
}

"""Thomas-Fermi Screening Parameters"""
default_si = {
     'DoScreen': True,
     'eps0': 11.3,                                         # Unitless
     'qTF': 4.13e3,                                        # In eV
     'omegaP':  16.6,                                       # In eV
     'alphaS': 1.563,                                      # Unitless
}
default_ge = {
     'DoScreen': True,
     'eps0': 14.0,                                          # Unitless
     'qTF': 3.99e3,                                         # In eV
     'omegaP': 15.2,                                        # In eV
     'alphaS': 1.563                                       # Unitless
}
default_no_sreen = {
     'DoScreen': False
}
default_screening = default_no_sreen

"""Make Plots Pretty!"""
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

"""Read Input File"""
def read_output(fileName):
     return form_factor(fileName)

"""Reduced Mass"""
def reducedMass(mX):                                        # In eV
     return mX*mElectron/(mX + mElectron)

"""Integrated Maxwell-Boltzmann Distribution"""
def eta_MB(qArr, E, mX, astro_model):                                    # In units of c^-1
     vEscape, vEarth, v0 = astro_model['vEscape']/lightSpeed, astro_model['vEarth']/lightSpeed, astro_model['v0']/lightSpeed
     val = list()
     for q in qArr:
          vMin = q/(2.0*mX) + E/q
          if (vMin < vEscape - vEarth):
               val.append(-4.0*vEarth*np.exp(-(vEscape/v0)**2) + np.sqrt(pi)*v0*(sp.special.erf((vMin+vEarth)/v0) - sp.special.erf((vMin - vEarth)/v0)))
          elif (vMin < vEscape + vEarth):
               val.append(-2.0*(vEarth+vEscape-vMin)*np.exp(-(vEscape/v0)**2) + np.sqrt(pi)*v0*(sp.special.erf(vEscape/v0) - sp.special.erf((vMin - vEarth)/v0)))
          else:
               val.append(0.0)
     
     val = np.array(val)
     K = (v0**3)*(-2.0*pi*(vEscape/v0)*np.exp(-(vEscape/v0)**2) + (pi**1.5)*sp.special.erf(vEscape/v0))
     return (v0**2)*np.pi/(2.0*vEarth*K)*val


""""Calculate F_DM"""
def F_DM(q, FDM_exp):                                                # Unitless
     return (alpha*mElectron/q)**FDM_exp

"""Thomas-Fermi Screening"""
def TFscreening(q, E, screening):
     if screening['DoScreen']==True:
          eps0, alphaS, qTF, omegaP = screening['eps0'], screening['alphaS'], screening['qTF'], screening['omegaP']
          val = 1.0/(eps0 - 1) + alphaS*((q/qTF)**2) + q**4/(4.*(mElectron**2)*(omegaP**2)) - (E/omegaP)**2
          return 1./(1. + 1./val)
     else:
          return 1.0

"""Define the integrand to perform Simpson's rule"""
def momentum_integrand(dq, dE, q_index, E_index, mX, F_crystal2, FDM_exp, screening, astro_model):
     qArr, E = dq*q_index + dq/2.0, dE*E_index + dE/2.0
     return E/(qArr**2)*eta_MB(qArr, E, mX, astro_model)*(F_DM(qArr, FDM_exp)**2)*F_crystal2[q_index, E_index]*(TFscreening(qArr, E, screening)**2)

"""Integrate the integrand to find dR/d(ln E) for fixed E"""
def d_rate_fixedE(dq, dE, MCell, E_index, mX, F_crystal2, FDM_exp, screening, astro_model):
     rhoX, crosssection = astro_model['rhoX'], astro_model['sigma_e']
     prefactor = (rhoX/mX)*(5.609588e35/MCell)*crosssection*alpha*((mElectron/reducedMass(mX))**2)
     qiArr = np.arange(np.shape(F_crystal2)[0])
     return prefactor*sp.integrate.simps(momentum_integrand(dq, dE, qiArr, E_index, mX, F_crystal2, FDM_exp, screening, astro_model), x = dq*qiArr + dq/2.0)

"""Find dR/dE as a function of E"""
def d_rate(mX, form_factor, FDM_exp = 0, screening = default_screening, astro_model = default_astro):
     dq, dE, MCell, F_crystal2 = form_factor.dq, form_factor.dE, form_factor.mCell, form_factor.ff
     numE = np.shape(F_crystal2)[1]
     vals, Earr = np.array([]), np.arange(numE)*dE + dE/2.0
     for E_index in np.arange(numE):
          vals = np.append(vals, d_rate_fixedE(dq, dE, MCell, E_index, mX, F_crystal2, FDM_exp, screening, astro_model))
     return Earr, vals/cm2sec/sec2yr/Earr

"""total rate"""
def rate(mX, form_factor, FDM_exp = 0, screening = default_screening, astro_model = default_astro):
     dR, E = d_rate(mX, form_factor, FDM_exp=FDM_exp, screening = screening, astro_model = astro_model)
     return sp.integrate.simps(dR, x = E)

"""Calculate dR/dQ assuming E2Q generates 1 unit of charge"""
def d_rate_FanoQ(mX, form_factor, E2Q, FDM_exp = 0, screening = default_screening, astro_model = default_astro):
     dE, E_gap = form_factor.dE, form_factor.band_gap
     E, dR = d_rate(mX, form_factor, FDM_exp=FDM_exp, screening = screening, astro_model = astro_model)
     numE, initE, binE = E.shape[0], int(E_gap/dE), int(round(E2Q/dE))
     numQ, dRbins, Ebins = (numE - initE)//binE, np.array([0.0]), np.array([0.0])
     for i in range(numQ):
          Ebins = np.append(Ebins, (initE + i*binE)*dE)
          vals = sp.integrate.simps(dR[i*binE + initE: (i+1)*binE + initE], x = E[i*binE + initE: (i+1)*binE + initE])
          dRbins = np.append(dRbins, vals)
     return Ebins, dRbins

def d_rate_RamanathanQ(mX, form_factor, ionizationFile, FDM_exp = 0, screening = default_screening, astro_model = default_astro):
     
     dE = form_factor.dE
     Earr, dRarr = d_rate(mX, form_factor, FDM_exp=FDM_exp, screening = screening, astro_model = astro_model)
     
     ionization_inp = np.genfromtxt(ionizationFile).transpose()
     x = ionization_inp[0]
     min, max = x.min(), x.max()
     print('Input file has probabilities listed for {:.2f} eV <= E <= {:.2f} eV.\nAll rates outside this range will be ignored.'.format(min, max))

     Earr = np.round(Earr, 5)
     Earr, dRarr = Earr[Earr >= min], dRarr[Earr >= min]
     Earr, dRarr = Earr[Earr <= max], dRarr[Earr <= max]
     vals = []
     for y in ionization_inp[1:]:
          f = interp1d(x, y, kind = 'linear')
          vals.append(np.sum(dRarr*dE*f(Earr)))
     return np.arange(len(vals)+1), np.array([0] + vals)

def plot_cff(ax, form_factor, title = None, plt_pars = None, plt_qmin = True, astro_model = default_astro):
     dq, dE, FF = form_factor.dq, form_factor.dE, form_factor.ff

     if plt_pars == None:
          vmin = 1e-3
          vmax = 1e2
          cmap = cm.gnuplot2_r
     else:
          vmin = plt_pars[0]
          vmax = plt_pars[1]
          cmap = plt_pars[2]

     ax.imshow(FF, norm = colors.LogNorm(vmin = vmin, vmax = vmax), cmap = cmap, origin = 'lower')
     if plt_qmin:
          EE = np.linspace(0, dE*FF.shape[1])
          vEscape, vEarth = astro_model['vEscape']/lightSpeed, astro_model['vEarth']/lightSpeed
          qq = EE/(vEscape + vEarth)
          ax.plot(EE/dE, qq/dq, lw = 0.8, color = 'k', ls = '-')
     ax.set_xlim([0, FF.shape[1]])
     ax.set_ylim([0, FF.shape[0]])

     maxq = dq*FF.shape[0]/BohrInv2eV
     maxE = dE*FF.shape[1]
     numqticks, numEticks = int(np.round(maxq,0)+0.001), int(np.round(maxE/5,0)+0.001)
     xticks, yticks = np.arange(numEticks + 1)*FF.shape[1]/numEticks, np.arange(numqticks+1)*FF.shape[0]/numqticks
     xtick_labels, ytick_labels = np.round(dE*xticks, 0).astype('int').astype('str'), np.round(dq*yticks/BohrInv2eV, 0).astype('int').astype('str')

     if title != None:
          ax.set_title(title)

     ax.set_xticks(xticks)
     ax.set_yticks(yticks)
     ax.set_xticklabels(xtick_labels)
     ax.set_yticklabels(ytick_labels)
     ax.set_xlabel(r'$E_e$ [eV]')
     ax.set_ylabel(r'$q$ [$\alpha m_e$]')

     return

def plot_colorbar(cax, plt_pars = None, orientation = 'horizontal'):
     if plt_pars == None:
          vmin = 1e-3
          vmax = 1e2
          cmap = cm.gnuplot2_r
     else:
          vmin = plt_pars[0]
          vmax = plt_pars[1]
          cmap = plt_pars[2]
     norm = colors.LogNorm(vmin = vmin, vmax = vmax)
     clb.ColorbarBase(cax, norm = norm, cmap = cmap, orientation = orientation, extend = 'both')
     return


class form_factor(object):
     '''
     Class containing our form factor object, including data 
     regarding input.
     
     To access crystal form factor, please use
          form_factor.ff
     For other information, please view listed data.

     Early versions of results did not store if the energies
     had been scissor corrected to band gap, and so backwards
     compatibility requires allowing this to be skipped.
     '''

     def __init__(self, filename):
          data = h5py.File(filename, 'r')
          self.lattice = data['run_settings/a'][...].copy()
          self.atom = data['run_settings'].attrs['atom']
          self.basis = data['run_settings'].attrs['basis']
          self.ecp = data['run_settings'].attrs['ecp']
          self.dft_rcut = float(data['run_settings/rcut'][...])
          self.dft_precision = float(data['run_settings/precision'][...])
          try:
               self.dark_Rvec = data['run_settings/Rvec'][...].copy()
          except:
               pass
          self.num_con = data['run_settings'].attrs['numcon']
          self.num_val = data['run_settings'].attrs['numval']
          self.dft_xc = data['run_settings'].attrs['xc']
          self.dft_density_fitting = data['run_settings'].attrs['df']
          self.kpts = data['run_settings/kpts'][...].copy()
          self.VCell = data['results'].attrs['VCell']
          self.mCell = data['results'].attrs['mCell']
          self.dq = data['results'].attrs['dq']*BohrInv2eV
          self.dE = data['results'].attrs['dE']
          self.ff = data['results/f2'][...].copy()
          self.band_gap = data['results'].attrs['bandgap']
          try:
               self.scissor_corrected = data['results'].attrs['scissor']
          except:
               pass
          self.convert_atom()
          if self.ecp == 'None':
               self.ecp = None
          data.close()

     def convert_atom(self):
          string = self.atom.replace(';', '\n')
          atoms = string.split('\n')
          self.atom = np.asarray([atom.split() for atom in atoms])
