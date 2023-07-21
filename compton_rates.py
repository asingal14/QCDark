import numpy as np
import scipy as sp
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
eVSqInv2barn = 2.56819e-15                                  # 1 barn in eV^-2

"""incoming photon frequency for comparison with RIA"""
omega0 = 1.461e6

"""Make Plots Pretty!"""
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

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
          self.dark_Rvec = data['run_settings/Rvec'][...].copy()
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

"""Read Input File"""
def read_output(fileName):
     return form_factor(fileName)

def compton_rates(form_factor, omega = omega0, maxq = None, averaging = None):

     def cos(q, E0, omega = omega0):
          numerator = (omega - E0)**2 + omega**2 - q**2
          denominator = 2*(omega - E0)*omega
          return numerator/denominator
    
     def limits(q, E0, omega = omega0):
          arr = np.zeros(q.shape)
          nx = np.where((q - E0 > 0)*(2*omega0 - E0 - q > 0))[0]
          arr[nx] = np.ones(nx.shape)
          return arr

     def q_integrand(FF_E0, q, E0, omega = omega0):
          cos_th = cos(q, E0, omega = omega)
          lim = limits(q, E0, omega = omega)
          val = lim*FF_E0*(1 + cos_th**2)/(qarr**2)
          return val

     def diff_cs(FF, qarr, Earr, omega = omega0):
          rates = np.array([])
          prefactor = 2*np.pi*(alpha**3)/(omega**2)
          for E0, FF_E0 in zip(Earr, FF.transpose()):
               val = sp.integrate.simps(q_integrand(FF_E0, qarr, E0, omega = omega), x = qarr)
               rates = np.append(rates, val)
          return prefactor*rates/eVSqInv2barn

     def averager(Earr, rates, DE, dE):
          num = int(round(DE/dE))
          newE, newr = np.array([]), np.array([])
          maxN = int(rates.shape[0]/num)
          for i in range(maxN):
               start, end = i*num, (i+1)*num
               tempE, tempR = np.mean(Earr[start:end]), np.mean(rates[start:end])
               newE = np.append(newE, tempE)
               newr = np.append(newr, tempR)
          return newE, newr

     dq, dE, FF = form_factor.dq, form_factor.dE, form_factor.ff
     qarr, Earr = dq*np.arange(FF.shape[0]) + 0.5*dq, dE*np.arange(FF.shape[1]) + 0.5*dE

     if maxq != None:
          truth = qarr/BohrInv2eV < maxq
          qarr, FF = qarr[truth], FF[truth]
     
     rates = diff_cs(FF, qarr, Earr, omega = omega)

     if averaging != None:
          Earr, rates = averager(Earr, rates, averaging, dE)

     return Earr, rates

def sum_rule(form_factor, averaging = None):

     def E_integrand(E, q0, FF0):
          return E/(q0**5)*FF0

     def diff_sr(qarr, Earr, FF, VCell):
          rates = np.array([])
          for q0, FF0 in zip(qarr, FF):
               rates = np.append(rates, sp.integrate.simps(E_integrand(Earr, q0, FF0), x = Earr))
          return 8.*((np.pi*alpha*mElectron)**2)*rates/VCell

     def averager(qarr, rates, Dq, dq):
          num = int(round(Dq/dq))
          newE, newr = np.array([]), np.array([])
          maxN = int(rates.shape[0]/num)
          for i in range(maxN):
               start, end = i*num, (i+1)*num
               tempE, tempR = np.mean(qarr[start:end]), np.mean(rates[start:end])
               newE = np.append(newE, tempE)
               newr = np.append(newr, tempR)
          return newE, newr

     dq, dE, FF = form_factor.dq, form_factor.dE, form_factor.ff
     qarr, Earr = dq*np.arange(FF.shape[0]) + 0.5*dq, dE*np.arange(FF.shape[1]) + 0.5*dE
     rates = diff_sr(qarr, Earr, FF, form_factor.VCell)

     if averaging != None:
          qarr, rates = averager(qarr, rates, averaging, dq/BohrInv2eV)

     return qarr/BohrInv2eV, rates

def sum_rule_bound(form_factor):
     return 16.*(np.pi**2)*alpha/mElectron/form_factor.VCell