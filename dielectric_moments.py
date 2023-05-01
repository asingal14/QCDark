import numpy as np
from scipy.special import factorial2 as fact2
from scipy.special import gamma

class atomic_orbital(object):
     """
     Defines atomic orbital by taking in the 1D exponents array. 
     Inputs:
          prim_gauss_1d: array of shape (Sum (atom*prim_indices_in_atom), 5)
                              format of each row: [exponent, shell_1D, atomic_location_1dx, atomic_location_1dy, atomic_location_1dz]
          indices:       array of shape (N,3)
                              indices from prim_gauss_1D to input to build contracted 
                              cartesian gaussian
          coefficients:  array of shape (N,)
                              coefficient of each primitive gaussian
     Attributes:
          exponents:     array of exponents
          coefficients:  taken directly from input
          norm:          normalization of each exponent
          shell:         tuple of shape (3,), contains information about shell_1D in all three directions.
                              should be same for all input indices (verified in code)
          A:             location of atom
                              should match atomic_location of all prim_gauss_1d (verified in code)
          indices:       taken directly from input
                              useful for accessing prim_overlap_1d array.
     """

     def __init__(self, prim_gauss_1d, indices, coefficients):
          self.indices = indices
          self.coefficients = coefficients
          self.exponents = None
          self.get_exponents(prim_gauss_1d, indices)
          self.shell = None
          self.get_shell(prim_gauss_1d, indices)
          self.A = None
          self.norm = None
          self.normalize()

     def get_exponents(self, prim_gauss_1d, indices):
          exp_x = prim_gauss_1d[indices[:,0],0]
          exp_y = prim_gauss_1d[indices[:,1],0]
          exp_z = prim_gauss_1d[indices[:,2],0]
          if (np.round(exp_x, 5) == np.round (exp_y, 5)).prod() and (np.round(exp_x, 5) == np.round (exp_z, 5)).prod():
               self.exponents = exp_x
          else:
               raise Exception("Primitive gaussian exponents in construction of contracted gaussians do not match.\nPlease check code.")
     
     def get_shell(self, prim_gauss_1d, indices):
          shell_x = np.round(prim_gauss_1d[indices[:,0],1], 0).astype('int')
          shell_y = np.round(prim_gauss_1d[indices[:,1],1], 0).astype('int')
          shell_z = np.round(prim_gauss_1d[indices[:,2],1], 0).astype('int')
          r_x = all(shell_x == shell_x[0])
          r_y = all(shell_y == shell_y[0])
          r_z = all(shell_z == shell_z[0])
          if r_x*r_y*r_z:
               self.shell = (shell_x[0], shell_y[0], shell_z[0])
          else:
               raise Exception("Shells of all indices in construction of contracted gaussians do not match.\nPlease check code.")
     
     def get_A(self, prim_gauss_1d, indices):
          shell_x = np.round(prim_gauss_1d[indices[:,0],2], 6)
          shell_y = np.round(prim_gauss_1d[indices[:,1],3], 6)
          shell_z = np.round(prim_gauss_1d[indices[:,2],4], 6)
          r_x = all(shell_x == shell_x[0])
          r_y = all(shell_y == shell_y[0])
          r_z = all(shell_z == shell_z[0])
          if r_x*r_y*r_z:
               self.A = [shell_x[0], shell_y[0], shell_z[0]]
          else:
               raise Exception("Atomic Positions in construction of contracted gaussians do not match.\nPlease check code.")
     
     def normalize(self):
          ''' Routine to normalize the basis functions, in case they
              do not integrate to unity.
          '''
          l,m,n = self.shell
          L = l+m+n
          if L < 2:
          # self.norm is a list of length equal to number primitives
          # normalize primitives first (PGBFs)
               self.norm = np.sqrt(np.power(2,2*(l+m+n)+1.5)*
                         np.power(self.exps,l+m+n+1.5)/
                         fact2(2*l-1)/fact2(2*m-1)/
                         fact2(2*n-1)/np.power(np.pi,1.5))
                    # now normalize the contracted basis functions (CGBFs)
               # Eq. 1.44 of Valeev integral whitepaper
               prefactor = np.power(np.pi,1.5)*\
                    fact2(2*l - 1)*fact2(2*m - 1)*fact2(2*n - 1)/np.power(2.0,L)

               N = 0.0
               num_exps = len(self.exps)
               for ia in range(num_exps):
                    for ib in range(num_exps):
                    N += self.norm[ia]*self.norm[ib]*self.coefs[ia]*self.coefs[ib]/\
                              np.power(self.exps[ia] + self.exps[ib],L+1.5)

               N *= prefactor
               N = np.power(N,-0.5)
               for ia in range(num_exps):
                    self.coefs[ia] *= N
          else:
               self.norm = np.sqrt(np.power(2, L + 2.5)*
                                   np.power(self.exps, L + 1.5)/
                                   gamma(1.5 + L))