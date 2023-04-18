import numpy as np
from scipy.special import factorial2 as fact2
from scipy.special import gamma

def E(i,j,t,Qx,a,b):
    ''' Recursive definition of Hermite Gaussian coefficients.
        Returns a float.
        a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
        i,j: orbital angular momentum number on Gaussian 'a' and 'b'
        t: number nodes in Hermite (depends on type of integral, 
           e.g. always zero for overlap integrals)
        Qx: distance between origins of Gaussian 'a' and 'b'
    '''
    p = a + b
    q = a*b/p
    if (t < 0) or (t > (i + j)):
        # out of bounds for t  
        return 0.0
    elif i == j == t == 0:
        # base case
        return np.exp(-q*Qx*Qx) # K_AB
    elif j == 0:
        # decrement index i
        return (1/(2*p))*E(i-1,j,t-1,Qx,a,b) - \
               (q*Qx/a)*E(i-1,j,t,Qx,a,b)    + \
               (t+1)*E(i-1,j,t+1,Qx,a,b)
    else:
        # decrement index j
        return (1/(2*p))*E(i,j-1,t-1,Qx,a,b) + \
               (q*Qx/b)*E(i,j-1,t,Qx,a,b)    + \
               (t+1)*E(i,j-1,t+1,Qx,a,b)

def overlap(a,lmn1,A,b,lmn2,B):
    ''' Evaluates overlap integral between two Gaussians
        times a plane wave exp(iqx)  see eq. 108-109 in 
       "Helgaker, T., & Taylor, P. R. (1995). 
       GAUSSIAN BASIS SETS AND MOLECULAR INTEGRALS. 
       Advanced Series in Physical Chemistry, 725–856. 
       doi:10.1142/9789812832115_0001 "
       <\phi_a | exp(iqr) | \phi_b> with \phi_a,b being
       a pair of gaussians and q being a 3d 
       plane wave vector.
        Returns a float.
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
        lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
              for Gaussian 'a'
        lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
        A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
        B:    list containing origin of Gaussian 'b'
        q:    list containing the vector q
    '''
    l1,m1,n1 = lmn1 # shell angular momentum on Gaussian 'a'
    l2,m2,n2 = lmn2 # shell angular momentum on Gaussian 'b'
    S1 = E(l1,l2,0,A[0]-B[0],a,b) # X
    S2 = E(m1,m2,0,A[1]-B[1],a,b) # Y
    S3 = E(n1,n2,0,A[2]-B[2],a,b) # Z
    return S1*S2*S3*np.power(np.pi/(a+b),1.5) 

def overlapq(a,lmn1,A,b,lmn2,B,q):
    ''' Evaluates overlap integral between two Gaussians times a plane wave
        Returns a 2d complex array of dimension (N1,N2)
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
        lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
              for Gaussian 'a'
        lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
        A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
        B:    list containing origin of Gaussian 'b'
        q:    N1*N2*3 array containing the grid of the plane wave vector
              N1 is number of points on the theta_q grid, N2 is number of points on the phi_q grid
              N1*N2 is the total number of points on the q grid
              the 3 columns are x,y,z components of the plane wave, respectively
    '''
    l1,m1,n1 = lmn1 # shell angular momentum on Gaussian 'a'
    l2,m2,n2 = lmn2 # shell angular momentum on Gaussian 'b'
    p = a + b
    P = gaussian_product_center(a,A,b,B) # Gaussian composite center
    pwx = 1j * q[:,:,0]
    pwy = 1j * q[:,:,1]
    pwz = 1j * q[:,:,2]
    
    val = 0.0
    for t in range(l1+l2+1): #X
        val += E(l1,l2,t,A[0]-B[0],a,b) * np.power(pwx, t)
    Sq0 =  val * np.exp(pwx * (P[0] + pwx / (4*p)) )
    val = 0.0
    for t in range(m1+m2+1):#Y
        val += E(m1,m2,t,A[1]-B[1],a,b) * np.power(pwy, t)
    Sq1 =  val * np.exp(pwy * (P[1] + pwy / (4*p)) )
    val = 0.0
    for t in range(n1+n2+1):#Z
        val += E(n1,n2,t,A[2]-B[2],a,b) * np.power(pwz, t)
    Sq2 =  val * np.exp(pwz * (P[2] + pwz / (4*p)) )


    return Sq0*Sq1*Sq2*np.power(np.pi/p,1.5)


def overlapq_tab(a,lmn1,A,iexpa,b,lmn2,B,iexpb,dist_ind,q,Rvec,Etab):
    ''' Evaluates overlap integral between two Gaussians times a plane wave,
        with center of the first Gaussian shifted by lattice vector \vec{R}.
        Returns a 3d array of shape (Lk,LG,nR), dtype = complex
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
        lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
              for Gaussian 'a'
        lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
        A:    1d array containing origin of Gaussian 'a', e.g. array([0., 0., 0.])
        B:    1d array containing origin of Gaussian 'b'
        iexpa: index of the exponent of Gaussian 'a' in the list allexps
        iexpb: index of the exponent of Gaussian 'b' in the list allexps
        q:    2d array of shape (LG,3), containing the plane wave vector
              the 3 components of the last dimension are x,y,z components of 
              the plane wave, respectively
        dist_ind: index of the distance between the center of 'a' and 'b'
                  in the list 'dists'
        Rvec: primitive lattice vectors, 2d array of shape (nR,3)
    '''
    #try:
    #    Etab = np.load('Etab_v3.npy')
    #except FileNotFoundError:
    #    print('Etab file not found. Please run function routines.Gen_Etab.')
    l1,m1,n1 = lmn1 # shell angular momentum on Gaussian 'a'
    l2,m2,n2 = lmn2 # shell angular momentum on Gaussian 'b'
    p = a + b
    P = gaussian_product_center(a,A,b,B,Rvec) 
    #Gaussian composite center, 2d array of shape (3,nR)
    pwx = 1j * q[:,0]
    pwy = 1j * q[:,1]
    pwz = 1j * q[:,2]
    pwx = pwx[:,np.newaxis]
    pwy = pwy[:,np.newaxis]
    pwz = pwz[:,np.newaxis]
    d = [dist_ind,dist_ind+1,dist_ind+2]
    
    val = 0.0
    for t in range(l1+l2+1): #X
        val += Etab[l1,l2,t,:,d[0],iexpa,iexpb][np.newaxis,:] * np.power(pwx, t)
    Sq0 =  val * np.exp(pwx * (P[0,:][np.newaxis,:] + pwx / (4*p)) )
    val = 0.0
    for t in range(m1+m2+1):#Y
        val += Etab[m1,m2,t,:,d[1],iexpa,iexpb][np.newaxis,:] * np.power(pwy, t)
    Sq1 =  val * np.exp(pwy * (P[1,:][np.newaxis,:] + pwy / (4*p)) )
    val = 0.0
    for t in range(n1+n2+1):#Z
        val += Etab[n1,n2,t,:,d[2],iexpa,iexpb][np.newaxis,:] * np.power(pwz, t)
    Sq2 =  val * np.exp(pwz * (P[2,:][np.newaxis,:] + pwz / (4*p)) )


    return Sq0*Sq1*Sq2*np.power(np.pi/p,1.5)


def S(a,b):
    '''Evaluates overlap between two contracted Gaussians
       Returns float.
       Arguments:
       a: contracted Gaussian 'a', BasisFunction object
       b: contracted Gaussian 'b', BasisFunction object
    '''
    s = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            s += a.norm[ia]*b.norm[ib]*ca*cb*\
                     overlap(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin)
    return s

def Sq(a,b,q,dist_ind,Etab):
    '''Evaluates overlap between two contracted Gaussians times 
       a plane wave exp(iqx)  see eq. 108-109 in 
       "Helgaker, T., & Taylor, P. R. (1995). 
       GAUSSIAN BASIS SETS AND MOLECULAR INTEGRALS. 
       Advanced Series in Physical Chemistry, 725–856. 
       doi:10.1142/9789812832115_0001 "
       <\phi_a | exp(iqr) | \phi_b> with \phi_a,b being
       a pair of contracted gaussians and q being a 3d 
       plane wave vector.
       
       Returns a 2d complex array of dimension (N1,N2)
       Arguments:
       a: contracted Gaussian 'a', BasisFunction object
       b: contracted Gaussian 'b', BasisFunction object
       q: N1*N2*3 array containing the grid of the plane wave vector
          N1 is number of points on theta_q grid, N2 is number of points on phi_q grid
          N1*N2 is the total number of points on the angular q grid
          the 3 columns are x,y,z components of the plane wave, respectively
       dist_ind: index of the distance between the center of 'a' and 'b'
                 in the list 'dists'
    '''
    sq = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            sq += a.norm[ia]*b.norm[ib]*ca*cb*\
                     overlapq_tab(a.exps[ia],a.shell,a.origin,a.exps_ind[ia],
                     b.exps[ib],b.shell,b.origin,b.exps_ind[ib],dist_ind,q,Etab)
    return sq


def SqR(a,b,q,dist_ind,Rvec,Etab):
    '''Evaluates overlap between two contracted Gaussians times 
       a plane wave exp(iqr), with center of the first Gaussian 
       shifted by lattice vector \vec{R}
       <Ga(r-R)|e^{iqr}|Gb(r)> with \G_a,b being
       a pair of contracted gaussians and q being a 3d 
       plane wave vector.
       
       Returns a 3d array of shape (Lk,LG,nR), dtype = complex      
       Arguments:
       a: contracted Gaussian 'a', BasisFunction object
       b: contracted Gaussian 'b', BasisFunction object
       q: 2d array of shape (LG,3), containing the plane wave vector
          the 3 components of the last dimension are x,y,z components of 
          the plane wave, respectively
       dist_ind: index of the distance between the center of 'a' and 'b'
                 in the list 'dists'
       Rvec: primitive lattice vectors, 2d array of shape (nR,3), 
             where nR is the total number of R vectors
    '''
    sq = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            sq += a.norm[ia]*b.norm[ib]*ca*cb*\
                     overlapq_tab(a.exps[ia],a.shell,a.origin,a.exps_ind[ia],
                     b.exps[ib],b.shell,b.origin,b.exps_ind[ib],dist_ind,q,Rvec,Etab)
    return sq


class BasisFunction(object):
    ''' A class that contains all our basis function data
        Attributes:
        origin:   1d array containing the coordinates of the Gaussian origin
        shell:    tuple of angular momentum
        num_exps: number of primitive Gaussian exponents
        exps:     1d array of primitive Gaussian exponents
        coefs:    1d array of primitive Gaussian coefficients
        exps_ind: 1d array of the index of the exponets in the list allexps
        atm_ind:  atom id of the center of the Gaussian orbital, 0-based, int
        norm:     1d array of normalization factors for Gaussian primitives        
    '''
    def __init__(self,origin=[0.0,0.0,0.0],shell=(0,0,0),num_exps=0,exps=[],coefs=[],exps_ind=[],atm_ind=0):
        self.origin = np.asarray(origin)
        self.shell = shell
        self.num_exps = num_exps
        self.exps  = np.asarray(exps)
        self.coefs = np.asarray(coefs)
#        self.num_exps = len(self.exps)
        self.exps_ind = np.asarray(exps_ind)
        self.atm_ind = atm_ind
        self.norm = None
        self.normalize()

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


def gaussian_product_center(a,A,b,B,Rvec):
    ''' Calculate the new center point of the product of two Guassians by Guassian product rule,
        with the center of the first Gaussian shifted by \vec{R}
        Returns a 2d array of shape (3,nR)
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
        A:    1d array containing origin of Gaussian 'a', e.g. array([0., 0., 0.])
        B:    1d array containing origin of Gaussian 'b'
        Rvec: primitive lattice vectors, 2d array of shape (nR,3)
    '''   
    P = (a*(A+Rvec)+b*B)/(a+b)
    return P.T

