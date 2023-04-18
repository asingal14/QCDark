#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 19:12:37 2020
definition of functions to be called in the main program 'cyrstal form factor' v4.0
@author: chengzhen
@editor: amansingal
"""

import os, time, shutil, itertools, functools, logging, pdb, struct, sys, h5py
import numpy as np
import pyscf.pbc.dft as pbcdft
import cartesian_moments as cartmoments
import input_parameters as parmt
from multiprocessing import Pool
from functools import partial
import shutil

##==== Constants in the whole calculation ============
c = 299792458 #c in m/s
me = 0.51099895000*10**6 #eV/c^2
alpha = 1/137

##==== Conversion factors ============================
har2ev = 27.211386245988 #convert energy from hartree to eV
p2ev = 1.99285191410*10**(-24)/(5.344286*10**(-28)) #convert momentum from a.u. to eV/c
bohr2m = 5.29177210903*10**(-11) #convert Bohr radius to meter
a2bohr = 1.8897259886 #convert Å to Bohr radius
hbarc = 0.1973269804*10**(-6) #hbarc in eV*m
amu2eV = 9.315e8 # eV/u

logging.basicConfig(filename=parmt.logname, level=logging.INFO, format='%(message)s') 

def patch():
    """Apply PR-10305 / bpo-17560 connection send/receive max size update

    See the original issue at https://bugs.python.org/issue17560 and 
    https://github.com/python/cpython/pull/10305 for the pull request.

    This only supports Python versions 3.3 - 3.7, this function
    does nothing for Python versions outside of that range.

    """
    patchname = "Multiprocessing connection patch for bpo-17560"
    if not (3, 3) < sys.version_info < (3, 8):
        logging.info(
            patchname + " not applied, not an applicable Python version: %s",
            sys.version
        )
        return

    from multiprocessing.connection import Connection

    orig_send_bytes = Connection._send_bytes
    orig_recv_bytes = Connection._recv_bytes
    if (
        orig_send_bytes.__code__.co_filename == __file__
        and orig_recv_bytes.__code__.co_filename == __file__
    ):
        logging.info(patchname + " already applied, skipping")
        return

    @functools.wraps(orig_send_bytes)
    def send_bytes(self, buf):
        n = len(buf)
        if n > 0x7fffffff:
            pre_header = struct.pack("!i", -1)
            header = struct.pack("!Q", n)
            self._send(pre_header)
            self._send(header)
            self._send(buf)
        else:
            orig_send_bytes(self, buf)

    @functools.wraps(orig_recv_bytes)
    def recv_bytes(self, maxsize=None):
        buf = self._recv(4)
        size, = struct.unpack("!i", buf.getvalue())
        if size == -1:
            buf = self._recv(8)
            size, = struct.unpack("!Q", buf.getvalue())
        if maxsize is not None and size > maxsize:
            return None
        return self._recv(size)

    Connection._send_bytes = send_bytes
    Connection._recv_bytes = recv_bytes

    logging.info(patchname + " applied")

def makedir(dirname, log = True):
    """
    Function that tries to make a directory if it does not exist.
    """
    path = '.'
    try:
        os.mkdir(dirname)
        if log:
            logging.info('Made directory ' + dirname + '.')
    except FileExistsError:
        if os.path.isdir(dirname):
            if log:
                logging.info('Directory '+dirname+' already exists.')
        else:
            logging.info('File exists with name same as directory, ' + dirname + '. Cannot make directory, raising exception.')
            raise Exception('Cannot proceed with making directory ' + dirname + '.')
    return

def create_bins(min, delta, n):
    '''
    function to create q bins and Ee bins
    use q as example
    input: qmin: mininum value, qmin = q1-dq/2
        dq: width of the bin 
        n: number of bins, int        
    output: 1d array of shape (n,), the l-th element is the center value of the l-th bin
    qmin                    qmax
    |____|____|____|...|____|
    q1   q2   q3       qn

    '''
    q1 = min+delta/2
    qn = q1+(n-1)*delta
    bins = np.linspace(q1,qn,num=n,endpoint=True)
    return bins

def find_bin(qbins,dq,myvalue):
    '''
    function to find the array of the indices of the bins
    which the given array of values fall into (given the bins)
    use q as example
    input: qbins: 1d array of shape(n,)
        dq: width of the bin
        myvalue: 2d array of shape (m1,m2), dtype = float
    output: bin_index, 2d array of shape (m1,m2), dtype = int 
            if 0 <= bin index <= n-1, myvlaue falls in range [qmin,qmax)
            if bin index = n, myvalue falls out of range 
            bin index = n is an overflow bin
    e1=qmin  e2  e3   e4  en   e(n+1)=qmax
        |____|____|____|...|____|
        q1   q2   q3       qn
    '''
    qmax = qbins[-1]+dq/2
    edges = np.append((qbins-dq/2),qmax) #edges of the bins
    myvalue = myvalue[:,:,np.newaxis] #reshape myvalue to (m1,m2,1)
    edges = edges[np.newaxis,np.newaxis,:] #reshape edges to (1,1,n+1)
    bin_index = np.sum(edges<=myvalue,axis=2)-1
    bin_index[np.nonzero(bin_index<0)] = qbins.size #qbins.size=n
    return bin_index

def getbandindices(cell):
    """
    Get band indices depending on cell.
    Set numval and numcon in input parameters for best practice.
    numval = 'all' --> all valence bands used.
    else numval needs to be integer valued.

    Similar for numcon
    """
    nbandstot = len(cell.cart_labels())
    nvaltot = int(cell.tot_electrons()/2)
    ncontot = nbandstot - nvaltot

    if parmt.numval == 'all':
        numval = nvaltot
        parmt.numval = nvaltot
    elif parmt.numval>nvaltot:
        raise Exception(('Number of valence bands ' 
                    'exceeds total number of valence bands ({}). Please re-enter.').format(nvaltot))
    else:
        numval = parmt.numval
    
    if parmt.numcon == 'all':
        numcon = ncontot
        parmt.numcon = ncontot
    elif parmt.numcon>ncontot:
        raise Exception(('Number of conduction bands '
                    'exceeds total number of conduction bands ({}). Please re-enter.').format(ncontot))
    else:
        numcon = parmt.numcon

    ivaltop = nvaltot-1 #index of the highest valence band included
    ivalbot = ivaltop-numval+1 #index of the lowest valence band included
    iconbot = ivaltop+1 #index of the lowest conduction band included
    icontop = iconbot+numcon-1 #index of the highest conduction band included

    np.save('./'+parmt.mid_step_folder + '/bands.npy', np.array([ivalbot, ivaltop, iconbot, icontop]))
    return nvaltot

def check_stored_data(cell):
    """
    function to check if stored data uses parameters the same as input parameters.
    Input:  cell: pyscf.pbc.gto.Mole object
    Output: None
    """
    def create_outfile(outfile, cell):
        outfile.create_dataset('a', data = cell.a)
        outfile.create_dataset('basis', data = cell._env)
        outfile.create_dataset('kpts', data = np.array([parmt.nkx, parmt.nky, parmt.nkz]))
        outfile.create_dataset('binning', data = np.array([[parmt.qmin_ame, parmt.dqbin_ame, parmt.nqbins], [parmt.emin, parmt.debin, parmt.nebins]]))
        outfile.attrs['dft_rcut'] = cell.rcut
        outfile.attrs['precision'] = cell.precision
        outfile.attrs['xc'] = parmt.xcfunc
        outfile.attrs['df'] = parmt.densityfit
        outfile.attrs['method'] = parmt.method
        if parmt.numerical:
            outfile.attrs['dark_rcut'] = 0
        else:
            outfile.attrs['dark_rcut'] = parmt.Rvec_cut
        outfile.attrs['atom'] = cell.atom
        outfile.attrs['numval'] = parmt.numval
        outfile.attrs['numcon'] = parmt.numcon
        if parmt.effective_core_potential == None:
            outfile.attrs['ecp'] = 'None'
        else:
            outfile.attrs['ecp'] = cell.ecp
        return

    def check_pars(cell, infile):
        check = np.ones(14)
        check[0] = np.prod(infile['a'][...] == cell.a)
        check[1] = np.prod(convert_atom(infile.attrs['atom']) == convert_atom(cell.atom))
        check[2] = np.prod(np.round(infile['basis'][...], decimals = 7) == np.round(cell._env, decimals = 7))
        if parmt.effective_core_potential == None:
            check[3] = np.prod(infile.attrs['ecp'] == 'None')
        else:
            check[3] = np.prod(infile.attrs['ecp'] == cell.ecp)
        check[4] = np.prod(infile.attrs['dft_rcut'][...] == cell.rcut)
        check[5] = np.prod(infile.attrs['precision'][...] == cell.precision)
        check[6] = np.prod(infile['kpts'][...] == np.array([parmt.nkx, parmt.nky, parmt.nkz]))
        check[7] = np.prod(infile.attrs['xc'] == parmt.xcfunc)
        check[8] = np.prod(infile.attrs['method'] == parmt.method)
        check[9] = np.prod(infile.attrs['df'] == parmt.densityfit)
        check[10] = np.prod(infile['binning'][...] == np.array([[parmt.qmin_ame, parmt.dqbin_ame, parmt.nqbins], [parmt.emin, parmt.debin, parmt.nebins]]))
        if parmt.numerical:
            check[11] = np.prod(infile.attrs['dark_rcut'] == 0)
        else:
            check[11] = np.prod(infile.attrs['dark_rcut'] == parmt.Rvec_cut)
        check[12] = np.prod(infile.attrs['numval'] == parmt.numval)
        check[13] = np.prod(infile.attrs['numcon'] == parmt.numcon)
        return np.prod(check[:10]), np.prod(check), np.prod(check[:7])*check[11]

    def remove_files(dir):
        for files in os.listdir(dir):
            path = os.path.join(dir, files)
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)
        return 

    def convert_atom(string):
        string = string.replace(';', '\n')
        atoms = string.split('\n')
        return np.asarray([atom.split() for atom in atoms])

    try:
        infile = h5py.File('./'+parmt.mid_step_folder + '/run_settings.h5py', 'r')
        dft_check, ff_check, smat_check = check_pars(cell, infile)
        infile.close()
        os.remove('./'+parmt.mid_step_folder + '/run_settings.h5py')
        makedir('./'+parmt.mid_step_folder + '/Smat', False)
        makedir('./'+parmt.mid_step_folder + '/DFT', False)
        makedir('./'+parmt.mid_step_folder + '/f2', False)    
        if not smat_check:
            logging.info('Parameters for Scattering matrix in AO basis changed, deleting prior results.')
            remove_files('./'+parmt.mid_step_folder + '/Smat/')
            os.remove('./' + parmt.mid_step_folder + '/kpts.npy')
            os.remove('./' + parmt.mid_step_folder + '/Etab.npy')
            os.remove('./' + parmt.mid_step_folder + '/Rvecs.npy')
        if not dft_check:
            logging.info('Parameters changed for DFT calculation, deleting prior results.')
            remove_files('./'+parmt.mid_step_folder + '/DFT/')
        if not ff_check:
            logging.info('Parameters changed for crystal form factor, deleting prior results.')
            remove_files('./'+parmt.mid_step_folder + '/f2/')
        if smat_check and dft_check and ff_check:
            logging.info('Parameters have not changed since previous run. Keeping all calculated results.')
    
    except (FileNotFoundError, ValueError):
        logging.info('No previous version of run settings found. Deleting all components in temporary folder, ' + parmt.mid_step_folder + '.')
        remove_files('./'+parmt.mid_step_folder + '/')
    
    outfile = h5py.File('./'+parmt.mid_step_folder + '/run_settings.h5py', 'a')
    create_outfile(outfile, cell)
    outfile.close()
    
    makedir('./'+parmt.mid_step_folder + '/Smat', False)
    makedir('./'+parmt.mid_step_folder + '/DFT', False)
    makedir('./'+parmt.mid_step_folder + '/f2', False)

    atom_list = convert_atom(cell.atom)
    if len(set(atom_list[:,0])) > 1:
        parmt.multibasis = True

    return   

def gen_kpts(cell):
    '''
    function to generate the array of k-point weights wk
    if without symmetry, wk = 1/(total # of k-points)
    input: pyscf cell object
    output: 1d array of shape (nk,), each element is the weight of
            the corresponding k-point
    '''
    def gen_k_weights(nk,symmetry=False):
        if not symmetry:
            weights = np.ones(nk)*(1/nk)
        return weights

    try:
        kpts = np.load('./' + parmt.mid_step_folder + '/kpts.npy')
        logging.info('Prior k-points grid found and loaded.')
    except (FileNotFoundError, ValueError):
        nk = np.array([parmt.nkx,parmt.nky,parmt.nkz])
        kpts = cell.make_kpts(nk, wrap_around=True,with_gamma_point=True)
        np.save('./' + parmt.mid_step_folder + '/kpts.npy', kpts)
        logging.info('No prior k-points grid found, new grid made and saved.')
        trigger = True
    nktot = parmt.nkx*parmt.nky*parmt.nkz #total number of k-points
    kweights = gen_k_weights(nktot)
    return kpts, kweights, nktot

def run_gammapointdft(mycell,df,myxc,kpts):
    '''
    function to run k-points dft for the given cell.
    check the convergence of the run, and save the results of the run if converged
    input: mycell: pyscf pbc Cell object
        mykpts: k-points in 1/Bohr, 2d array of shape (nktot,3)
        df: density fit method, str
        myxc: exchange-correlation functional, str, in format 'exchange,correlation'
    output: returns energies, occupancy and coefficients
            saves the orbital energies, occupancy and coefficients along with relevant input parameters
    '''
    start_time = time.time()
    if df=='test': #fast density fitting with large error, just for testing the code
        mf = pbcdft.RKS(mycell).density_fit(auxbasis='weigend')
        mf.xc = myxc
        mf.kernel()
    if df=='MDF': #recommended for all-electron calculations  
        mf = pbcdft.RKS(mycell).mix_density_fit()
        mf.xc = myxc
        mf.kernel()
    runtime= time.time() - start_time
    if mf.converged:
        dft_path = './'+parmt.mid_step_folder + '/DFT/'
        np.save(dft_path + 'energies.npy', mf.mo_energy)
        np.save(dft_path + 'MO_coefficients.npy', mf.mo_coeff)
        np.save(dft_path + 'occupations.npy', mf.mo_occ)
    else:
        raise Exception('Gamma-point dft doesn\'t converge. Please try again.')
    return np.array(mf.mo_energy), np.array(mf.mo_occ), np.array(mf.mo_coeff)

def run_kpointsdft(mycell,mykpts,df,myxc):
    '''
    function to run k-points dft for the given cell.
    check the convergence of the run, and save the results of the run if converged
    input: mycell: pyscf pbc Cell object
        mykpts: k-points in 1/Bohr, 2d array of shape (nktot,3)
        df: density fit method, str
        myxc: exchange-correlation functional, str, in format 'exchange,correlation'
    output: returns energies, occupancy and coefficients
            saves the orbital energies, occupancy and coefficients along with relevant input parameters
    '''
    start_time = time.time()
    if df=='test': #fast density fitting with large error, just for testing the code
        kmf = pbcdft.KRKS(mycell, mykpts).density_fit(auxbasis='weigend')
        kmf.xc = myxc
        kmf.kernel()
    if df=='MDF': #recommended for all-electron calculations
        kmf = pbcdft.KRKS(mycell, mykpts).mix_density_fit()
        kmf.xc = myxc 
        #In the MDF scheme, modifying the default mesh for PWs to reduce the cost
        #kmf.with_df.mesh = [10,10,10]
        kmf.kernel()
    runtime= time.time() - start_time
    if kmf.converged:
        dft_path = './'+parmt.mid_step_folder + '/DFT/'
        np.save(dft_path + 'energies.npy', kmf.mo_energy)
        np.save(dft_path + 'MO_coefficients.npy', kmf.mo_coeff)
        np.save(dft_path + 'occupations.npy', kmf.mo_occ)
    else:
        raise Exception('k-points dft doesn\'t converge. Please try again.')
    return np.array(kmf.mo_energy), np.array(kmf.mo_occ), np.array(kmf.mo_coeff)

def loadenergy(mysettings):
    '''function to read the saved orbtial energies from file. If file not found,
    run the corresponding dft calculation.
    input: mysettings: dictionary of the parameters of the all-electron calculation
        keys:
        "method": 'gpdft' = gamma-point dft, 'kptsdft' = k-points dft
        "cell": pyscf pbc Cell object
        "kpts": k-points in 1/Bohr, 2d array of shape (nktot,3)
        "dfschm": density fitting scheme, 'test' or 'MDF'
        "xcfunc": exchange-correlation functional, in format 'exchange,correlation'
    output: orbital energy, 2d array of shape (nktot,nbandstot)
    '''
    st = time.time()
    method = mysettings["method"]
    cell = mysettings["cell"]
    kpts = mysettings["kpts"]
    dfschm = mysettings["densityfit"]
    xcfunc = mysettings["xcfunc"]
    try:
        energyarr = np.load('./'+parmt.mid_step_folder + '/DFT/energies.npy')
    except (FileNotFoundError, ValueError):
        logging.info('Running {} calculation.'.format(method))
        if method=='gpdft':
            energyarr, _, _ = run_gammapointdft(cell,dfschm,xcfunc,kpts)
        elif method=='kptsdft':
            energyarr, _, _ = run_kpointsdft(cell,kpts,dfschm,xcfunc)
        else:
            raise Exception('Invalid method.')
    
    if method=='gpdft':
        energyarr = energyarr[np.newaxis,:]
    logging.info('Energy file loaded. Time spend = {:.2f}s'.format(time.time() - st))
    return np.asarray(energyarr)

def loadcoeff(mysettings):
    '''function to read the saved orbtial coefficients from file. If file not found,
    run the corresponding dft calculation.
    input: mysettings: dictionary of the parameters of the all-electron calculation
        keys:
        "method": 'gpdft' = gamma-point dft, 'kptsdft' = k-points dft
        "cell": pyscf pbc Cell object
        "kpts": k-points in 1/Bohr, 2d array of shape (nktot,3)
        "dfschm": density fitting scheme, 'test' or 'MDF'
        "xcfunc": exchange-correlation functional, in format 'exchange,correlation'
    output: orbital energy, 2d array of shape (nktot,nbandstot)
    '''
    method = mysettings["method"]
    cell = mysettings["cell"]
    kpts = mysettings["kpts"]
    dfschm = mysettings["densityfit"]
    xcfunc = mysettings["xcfunc"]
    try:
        coefficients = np.load('./'+parmt.mid_step_folder + '/DFT/MO_coefficients.npy')
    except (FileNotFoundError, ValueError):
        logging.info('Running {} calculation.'.format(method))
        if method=='gpdft':
            _, _, coeffarr = run_gammapointdft(cell,dfschm,xcfunc)
        elif method=='kptsdft':
            _, _, coeffarr = run_kpointsdft(cell,kpts,dfschm,xcfunc)
        else:
            raise Exception('Invalid method.')
    
    if method=='gpdft':
        coeffarr = coeffarr[np.newaxis,:,:]
    return np.array(coeffarr)

def scissorcorrection(energy, nvalbands):
    '''function to perform scissor correction to the band structure-
    DFT underestimates bandgap. To deal with this, find bandgap, correct with 
    observed bandgap by adding energy to conduction bands.
    input:  energy: output from DFT
            nvalbands: number of valence bands in restricted calculation
    output: energies in eV
    '''
    energy = energy*har2ev
    calc_BG = energy[:,nvalbands:].min() - energy[:,:nvalbands].max()
    if parmt.do_scissor:
        correction = (parmt.expt_BG - calc_BG)*np.ones(energy[:,nvalbands:].shape)
        energy[:,nvalbands:] = energy[:,nvalbands:] + correction
        energy = energy - energy[:,:nvalbands].max()
        logging.info('Scissor correction applied, calculated BG = {},  new BG = {} eV.'.format(calc_BG, parmt.expt_BG))
        logging.info('Scissor correction procedure also displaces energy so that max energy of valence bands is 0.')
    else:
        logging.info('Scissor correction not applied, calculated BG = {}.'.format(calc_BG))
    np.save('./' + parmt.mid_step_folder + '/energy_eV.npy', energy)
    return energy

def gen_lattice(prim_vec, cutoff, cutoff_type):
    '''
    function to generate list of direct/reciprocal lattic vectors up to a cutoff value
    old -> assume cubic lattice
    current -> no cubic cell assumption
    input: prim_vec: primitive direct/reciprocal lattice vectors, in unit Bohr
                    2d array of shape(3,3), each row is a primitive lattice vector
        cutoff: cutoff value
        cutoff_type: 'mult': cutoff is in multiples of the length of the primitive 
                                vector |a| or |b|, int
                        'value': cutoff is in unit Bohr, float
    output: 2d array of shape (#,3), each row is the x,y,z component of the lattice vector
            # is the total number of allowed lattice vectors 
    '''
    latnorm = np.linalg.norm(prim_vec, axis = 1)
    if cutoff_type=='mult':
        n = 2*int(cutoff)
        cutoff = cutoff*np.min(latnorm)
    elif cutoff_type=='value':
        n = int(2*np.floor(cutoff/np.min(latnorm)))
    else:
        raise Exception('cutoff_type incorrect. Please use:\n\'mult\' for number of nearest neighbors to include, or \n\'value\' for cutoff scale')
    N_range = list(range(-n,n+1)) 
    triplets = list(itertools.product(N_range, repeat=3))
    mygrid = np.asarray(triplets)
    lattice = np.sum(mygrid[:,:,np.newaxis]*prim_vec[np.newaxis,:,:], axis = 1).astype('float')
    lattice = lattice[np.linalg.norm(lattice, axis=1)<=cutoff]
    sortindx = np.argsort(np.linalg.norm(lattice, axis=1))
    lattice = lattice[sortindx]
    return lattice

def prep_Etab(mycell,Rvec):
    '''
    function to create arrays used in generating the table of E_{ij}^t
    input: mycell: pyscf pbc Cell object
        Rvec: 2d array of shape (nR,3), nR is the total number of R vectors
    output: tuple (allexps,allpowers,dists)
            allexps: 1d array of all exponents in the basis set
            allpowers: 1d array of all angular quantum numbers corresponding to "allexps", 
                    datatype = int
            dists: all possible distances between all nucleus in unit Bohr, 
                with the first nucleus shifted by lattice vector \vec{R}
                2d array of shape (nR,nd), where nd is the # of distances for each R,
                nd = 3*(N^2-N+1), N=# of atoms in the unit cell
    '''
    natm = mycell.natm
    if parmt.multibasis:
        nshells = int(mycell.nbas)
    else:
        nshells = int(mycell.nbas/natm)
    #number of shells for the basis set, =num of shells/num of atoms
    allexps = mycell.bas_exp(0)
    allpowers = np.ones(mycell._bas[0][2],dtype=int)*mycell._bas[0][1]
    for i in range(1,nshells):
        exps = mycell.bas_exp(i) #exponents of the given shell
        L=mycell._bas[i][1] #angular momentum
        powers = np.ones(mycell._bas[i][2],dtype=int)*L
        allexps = np.append(allexps,exps)
        allpowers = np.append(allpowers,powers)
    ## Generate the array of all possible distances between all nucleus
    nR = Rvec.shape[0] #total number of R vectors
    nd = 3*(natm**2-natm+1) #number of distances for each R
    dists = np.zeros((nR,nd))
    for r in range(nR):
        distsR = Rvec[r,:]
        for i in range(natm):
            for j in range(natm):
                if i!=j:
                    mydist = Rvec[r,:]+mycell.atom_coord(i)-mycell.atom_coord(j)
                    distsR = np.append(distsR,mydist)
        dists[r,:] = distsR
    return (allexps,allpowers,dists)

def Gen_Etab(mycell,Rvec):
    '''
    function to generate the table of all values of E_{ij}^t, 
    store them in an array Etab and save the array to file
    input: mycell: pyscf pbc Cell object
        Rvec: 2d array of shape (nR,3), nR is the total number of R vectors
    output: Etab, 7d array of shape (lmax+1,lmax+1,2*lmax+1,nR,nd,# exps,# exps)
            saved to file 'Etab.npy'
            returns None
    '''
    try:
        Etab = np.load('./' + parmt.mid_step_folder + '/Etab.npy')
        del Etab
        logging.info('Etab file found.')
    except (FileNotFoundError, ValueError):
        logging.info('Etab file not found. Constructing table of Hermite Gaussian coefficients.')
        st = time.time()
        allexps,allpowers,dists = prep_Etab(mycell,Rvec)
        lmax = np.max(allpowers)
        tabshape = (lmax+1,lmax+1,2*lmax+1)+dists.shape+(len(allexps),len(allexps))
        Etab = np.zeros(tabshape)
        for a in range(len(allexps)):
            for b in range(len(allexps)):
                for i in range(allpowers[a]+1):
                    for j in range(allpowers[b]+1):
                        for t in range(i+j+1):
                            for r in range(dists.shape[0]):
                                for d in range(dists.shape[1]):
                                    try:
                                        Etab[i,j,t,r,d,a,b] = cartmoments.E(i,j,t,dists[r,d],allexps[a],allexps[b])
                                    except IndexError:
                                        pdb.set_trace()
        np.save('./' + parmt.mid_step_folder + '/Etab.npy',Etab)
        del Etab
        logging.info('Etab file constructed, time taken = {:.2f} s.'.format(time.time() - st))
    return None

def create_Rvecs_Etab(cell):
    """
    Function to create Rvectors and Etab given PySCF cell object.
    input:  cell: pyscf.pbc.gto.Mole object
    output: None
    """
    if not parmt.numerical:
        try:
            Rvectors = np.load('./' + parmt.mid_step_folder + '/Rvecs.npy')
        except:
            latvec_bohr = cell.a*a2bohr
            Rvectors = gen_lattice(latvec_bohr,parmt.Rvec_cut,'mult')
            np.save('./' + parmt.mid_step_folder + '/Rvecs.npy', Rvectors)
        Gen_Etab(cell, Rvectors)
        del Rvectors
    return

def estimate_G_cut(qmax, cell):
    '''
    In general, we need to create a lattice larger than qmax_bohr.
    This fn calculates the maximum |G| required to obtain q_max for any pair of k-points
    in the first Brillouin Zone.
    input:  qmax: maximum momenta transfer to be included in form factor
            cell: PySCF pbc.gto.cell object
    output: Gcut: max |G| required
    '''
    N_range = list(range(-1,2)) 
    triplets = list(itertools.product(N_range, repeat=2))
    mygrid = np.asarray(triplets)
    recs = cell.reciprocal_vectors()
    return qmax + np.linalg.norm(recs[0]+(recs[np.newaxis,1:,:]*mygrid[:,:,np.newaxis]).sum(axis = 1), axis = 1).max()

def calculate_Gvectors(cell):
    '''
    Calculate all required G-vectors and save to file.
    '''
    qmax = parmt.qmin_ame + parmt.dqbin_ame*parmt.nqbins
    st = time.time()
    Gvec_cut = estimate_G_cut(qmax, cell)
    Gvectors = gen_lattice(cell.reciprocal_vectors(),Gvec_cut,'value')
    np.save('./' + parmt.mid_step_folder + '/all_Gvectors.npy', Gvectors)
    logging.info('Modelling total {} Gvectors, all pairs of k-points will use less. Time taken = {:.2f} s.'.format(Gvectors.shape[0], time.time() - st))
    del Gvectors
    return 

def getprefactor(cell):
    """
    Create the prefactor for the form factor calculation.
    Input:  cell: pyscf.pbc.gto.Mole object
    Output: None
    """
    ebins = create_bins(parmt.emin,parmt.debin,parmt.nebins)
    vcell = cell.vol #volume of the unit cell in Bohr^3
    vcell_nat = vcell*bohr2m**3/(hbarc**3) #vcell in natural units (hbar*c/eV)**3
    factor1 = 2*np.pi**2/(alpha*me**2*vcell_nat) #numerator of the prefactor term, in eV
    factor2 = factor1/ebins
    np.save('./' + parmt.mid_step_folder + '/prefactor.npy', factor2)
    return None

def Gen_AO(mycell):
    '''
    function to generate a list of atomic orbitals (contracted cartesian guassians),
    which are BasisFunction objects defined in cartmoments
    input: mycell: pyscf pbc Cell object       
    output: list of atomic orbitals, BasisFunction objects
    '''
    cart_labs = mycell.cart_labels() #list of contracted Cartesian Gaussian orbitals
    if parmt.multibasis:
        nshells = mycell.nbas
    else:
        nshells = int(mycell.nbas/mycell.natm)
    len_allexps = 0 #total number of exponents in the basis set, = len(allexps)  
    for s in range(nshells):
        len_allexps += mycell._bas[s][2]         
    a = [] 
    index_cart = 0
    myindex = 0
    for i in range(mycell.nbas): #num of shells
        atm_indx=mycell._bas[i][0] #atom id
#       L=mycell._bas[i][1] #angular momentum
        nprim=mycell._bas[i][2] #num of primitive GTOs
        ncgto=mycell._bas[i][3] #num of contracted GTOs
        origin = mycell.atom_coord(atm_indx) #in Bohr
        for j in range(ncgto):
            for k in range(mycell.bas_len_cart(i)): #num of Cartesian functions for this shell
                shell = (cart_labs[index_cart].count('x'),cart_labs[index_cart].count('y'),cart_labs[index_cart].count('z'))
#                print('shell=',i,'ncgto=',j,'nCart=',k,'orbital=',cart_labs[index_cart])
                index_cart += 1
                exps = mycell.bas_exp(i)
                coeffs = mycell.bas_ctr_coeff(i)[:,j]
                exps_ind = np.arange(nprim)+myindex%(len_allexps)
#                print('exps_ind=',exps_ind,'myindex=',myindex)
                cg = cartmoments.BasisFunction(origin,shell,nprim,exps,coeffs,exps_ind,atm_indx) #cg=cartesian guassian
                a.append(cg)
        myindex += nprim
    return a

def initiate_numerical(cell):
    st = time.time()
    dirpath = './' + parmt.mid_step_folder + '/numerical'
    makedir(dirpath, log = False)
    kpts = np.load('./' + parmt.mid_step_folder + '/kpts.npy')
    coords, weights = pbcdft.gen_grid.gen_becke_grids(cell)
    np.save(dirpath + '/coords.npy', coords)
    np.save(dirpath + '/weights', weights)
    for i, kpt in enumerate(kpts):
        ao_i = pbcdft.numint.eval_ao(cell, coords, kpt = kpt)
        np.save(dirpath + '/ao_{}'.format(i), ao_i)
    del coords, weights, ao_i
    logging.info('initial data stored for numerical calculation, time taken = {:.2f} s'.format(time.time() - st))
    return

def calculate_q_vecs(dk):
    """
    Read all G vectors and calculate q = dk + G. Only returns q: |q| < qmax
    Input:  dk: 1d vector (3,). Final k-point - initial k-point. Units of Bohr^-1
    Output: qvecs: 2d array (LG, 3)
    """
    qmax = parmt.qmin_ame + parmt.dqbin_ame*parmt.nqbins
    Gvectors = np.load('./' + parmt.mid_step_folder + '/all_Gvectors.npy')
    qvecs = dk + Gvectors
    qvecs = qvecs[np.linalg.norm(qvecs, axis = 1) <= qmax]
    qvecs = qvecs[np.linalg.norm(qvecs, axis = 1) >= parmt.qmin_ame]
    return qvecs

def split_qvecs(qvecs, ncart, numk, iki, ikf):
    """
    function to split the calculation so computer does not run out
    of memory.

    Input:  dirpath: directory to store split q vectors in. 
            qvecs: all q vectors relevant for given k-point
            ncart: number of cartesian basis functions
            numk: number of k-points
    Return: num_sets: number of split sets
    """
    dirpath = './' + parmt.mid_step_folder + '/Smat/{}_{}'.format(iki, ikf)
    makedir(dirpath, log = False)
    val = np.min([numk**2, parmt.max_cores])
    memG = val*(ncart**2 + parmt.numcon*parmt.numval)/(2**26)
    maxG = int(parmt.max_memory/memG)
    numq = qvecs.shape[0]
    qlist = []
    if numq%maxG == 0:
        num_sets = numq//maxG
    else:
        num_sets = numq//maxG + 1
    for i in range(num_sets):
        if maxG*(i+1) < numq:
            qlist.append(qvecs[maxG*i:maxG*(i+1)])
        else:
            qlist.append(qvecs[maxG*i:])
    return num_sets, qlist

def calculate_E_indices(kf, ki, energies):
    ind = np.load('./'+parmt.mid_step_folder + '/bands.npy')
    ebins = create_bins(parmt.emin,parmt.debin,parmt.nebins)
    Ef, Ei = energies[kf], energies[ki]
    Ediff = Ef[ind[2]:ind[3]+1, None] - Ei[None,ind[0]:ind[1]+1]
    return find_bin(ebins, parmt.debin, Ediff)

def calculate_q_indices(qvec):
    qbins = create_bins(parmt.qmin_ame,parmt.dqbin_ame,parmt.nqbins)
    q = np.linalg.norm(qvec, axis = 1)
    return find_bin(qbins, parmt.dqbin_ame, q[None,:])[0]

def comp_ovlp(i,j,a,qvec,Rvec,natm,amp,Etab):
    '''
    function to compute the term \sum_R e^{-ik'R}<Ga(r-R)|e^{iqr}|Gb(r)>, 
    the term in the bracket is the overlap between two contracted Gaussians
    times a plane wave, with center of the first Gaussian shifted 
    by lattice vector \vec{R}
    input: i: int, index of the first atomic orbital in a
        j: int, index of the second atomic orbital in a
        a: list of Atomic Orbitals, list of BasisFunction objects
        qvec: 2d array of shape (LG,3), the 3 components of the last 
                dimension correspond to (qx,qy,qz), respectively
        Rvec: 2d array of shape (nR,3), nR is the total number of R vectors
        natm: int, total number of atoms in the unit cell
        amp: 1d array of shape (nR,), dtype = complex, e^{-ik'R}
    output: 2d array of shape (LG), dtype = complex 
    '''
    if a[i].atm_ind == a[j].atm_ind:
        dist_ind = 0
    else:
        dist_ind = (a[i].atm_ind*(natm-1)+1)*3
        if a[j].atm_ind < a[i].atm_ind:
            dist_ind += a[j].atm_ind*3
        else:
            dist_ind += (a[j].atm_ind-1)*3
    mySqR = cartmoments.SqR(a[i],a[j],qvec,dist_ind,Rvec,Etab)
    SqRsum = np.sum(amp[np.newaxis,:]*mySqR,axis=1)
    return SqRsum

def AO_ovlp_matrix(ikf, iki, kf, ao_list, natm, indx, qvecs):
    """
    function to compute \sum_R e^{-ik'R}<Ga(r-R)|e^{iqr}|Gb(r)> for all
    a and b. Numerical parts are loaded from data stored on disk
    to reduce memory cost.

    Input:  ikf: int, index of final k-point
            iki: int, index of initial k-point
            kf: 1d vector (3,) of final k-point
            ao_list: list of cartmoments.BasisFunction objects
            natm: int, number of atoms in the calculation
            indx: int, marker if multiple files are needed to compute these results.
    Output: 3D array of shape (LG, numCart, numCart), dtype = np.complex128
    """
    dirpath = './' + parmt.mid_step_folder + '/Smat/{}_{}'.format(iki, ikf)
    try:
        matrix = np.load(dirpath + '/{}.npy'.format(indx))
        if qvecs.shape[0] != matrix.shape[0]:
            del matrix
            os.remove(dirpath + '/{}.npy'.format(indx))
            raise FileNotFoundError
    except (FileNotFoundError, ValueError):
        Etab = np.load('./' + parmt.mid_step_folder + '/Etab.npy')
        Rvec = np.load('./' + parmt.mid_step_folder + '/Rvecs.npy')
        numCart = len(ao_list)
        amp = np.exp(-1j*np.sum(kf[np.newaxis,:]*Rvec[:,:],axis=1))
        matrix = np.zeros((qvecs.shape[0], numCart, numCart), dtype = np.complex128)
        for i in range(numCart):
            for j in range(numCart):
                matrix[:, i, j] = comp_ovlp(i, j, ao_list, qvecs, Rvec, natm, amp, Etab)
        if parmt.save_Smat:
            np.save(dirpath + '/{}.npy'.format(indx), matrix)
    return matrix

def numerical_AO_overlap(ikf, iki, qvecs, indx):
    try:
        dirpathS = './' + parmt.mid_step_folder + '/Smat/{}_{}'.format(iki, ikf)
        ovlp = np.load(dirpathS + '/{}.npy'.format(indx))
        if qvecs.shape[0] != ovlp.shape[0]:
            del ovlp
            os.remove(dirpath + '/{}.npy'.format(indx))
            raise FileNotFoundError
    except (FileNotFoundError, ValueError):
        dirpath = './' + parmt.mid_step_folder + '/numerical/'
        coords, weights = np.load(dirpath + 'coords.npy'), np.load(dirpath + 'weights.npy')
        ao_f, ao_i = np.load(dirpath + 'ao_{}.npy'.format(ikf)), np.load(dirpath + 'ao_{}.npy'.format(iki))
        operator = np.exp(1.j * np.einsum('wx, qx -> wq', coords, qvecs))
        ovlp = np.einsum('w, wi, wq, wj -> qij', weights, ao_f.conj(), operator, ao_i)
        if parmt.save_Smat:
            np.save(dirpathS + '/{}.npy'.format(indx), ovlp)
    return ovlp

def bin_to_f2(iki, ikf, mo_prob, kweights, eindx, qindx):
    """
    function to bin mo_prob into f2
    input:  iki: index of initial k point
            ikf: index of final k point
            mo_prob: 3d array of transition probabilities (Lq, num_final_bands, num_initial_bands)
            eindx: array of energy indices for transitions
            qindx: array of momentum indices for each q vector
    output: f2: 2d array (parmt.nqbins, parmt.nebins) of crystal form factor
    """
    f2 = np.zeros((parmt.nqbins + 1, parmt.nebins + 1), dtype = float)
    for iq, ov in zip(qindx, mo_prob):
        for ie1, ov1 in zip(eindx, ov):
            for ie, ov2 in zip(ie1, ov1):
                f2[iq, ie] += kweights[iki]*kweights[ikf]*ov2/kweights.sum()
    f2 = f2[:parmt.nqbins, :parmt.nebins]
    qbins, ebins = create_bins(parmt.qmin_ame,parmt.dqbin_ame,parmt.nqbins), create_bins(parmt.emin,parmt.debin,parmt.nebins)
    prefactor = np.load('./' + parmt.mid_step_folder + '/prefactor.npy')
    f2 *= qbins[:,np.newaxis]*ebins[np.newaxis,:]/(parmt.dqbin_ame*parmt.debin)*prefactor[np.newaxis,:]
    return f2

def compute_for_dk(ikf, iki, ao_list, natm, kweights):
    st = time.time()
    if iki < ikf:
        iki, ikf = ikf, iki

    dirpath0 = './' + parmt.mid_step_folder + '/f2/{}_{}'.format(iki, ikf)
    makedir(dirpath0, log = False)
    if ikf != iki:
        dirpath1 = './' + parmt.mid_step_folder + '/f2/{}_{}'.format(ikf, iki)
        makedir(dirpath1, log = False) 

    try:
        f2_0 = np.load(dirpath0 + '/f2.npy')
        if ikf != iki:
            f2_1 = np.load(dirpath1 + '/f2.npy')
        logging.info('Prior evaluation found for ki = {}, kf = {}, using this result.'.format(iki, ikf))
    except (FileNotFoundError, ValueError):
        kpts = np.load('./' + parmt.mid_step_folder + '/kpts.npy')
        ki, kf = kpts[iki], kpts[ikf]
        coeffs = np.load('./' + parmt.mid_step_folder + '/DFT/MO_coefficients.npy')
        energies = np.load('./' + parmt.mid_step_folder + '/energy_eV.npy')
        qvecs = calculate_q_vecs(kf - ki)
        logging.info('Starting evaluation, contribution of ki = {}, kf = {} to cff, number of qvecs = {}.'.format(iki, ikf, qvecs.shape[0]))
        ncart, numk = len(ao_list), kpts.shape[0]
        num_sets, qlist = split_qvecs(qvecs, ncart, numk, iki, ikf)
        eindx0 = calculate_E_indices(ikf, iki, energies)
        eindx1 = calculate_E_indices(iki, ikf, energies)
        ind = np.load('./'+parmt.mid_step_folder + '/bands.npy')
        f2_0 = np.zeros((parmt.nqbins, parmt.nebins), dtype = float)
        f2_1 = np.zeros((parmt.nqbins, parmt.nebins), dtype = float)
        for indx, qvecs in enumerate(qlist):
            if parmt.numerical:
                ao_ovlp = numerical_AO_overlap(ikf, iki, qvecs, indx)
            else:
                ao_ovlp = AO_ovlp_matrix(ikf, iki, kf, ao_list, natm, indx, qvecs)
            logging.info('\tki = {}, kf = {}, ovlp in AO basis calculated for set {}/{}, time elapsed = {:.2f} s.'.format(iki, ikf, indx+1, num_sets, time.time() - st))
            mo_prob0 = np.abs(np.einsum('im,qmn,nj->qij', coeffs[ikf].T.conj()[ind[2]:ind[3]+1,:], ao_ovlp, coeffs[iki,:,ind[0]:ind[1]+1]))**2
            if iki != ikf:
                mo_prob1 = np.abs(np.einsum('im,qmn,nj->qij', coeffs[iki].T.conj()[ind[2]:ind[3]+1,:], np.einsum('qij->qji', ao_ovlp.conj()), coeffs[ikf,:,ind[0]:ind[1]+1]))**2
            del ao_ovlp
            logging.info('\tki = {}, kf = {}, transition probabilities calculated for set {}/{}, time elapsed = {:.2f} s.'.format(iki, ikf, indx+1, num_sets, time.time() - st))
            qindx = calculate_q_indices(qvecs)
            f2_0 += bin_to_f2(iki, ikf, mo_prob0, kweights, eindx0, qindx)
            del mo_prob0
            if iki != ikf:
                f2_1 += bin_to_f2(ikf, iki, mo_prob1, kweights, eindx1, qindx)
                del mo_prob1
            logging.info('\tki = {}, kf = {}, cff calculated for set {}/{}, time elapsed = {:.2f} s.'.format(iki, ikf, indx+1, num_sets, time.time() - st))
        logging.info('Computation complete for ki = {}, kf = {}, time elapsed = {:.2f} s.'.format(iki, ikf, time.time() - st))
    if parmt.save_temp_f2:
        np.save(dirpath0 + '/f2.npy', f2_0)
        if iki != ikf:
            np.save(dirpath1 + '/f2.npy', f2_1)
        return 0
    return f2_0 + f2_1

def load_calculated_f2(nkpoints):
    f2 = np.zeros((parmt.nqbins, parmt.nebins), dtype = float)
    for i in range(nkpoints):
        for j in range(nkpoints):
            f2 += np.load('./' + parmt.mid_step_folder + '/f2/{}_{}/f2.npy'.format(i, j))
    return f2

def calculate_parallel(cell, kweights):
    st = time.time()
    natm = cell.natm
    ao_list = Gen_AO(cell)
    #ikf_iki_list = list(itertools.product(list(range(kweights.shape[0])), repeat=2))
    li = list(range(kweights.shape[0]))
    ikf_iki_list = []
    for iki in li:
        for ikf in li:
            if iki >= ikf:
                ikf_iki_list.append((ikf, iki))
    logging.info('Number of unique ki, kf pairs = {}.'.format(len(ikf_iki_list)))
    if parmt.numerical:
        logging.info('Scattering matrix calculation in AO basis is set to numerical.')
        initiate_numerical(cell)
    with Pool(parmt.max_cores) as p:
        f2_arr = p.starmap(partial(compute_for_dk, ao_list = ao_list, natm = natm, kweights = kweights), ikf_iki_list)
    if parmt.save_temp_f2:
        f2 = load_calculated_f2(kweights.shape[0])
    else:
        f2 = np.sum(f2_arr, axis = 0)
    logging.info('Crystal form factor calculated, time taken = {:.2f} s.'.format(time.time() - st))
    return f2

def final_file(cell, f2):
    # Remove any earlier iterations of the form factor file.
    Filepath = os.getcwd()+'/'+ parmt.ff_file
    if os.path.isfile(Filepath):
        os.remove(parmt.ff_file)
    
    kpts = np.load('./' + parmt.mid_step_folder + '/kpts.npy')
    
    # Make new file
    outfile = h5py.File(parmt.ff_file, 'a')
    
    runsettings = outfile.create_group('run_settings')
    runsettings.create_dataset('a', data = cell.a)
    runsettings.attrs['atom'] = cell.atom
    runsettings.attrs['basis'] = cell.basis
    if parmt.effective_core_potential == None:
        runsettings.attrs['ecp'] = 'None'
    else:
        runsettings.attrs['ecp'] = cell.ecp
    runsettings.create_dataset('rcut', data = cell.rcut)
    runsettings.create_dataset('precision', data = cell.precision)
    if not parmt.numerical:
        Rvec = np.load('./' + parmt.mid_step_folder + '/Rvecs.npy')
        runsettings.create_dataset('Rvec', data = Rvec)
    runsettings.attrs['numcon'] = parmt.numcon
    runsettings.attrs['numval'] = parmt.numval
    runsettings.attrs['xc'] = parmt.xcfunc
    runsettings.attrs['df'] = parmt.densityfit
    runsettings.attrs['method'] = parmt.method
    runsettings.create_dataset('kpts', data = kpts)
    runsettings.create_dataset('binning', data = np.array([[parmt.qmin_ame, parmt.dqbin_ame, parmt.nqbins], [parmt.emin, parmt.debin, parmt.nebins]]))

    results = outfile.create_group('results')
    results.attrs['dq'] = parmt.dqbin_ame
    results.attrs['dE'] = parmt.debin
    results.attrs['VCell'] = cell.vol*(bohr2m**3)/(hbarc**3)
    results.attrs['mCell'] = cell.atom_mass_list(isotope_avg = True).sum()*amu2eV
    results.attrs['bandgap'] = parmt.expt_BG
    results.attrs['scissor'] = parmt.do_scissor
    results.create_dataset('f2', data = f2)

    outfile.close()
    logging.info('Crystal form factor saved to {}.'.format(Filepath))
    return
