#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Do 25 Feb 2016 14:13:27 CET
# modified by Xiang-Yu Wu
'''calc the Lambda polarization on the freeze out hypersurface'''

from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import os
import sys
from time import time
import math
import four_momentum as mom
from scipy.interpolate import interp1d, splrep, splint
import pyopencl as cl
from pyopencl.array import Array
import pyopencl.array as cl_array
from tqdm import tqdm

os.environ['PYOPENCL_CTX']=':1'

class Polarization(object):
    '''The pyopencl version for lambda polarisation,
    initialize with freeze out hyper surface and omega^{mu}
    on freeze out hyper surface.'''
    def __init__(self, fpath,T=0.165, Mu=0.0,mass=1.115,path="./", Baryon_on = False,gpu_id = 0,themal=True,shear=False,accT=False):
        '''Param:
             sf: the freeze out hypersf ds0,ds1,ds2,ds3,vx,vy,veta,etas
             omega: omega^tau, x, y, etas
             T: freezeout temperature
        '''
        self.cwd, cwf = os.path.split(__file__)

        platform = cl.get_platforms()
        devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        devices = [devices[gpu_id]]
        self.ctx = cl.Context(devices=devices)
        self.queue = cl.CommandQueue(self.ctx)
        mf = cl.mem_flags

        self.Tfrz = T
        self.Mu = Mu
        self.mass = mass
        self.Baryon_on = Baryon_on

        self.themal = themal
        self.shear = shear
        self.accT = accT

        # calc umu since they are used for each (Y,pt,phi)
        sf = np.loadtxt(fpath+'/hypersf.dat', dtype=np.float32)
        self.size_sf = len(sf[:,0])
        h_sf = sf.astype(np.float32)
        self.d_sf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_sf)
        if self.themal:    
            omega = np.loadtxt(fpath+'/omegamu_sf.dat', dtype=np.float32).flatten()
            h_omega = omega.astype(np.float32)
            self.d_omega = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_omega)
        

        if self.shear:
            omega_shear1 = pd.read_csv(fpath+'/omegamu_shear1_sf.dat', dtype=np.float32,sep=" ",header=None,comment="#").values.flatten()
            omega_shear2 = pd.read_csv(fpath+'/omegamu_shear2_sf.dat', dtype=np.float32,sep=" ",header=None,comment="#").values.flatten()
            h_omega_shear1 = omega_shear1.astype(np.float32)
            h_omega_shear2 = omega_shear2.astype(np.float32)
            
            self.d_omega_shear1 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_omega_shear1)
            self.d_omega_shear2 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_omega_shear2)


        if self.accT:
            omega_accT = pd.read_csv(fpath+'/omegamu_accT_sf.dat', dtype=np.float32,sep=" ",header=None,comment="#").values.flatten()
            h_omega_accT = omega_accT.astype(np.float32)
            self.d_omega_accT = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_omega_accT)
            

        if self.Baryon_on:
            pass

    def set_momentum(self,momentum_list):
        self.d_momentum = cl_array.to_device(self.queue,
                                        momentum_list.astype(np.float32))

        num_of_mom = len(momentum_list)

        print('num_of_mom=', num_of_mom)

        compile_options = ['-D num_of_mom=%s'%num_of_mom]

        cwd, cwf = os.path.split(__file__)

        self.block_size = 256
        compile_options.append('-D BSZ=%s'%self.block_size)
        compile_options.append('-D SIZE_SF=%s'%np.int32(self.size_sf))
        compile_options.append('-D MASS=%s'%np.float32(self.mass))
        if self.Baryon_on:
            compile_options.append('-D BARYON_ON')
        else:
            compile_options.append('-D TFRZ=%s'%self.Tfrz)
        compile_options.append('-D Mu=%s'%self.Mu)
        compile_options.append('-I '+os.path.join(cwd, '../kernel/')) 
        compile_options.append( '-D USE_SINGLE_PRECISION' )
        
        print (compile_options)
        fpath = os.path.join(cwd, '../kernel/kernel_polarization.cl')

        with open(fpath, 'r') as f:
            src = f.read()
            self.prg = cl.Program(self.ctx, src).build(options=' '.join(compile_options))

        self.size = num_of_mom

    def get_themal(self):

        
        h_pol_th = np.zeros((self.size, 4), np.float32)
        h_rho_th = np.zeros(self.size, np.float32)

        # boost Pi^{mu} to the local rest frame of Lambda
        h_pol_lrf_th = np.zeros((self.size, 4), np.float32)

        self.d_pol = cl_array.to_device(self.queue, h_pol_th)
        self.d_rho = cl_array.to_device(self.queue, h_rho_th)

        self.d_pol_lrf = cl_array.to_device(self.queue, h_pol_lrf_th)
        


        # block_size*num_of_mom: gloabl size
        # block_size: local size
        self.prg.polarization_on_sf(self.queue, (self.block_size*self.size,),
            (self.block_size,), self.d_pol.data, self.d_rho.data, self.d_pol_lrf.data,
            self.d_sf, self.d_omega, self.d_momentum.data).wait()

        polarization = self.d_pol.get()
        density = self.d_rho.get()
        pol_lrf = self.d_pol_lrf.get()
        return polarization, density, pol_lrf


    def get_shear(self):

        
        h_pol_th = np.zeros((self.size, 4), np.float32)
        h_rho_th = np.zeros(self.size, np.float32)

        # boost Pi^{mu} to the local rest frame of Lambda
        h_pol_lrf_th = np.zeros((self.size, 4), np.float32)

        self.d_pol = cl_array.to_device(self.queue, h_pol_th)
        self.d_rho = cl_array.to_device(self.queue, h_rho_th)

        self.d_pol_lrf = cl_array.to_device(self.queue, h_pol_lrf_th)
        


        # block_size*num_of_mom: gloabl size
        # block_size: local size
        self.prg.polarization_shear_on_sf(self.queue, (self.block_size*self.size,),
            (self.block_size,), self.d_pol.data, self.d_rho.data, self.d_pol_lrf.data,
            self.d_sf, self.d_omega_shear1, self.d_omega_shear2,self.d_momentum.data).wait()

        polarization = self.d_pol.get()
        density = self.d_rho.get()
        pol_lrf = self.d_pol_lrf.get()
        return polarization, density, pol_lrf

    def get_accT(self):

        
        h_pol_th = np.zeros((self.size, 4), np.float32)
        h_rho_th = np.zeros(self.size, np.float32)

        # boost Pi^{mu} to the local rest frame of Lambda
        h_pol_lrf_th = np.zeros((self.size, 4), np.float32)

        self.d_pol = cl_array.to_device(self.queue, h_pol_th)
        self.d_rho = cl_array.to_device(self.queue, h_rho_th)

        self.d_pol_lrf = cl_array.to_device(self.queue, h_pol_lrf_th)
        


        # block_size*num_of_mom: gloabl size
        # block_size: local size
        self.prg.polarization_accT_on_sf(self.queue, (self.block_size*self.size,),
            (self.block_size,), self.d_pol.data, self.d_rho.data, self.d_pol_lrf.data,
            self.d_sf, self.d_omega_accT,self.d_momentum.data).wait()

        polarization = self.d_pol.get()
        density = self.d_rho.get()
        pol_lrf = self.d_pol_lrf.get()
        return polarization, density, pol_lrf



def rapidity_integral(spec_along_y, ylo=-0.5, yhi=0.5):
    '''1D integration along rapidity/pseudo-rapidity 
    The spline interpolation and integration is much faster than
    the interp1d() and quad combination'''
    #f = interp1d(Y, spec_along_y, kind='cubic')
    #return quad(f, ylo, yhi, epsrel=1.0E-5)[0]
    tck = splrep(mom.Y, spec_along_y)
    return splint(ylo, yhi, tck)

def pt_integral(spec_along_pt, ptlo=0.0, pthi=3.0):
    '''1D integration along rapidity/pseudo-rapidity 
    The spline interpolation and integration is much faster than
    the interp1d() and quad combination'''
    #f = interp1d(Y, spec_along_y, kind='cubic')
    #return quad(f, ylo, yhi, epsrel=1.0E-5)[0]
    tck = splrep(mom.PT, spec_along_pt)
    return splint(ptlo, pthi, tck)



def calc_pol_th(fpath):
    
    pol = Polarization(fpath,themal=True,shear=False,accT=False)

    ny,npt, nphi = mom.NY, mom.NPT, mom.NPHI
    momentum_list = np.zeros((ny*npt*nphi, 4), dtype=np.float32)
    #mom_list = np.zeros((11*61*21, 4), dtype=np.float32)
    mass = 1.115
    for k, Y in enumerate(mom.Y):
       for i, pt in enumerate(mom.PT):
           for j, phi in enumerate(mom.PHI):
               px = pt * math.cos(phi)
               py = pt * math.sin(phi)
               mt = math.sqrt(mass*mass + px*px + py*py)
               index = k*npt*nphi + i*nphi + j
               momentum_list[index, 0] = mt
               momentum_list[index, 1] = Y
               momentum_list[index, 2] = px
               momentum_list[index, 3] = py
    
    pol.set_momentum(momentum_list)
    polar, density, pol_lrf = pol.get_themal()

    
    
    polar[:,0] = polar[:,0]/density
    polar[:,1] = polar[:,1]/density
    polar[:,2] = polar[:,2]/density
    polar[:,3] = polar[:,3]/density
    
    polar = polar.reshape(mom.NY,mom.NPT,mom.NPHI,4)
    pol_lrf = pol_lrf.reshape(mom.NY,mom.NPT,mom.NPHI,4)
    density = density.reshape(mom.NY,mom.NPT,mom.NPHI)
    
    polar_phi = np.zeros((mom.NPHI,4))
    polar_phi_Y = np.zeros((mom.NPHI,mom.NY,4))

    polar_phi_lrf = np.zeros((mom.NPHI,4))
    polar_phi_Y_lrf = np.zeros((mom.NPHI,mom.NY,4))

    for i in range(4):
        for k, phi in enumerate(mom.PHI):
            for j, Y in enumerate(mom.Y):
                spec_along_PT = polar[j,:,k,i]
                polar_phi_Y[k,j,i] = pt_integral(spec_along_PT,ptlo=0.0,pthi=3.0)/(3.0)

                spec_along_PT = pol_lrf[j,:,k,i]
                polar_phi_Y_lrf[k,j,i] = pt_integral(spec_along_PT,ptlo=0.0,pthi=3.0)/(3.0)

            spec_along_y = polar_phi_Y[k,:,i]
            polar_phi[k,i] = rapidity_integral(spec_along_y,ylo=-1,yhi=1)

            spec_along_y = polar_phi_Y_lrf[k,:,i]
            polar_phi_lrf[k,i] = rapidity_integral(spec_along_y,ylo=-1,yhi=1)

    polar_phi = np.column_stack((mom.PHI,polar_phi))
    np.savetxt('%s/polar_th.dat'%fpath, polar_phi)
    np.savetxt('%s/polar_th_lrf.dat'%fpath, polar_phi)
    #print (density[10,10,10])


    #polar, density, pol_lrf = pol.get_shear()

    #polar, density, pol_lrf = pol.get(momentum_list)
    
    #polar[:,0] = polar[:,0]/density
    #polar[:,1] = polar[:,1]/density
    #polar[:,2] = polar[:,2]/density
    #polar[:,3] = polar[:,3]/density
    #
    #polar = polar.reshape(mom.NY,mom.NPT,mom.NPHI,4)
    #pol_lrf = pol_lrf.reshape(mom.NY,mom.NPT,mom.NPHI,4)
    #density = density.reshape(mom.NY,mom.NPT,mom.NPHI)
    #
    #polar_phi = np.zeros((mom.NPHI,4))
    #polar_phi_Y = np.zeros((mom.NPHI,mom.NY,4))

    #polar_phi_lrf = np.zeros((mom.NPHI,4))
    #polar_phi_Y_lrf = np.zeros((mom.NPHI,mom.NY,4))

    #for i in range(4):
    #    for k, phi in enumerate(mom.PHI):
    #        for j, Y in enumerate(mom.Y):
    #            spec_along_PT = polar[j,:,k,i]
    #            polar_phi_Y[k,j,i] = pt_integral(spec_along_PT,ptlo=0.0,pthi=3.0)/(3.0)

    #            spec_along_PT = pol_lrf[j,:,k,i]
    #            polar_phi_Y_lrf[k,j,i] = pt_integral(spec_along_PT,ptlo=0.0,pthi=3.0)/(3.0)

    #        spec_along_y = polar_phi_Y[k,:,i]
    #        polar_phi[k,i] = rapidity_integral(spec_along_y,ylo=-1,yhi=1)

    #        spec_along_y = polar_phi_Y_lrf[k,:,i]
    #        polar_phi_lrf[k,i] = rapidity_integral(spec_along_y,ylo=-1,yhi=1)

    #polar_phi = np.column_stack((mom.PHI,polar_phi))
    #np.savetxt('%s/polar_shear.dat'%fpath, polar_phi)
    #np.savetxt('%s/polar_shear_lrf.dat'%fpath, polar_phi)
    ##print (density[10,10,10])

    #polar, density, pol_lrf = pol.get_accT()

    ##polar, density, pol_lrf = pol.get(momentum_list)
    #
    #polar[:,0] = polar[:,0]/density
    #polar[:,1] = polar[:,1]/density
    #polar[:,2] = polar[:,2]/density
    #polar[:,3] = polar[:,3]/density
    #
    #polar = polar.reshape(mom.NY,mom.NPT,mom.NPHI,4)
    #pol_lrf = pol_lrf.reshape(mom.NY,mom.NPT,mom.NPHI,4)
    #density = density.reshape(mom.NY,mom.NPT,mom.NPHI)
    #
    #polar_phi = np.zeros((mom.NPHI,4))
    #polar_phi_Y = np.zeros((mom.NPHI,mom.NY,4))

    #polar_phi_lrf = np.zeros((mom.NPHI,4))
    #polar_phi_Y_lrf = np.zeros((mom.NPHI,mom.NY,4))

    #for i in range(4):
    #    for k, phi in enumerate(mom.PHI):
    #        for j, Y in enumerate(mom.Y):
    #            spec_along_PT = polar[j,:,k,i]
    #            polar_phi_Y[k,j,i] = pt_integral(spec_along_PT,ptlo=0.0,pthi=3.0)/(3.0)

    #            spec_along_PT = pol_lrf[j,:,k,i]
    #            polar_phi_Y_lrf[k,j,i] = pt_integral(spec_along_PT,ptlo=0.0,pthi=3.0)/(3.0)

    #        spec_along_y = polar_phi_Y[k,:,i]
    #        polar_phi[k,i] = rapidity_integral(spec_along_y,ylo=-1,yhi=1)

    #        spec_along_y = polar_phi_Y_lrf[k,:,i]
    #        polar_phi_lrf[k,i] = rapidity_integral(spec_along_y,ylo=-1,yhi=1)

    #polar_phi = np.column_stack((mom.PHI,polar_phi))
    #np.savetxt('%s/polar_accT.dat'%fpath, polar_phi)
    #np.savetxt('%s/polar_accT_lrf.dat'%fpath, polar_phi)
    #print (density[10,10,10])

if __name__ == '__main__':

    for i in tqdm(range(0,1000)):
        fpath = '/media/xywu/disk21/physics/code/clvisc/new/clvisc_git/clvisc/pyvisc/results/auau200_results_ampt/etas0p08_kfactor1p4_wo_hz/20_50/event%s'%i
        calc_pol_th(fpath)
        
    
