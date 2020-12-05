#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Thu 15 Oct 2015 14:02:44 CEST

from __future__ import absolute_import, division, print_function
import numpy as np
import pyopencl as cl
from pyopencl import array
import pyopencl.array as cl_array
from pyopencl.array import Array

import os
import sys
from time import time
from math import floor

cwd, cwf = os.path.split(__file__)
sys.path.append(cwd)
from eos.eos import Eos
import h5py


class BulkInfo(object):
    '''The bulk information like:
       ed(x), ed(y), ed(eta), T(x), T(y), T(eta)
       vx, vy, veta, ecc_x, ecc_p'''
    def __init__(self, cfg, ctx, queue, eos_table, compile_options):
        self.cfg = cfg
        self.ctx = ctx
        self.queue = queue
        self.eos_table = eos_table
        self.compile_options = list(compile_options)

        NX, NY, NZ = cfg.NX, cfg.NY, cfg.NZ

        if NX%2 == 1:
            self.x = np.linspace(-floor(NX/2)*cfg.DX, floor(NX/2)*cfg.DX, NX, endpoint=True)
            self.y = np.linspace(-floor(NY/2)*cfg.DY, floor(NY/2)*cfg.DY, NY, endpoint=True)
            self.z = np.linspace(-floor(NZ/2)*cfg.DZ, floor(NZ/2)*cfg.DZ, NZ, endpoint=True)
            #including grid point 0 
        elif NX%2 == 0:
            self.x = np.linspace(-((NX-1)/2.0)*cfg.DX, ((NX-1)/2.0)*cfg.DX, NX, endpoint=True)
            self.y = np.linspace(-((NY-1)/2.0)*cfg.DY, ((NY-1)/2.0)*cfg.DY, NY, endpoint=True)
            self.z = np.linspace(-floor(NZ/2)*cfg.DZ, floor(NZ/2)*cfg.DZ, NZ, endpoint=True)
            #NOT including grid point 0  for trento2D
        self.h_ev = np.zeros((NX*NY*NZ, 4), cfg.real)

        self.a_ed = cl_array.empty(self.queue, NX*NY*NZ, cfg.real)
        self.a_entropy = cl_array.empty(self.queue, NX*NY*NZ, cfg.real)

        # the momentum eccentricity as a function of rapidity
        self.a_eccp1 = cl_array.empty(self.queue, NZ, cfg.real)
        self.a_eccp2 = cl_array.empty(self.queue, NZ, cfg.real)
        
        # store the data in hdf5 file
        #h5_path = os.path.join(cfg.fPathOut, 'bulkinfo.h5')
        #self.f_hdf5 = h5py.File(h5_path, 'w')

        self.eos = Eos(cfg.eos_type)

        self.__load_and_build_cl_prg()

        # time evolution for , edmax and ed, T at (x=0,y=0,etas=0)
        self.time = []
        self.edmax = []
        self.edcent = []
        self.Tcent = []

        # time evolution for total_entropy, eccp, eccx and <vr>
        self.energy = []
        self.entropy = []
        self.eccp_vs_tau = []
        self.eccx = []
        self.vr= []


        # time evolution for bulk3D
        self.Tau_tijk =[]
        self.X_tijk = []
        self.Y_tijk = []
        self.Z_tijk = []
        self.ED_tijk = []
        self.Tp_tijk = []
#       self.Frc_tijk = []
        self.Vx_tijk = []
        self.Vy_tijk = []
        self.Vz_tijk = []

        # time evolution for bulk2D
        self.Tau_2d =[]
        self.X_2d=[]
        self.Y_2d=[]
        self.ED_2d=[]
        self.Tp_2d=[]
        self.Vx_2d = []
        self.Vy_2d = []
        self.Vz_2d = []
        self.Frc_2d = []



    def __load_and_build_cl_prg(self):
        with open(os.path.join(cwd, 'kernel', 'kernel_bulkinfo.cl')) as f:
                prg_src = f.read()
                self.kernel_bulk = cl.Program(self.ctx, prg_src).build(
                    options = ' '.join(self.compile_options))

    #@profile
    def get(self, tau, d_ev, edmax, d_pi=None):
        ''' store the bulkinfo to hdf5 file '''
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
        self.time.append(tau)
        self.edmax.append(edmax)

        cl.enqueue_copy(self.queue, self.h_ev, d_ev).wait()
        bulk = self.h_ev.reshape(NX, NY, NZ, 4)

        # tau=0.6 changes to tau='0p6'
        time_stamp = ('%s'%tau).replace('.', 'p')

        i0, j0, k0 = NX//2, NY//2, NZ//2

        exy = bulk[:, :, k0, 0]
        vx = bulk[:, :, k0, 1]
        vy = bulk[:, :, k0, 2]
        vz2d = bulk[:,:,k0,3].flatten()
        exy2d = bulk[:, :, k0, 0].flatten()
        vx2d = bulk[:, :, k0, 1].flatten()
        vy2d = bulk[:, :, k0, 2].flatten()
        Tp2d = self.eos.f_T(exy2d)


        ed_ijk = bulk[:,:,:,0].flatten()
        vx_ijk = bulk[:,:,:,1].flatten()
        vy_ijk = bulk[:,:,:,2].flatten()
        vz_ijk = bulk[:,:,:,3].flatten()
        Tp_ijk = self.eos.f_T(ed_ijk)


        xline = self.x
        xline2d = np.repeat(xline,NY)
        self.X_2d.extend( xline2d )
        x_ijk = np.repeat(xline,NY*NZ)
        self.X_tijk.extend(x_ijk)



        yline = self.y
        y_ij = np.tile(yline,NX)
        yline2d=np.tile(yline,NX)
        self.Y_2d.extend(yline2d)
        y_ijk = np.repeat(y_ij,NZ)
        self.Y_tijk.extend(y_ijk)

        zline = self.z
        z_ijk = np.tile(zline,NX*NY)
        self.Z_tijk.extend(z_ijk)

        tau_ijk = np.repeat(tau,NX*NY*NZ)
        tau2d= np.repeat(tau,NX*NY)
        frac2d= np.repeat(0,NX*NY)



        self.Tau_tijk.extend(tau_ijk)
        self.ED_tijk.extend(ed_ijk)
        self.Tp_tijk.extend(Tp_ijk)
        self.Vx_tijk.extend(vx_ijk)
        self.Vy_tijk.extend(vy_ijk)
        self.Vz_tijk.extend(vz_ijk)


        self.Tau_2d.extend(tau2d)
        self.ED_2d.extend(exy2d)
        self.Tp_2d.extend(Tp2d)
        self.Vx_2d.extend(vx2d)
        self.Vy_2d.extend(vy2d)
        self.Vz_2d.extend(vz2d)
        self.Frc_2d.extend(frac2d)



        self.eccp_vs_tau.append(self.eccp(exy, vx, vy)[1])
        self.vr.append(self.mean_vr(exy, vx, vy))

        #self.get_total_energy_and_entropy_on_gpu(tau, d_ev)

        ed_cent = exy[i0, j0]

        self.edcent.append(ed_cent)
        self.Tcent.append(self.eos.f_T(ed_cent))

        #ecc1, ecc2 = self.ecc_vs_rapidity(bulk)
        #ecc1, ecc2 = self.ecc_vs_rapidity_on_gpu(tau, d_ev)
        #self.f_hdf5.create_dataset('bulk1d/eccp1_tau%s'%time_stamp, data = ecc1)
        #self.f_hdf5.create_dataset('bulk1d/eccp2_tau%s'%time_stamp, data = ecc2)

        ## ed_x(y=0, z=0), ed_y(x=0, z=0), ed_z(x=0, y=0)
        #self.f_hdf5.create_dataset('bulk1d/ex_tau%s'%time_stamp, data = bulk[:, j0, k0, 0])
        #self.f_hdf5.create_dataset('bulk1d/ey_tau%s'%time_stamp, data = bulk[i0, :, k0, 0])
        #self.f_hdf5.create_dataset('bulk1d/ez_tau%s'%time_stamp, data = bulk[i0, j0, :, 0])

        ## vx_x(y=0, z=0), vy_y(x=0, z=0), vz_z(x=0, y=0)
        #self.f_hdf5.create_dataset('bulk1d/vx_tau%s'%time_stamp, data = bulk[:, j0, k0, 1])
        #self.f_hdf5.create_dataset('bulk1d/vy_tau%s'%time_stamp, data = bulk[i0, :, k0, 2])
        #self.f_hdf5.create_dataset('bulk1d/vz_tau%s'%time_stamp, data = bulk[i0, j0, :, 3])

        ## ed_xy(z=0), ed_xz(y=0), ed_yz(x=0)
        #self.f_hdf5.create_dataset('bulk2d/exy_tau%s'%time_stamp, data = bulk[:, :, k0, 0])
        #self.f_hdf5.create_dataset('bulk2d/exz_tau%s'%time_stamp, data = bulk[:, j0, :, 0])
        #self.f_hdf5.create_dataset('bulk2d/eyz_tau%s'%time_stamp, data = bulk[i0, :, :, 0])

        ## vx_xy(z=0), vx_xz(y=0), vx_yz(x=0)
        #self.f_hdf5.create_dataset('bulk2d/vx_xy_tau%s'%time_stamp, data = bulk[:, :, k0, 1])
        #self.f_hdf5.create_dataset('bulk2d/vx_xz_tau%s'%time_stamp, data = bulk[:, j0, :, 1])
        ##self.f_hdf5.create_dataset('bulk2d/vx_yz_tau%s'%time_stamp, data = bulk[i0, :, :, 1])

        ## vy_xy(z=0), vy_xz(y=0), vy_yz(x=0)
        #self.f_hdf5.create_dataset('bulk2d/vy_xy_tau%s'%time_stamp, data = bulk[:, :, k0, 2])
        ##self.f_hdf5.create_dataset('bulk2d/vy_xz_tau%s'%time_stamp, data = bulk[:, j0, :, 2])
        #self.f_hdf5.create_dataset('bulk2d/vy_yz_tau%s'%time_stamp, data = bulk[i0, :, :, 2])

        ## vz_xy(z=0), vz_xz(y=0), vz_yz(x=0)
        #self.f_hdf5.create_dataset('bulk2d/vz_xy_tau%s'%time_stamp, data = bulk[:, :, k0, 3])
        #self.f_hdf5.create_dataset('bulk2d/vz_xz_tau%s'%time_stamp, data = bulk[:, j0, :, 3])
        ##self.f_hdf5.create_dataset('bulk2d/vz_yz_tau%s'%time_stamp, data = bulk[i0, :, :, 3])

    def eccp(self, ed, vx, vy, vz=0.0):
        ''' eccx = <y*y-x*x>/<y*y+x*x> where <> are averaged 
            eccp = <Txx-Tyy>/<Txx+Tyy> '''
        ed[ed<1.0E-10] = 1.0E-10
        pre = self.eos.f_P(ed)

        vr2 = vx*vx + vy*vy + vz*vz
        vr2[vr2>1.0] = 0.999999
        u0 = 1.0/np.sqrt(1.0 - vr2)

        Tyy = (ed + pre)*u0*u0*vy*vy + pre
        Txx = (ed + pre)*u0*u0*vx*vx + pre
        T0x = (ed + pre)*u0*u0*vx
        v2 = (Txx - Tyy).sum() / (Txx + Tyy).sum()
        v1 = T0x.sum() / (Txx + Tyy).sum()
        return v1, v2

    def mean_vr(self, ed, vx, vy, vz=0.0):
        ''' <vr> = <gamma * ed * sqrt(vx*vx + vy*vy)>/<gamma*ed>
        where <> are averaged over whole transverse plane'''
        ed[ed<1.0E-10] = 1.0E-10
        vr2 = vx*vx + vy*vy + vz*vz
        vr2[vr2>1.0] = 0.999999
        u0 = 1.0/np.sqrt(1.0 - vr2)
        vr = (u0*ed*np.sqrt(vx*vx + vy*vy)).sum() / (u0*ed).sum()
        return vr

    def total_entropy(self, tau, ed, vx, vy, vz=0.0):
        '''get the total entropy (at mid rapidity ) as a function of time'''
        ed[ed<1.0E-10] = 1.0E-10
        vr2 = vx*vx + vy*vy + vz*vz
        vr2[vr2>1.0] = 0.999999
        u0 = 1.0/np.sqrt(1.0 - vr2)
        return (u0*self.eos.f_S(ed)).sum() * tau * self.cfg.DX * self.cfg.DY

    def get_total_energy_and_entropy_on_gpu(self, tau, d_ev):
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
        self.kernel_bulk.total_energy_and_entropy(self.queue, (NX, NY, NZ), None,
                self.a_ed.data, self.a_entropy.data, d_ev,
                self.eos_table, np.float32(tau)).wait()

        volum = tau * self.cfg.DX * self.cfg.DY * self.cfg.DZ

        e_total = cl_array.sum(self.a_ed).get() * volum
        s_total = cl_array.sum(self.a_entropy).get() * volum

        self.energy.append(e_total)
        self.entropy.append(s_total)
        

    def ecc_vs_rapidity(self, bulk):
        ''' bulk = self.h_ev.reshape(NX, NY, NZ, 4)'''
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
        ecc1 = np.empty(NZ)
        ecc2 = np.empty(NZ)
        for k in range(NZ):
            ed = bulk[:,:,k,0]
            vx = bulk[:,:,k,1]
            vy = bulk[:,:,k,2]
            vz = bulk[:,:,k,3]
            ecc1[k], ecc2[k] = self.eccp(ed, vx, vy, vz)
        return ecc1, ecc2

    def ecc_vs_rapidity_on_gpu(self, tau, d_ev):
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
        self.kernel_bulk.eccp_vs_rapidity(self.queue, (NZ*256,), (256,),
                self.a_eccp1.data, self.a_eccp2.data, d_ev,
                self.eos_table, np.float32(tau)).wait()

        return self.a_eccp1.get(), self.a_eccp2.get()
        
    def save(self, viscous_on=False):
        # use absolute path incase call bulkinfo.save() from other directory
        path_out = os.path.abspath(self.cfg.fPathOut)

        np.savetxt(path_out + '/avg.dat',
                   np.array(list(zip(self.time, self.eccp_vs_tau, self.edcent,
                            self.entropy, self.energy, self.vr))),
                   header='tau, eccp, ed(0,0,0), stotal, Etotal, <vr>')

        #self.f_hdf5.create_dataset('coord/tau', data = self.time)
        #self.f_hdf5.create_dataset('coord/x', data = self.x)
        #self.f_hdf5.create_dataset('coord/y', data = self.y)
        #self.f_hdf5.create_dataset('coord/etas', data = self.z)

        #self.f_hdf5.create_dataset('avg/eccp', data = np.array(self.eccp_vs_tau))
        #self.f_hdf5.create_dataset('avg/edcent', data = np.array(self.edcent))
        #self.f_hdf5.create_dataset('avg/Tcent', data = self.eos.f_T(np.array(self.edcent)))
        #self.f_hdf5.create_dataset('avg/entropy', data = np.array(self.entropy))
        #self.f_hdf5.create_dataset('avg/energy', data = np.array(self.energy))
        #self.f_hdf5.create_dataset('avg/vr', data = np.array(self.vr))

        #self.f_hdf5.close()



        #np.savetxt(path_out + '/bulk3D.dat', \
        #np.array(zip(self.Tau_tijk, self.X_tijk, self.Y_tijk, self.Z_tijk, \
        #self.ED_tijk, self.Tp_tijk, self.Vx_tijk, self.Vy_tijk, self.Vz_tijk)), \
        #fmt='%.2f %.2f %.2f %.2f %.8e %.8e %.8e %.8e %.8e',header = 'tau x y z Ed T vx vy veta')

        np.savetxt(path_out + '/bulk2D.dat', \
        np.array(zip(self.Tau_2d, self.X_2d, self.Y_2d, \
        self.ED_2d, self.Tp_2d, self.Vx_2d, self.Vy_2d, self.Vz_2d , self.Frc_2d)), \
        fmt='%.2f %.2f %.2f %.8e %.8e %.8e %.8e %.8e %.1f',header = 'tau x y Ed T vx vy veta frc')

