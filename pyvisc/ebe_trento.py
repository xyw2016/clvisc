#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com

import numpy as np
from subprocess import call
import os
from time import time
from glob import glob
import pyopencl as cl
#import matplotlib.pyplot as plt
import h5py
from ini.trento import AuAu200, PbPb2760, PbPb5020,Xe2Xe25440,RuRu200,Ru2Ru2200,Ru3Ru3200,ZrZr200,Zr2Zr2200,Zr3Zr3200
from scipy.interpolate import InterpolatedUnivariateSpline
import gc
import os, sys
cwd, cwf = os.path.split(__file__)
print('cwd=', cwd)

sys.path.append(os.path.join(cwd, '../pyvisc'))
from config import cfg, write_config
from visc import CLVisc

import h5py

def from_sd_to_ed(entropy, eos):
    '''using eos to  convert the entropy density to energy density'''
    s = eos.s
    ed = eos.ed
    # the InterpolatedUnivariateSpline works for both interpolation
    # and extrapolation
    f_ed = InterpolatedUnivariateSpline(s, ed, k=1)
    return f_ed(entropy)


def ebehydro(fpath, cent='0_5', etaos=0.12, gpu_id=0, system='pbpb2760', boost_invariance=True,oneshot=True):
    ''' Run event_by_event hydro, with initial condition 
    from smearing on the particle list'''

    fout = fpath
    if not os.path.exists(fout):
        os.mkdir(fout)

    cfg.NX = 66 
    cfg.NY = 66
    cfg.NZ = 67 
    cfg.DT = 0.02
    cfg.DX = 0.3
    cfg.DY = 0.3
    cfg.DZ = 0.3

    cfg.ntskip = 10 
    cfg.nxskip = 4 
    cfg.nyskip = 4
    cfg.nzskip = 2 

    cfg.eos_type = 'hotqcd2014'
    #cfg.eos_type = 'lattice_pce150'
    cfg.TAU0 = 0.6
    cfg.fPathOut = fout

    cfg.TFRZ = 0.154

    cfg.ETAOS_XMIN = 0.16

    cfg.ETAOS_YMIN = 0.16
    cfg.ETAOS_RIGHT_SLOP = 0.0
    cfg.ETAOS_LEFT_SLOP =  0.0

    cfg.save_to_hdf5 = True

    # for auau
    if system == 'auau200':
        cfg.Eta_gw = 1.3
        cfg.Eta_flat = 1.5
        comments = 'au+au IP-Glasma'
        collision = AuAu200()
        scale_factor = 57.0

    elif system == 'RuRu200':
        cfg.Eta_gw = 1.3
        cfg.Eta_flat = 1.5
        comments = 'Ru+Ru IP-Glasma'
        collision = RuRu200()
        scale_factor = 57.0
    
    elif system == 'Ru2Ru2200':
        cfg.Eta_gw = 1.3
        cfg.Eta_flat = 1.5
        comments = 'Ru2+Ru2 IP-Glasma'
        collision = Ru2Ru2200()
        scale_factor = 57.0
    
    elif system == 'Ru3Ru3200':
        cfg.Eta_gw = 1.3
        cfg.Eta_flat = 1.5
        comments = 'Ru3+Ru3 IP-Glasma'
        collision = Ru3Ru3200()
        scale_factor = 57.0

    elif system == 'ZrZr200':
        cfg.Eta_gw = 1.3
        cfg.Eta_flat = 1.5
        comments = 'Zr+Zr IP-Glasma'
        collision = RuRu200()
        scale_factor = 57.0
    
    elif system == 'Zr2Zr2200':
        cfg.Eta_gw = 1.3
        cfg.Eta_flat = 1.5
        comments = 'Zr2+Zr2 IP-Glasma'
        collision = Ru2Ru2200()
        scale_factor = 57.0
    
    elif system == 'Zr3Zr3200':
        cfg.Eta_gw = 1.3
        cfg.Eta_flat = 1.5
        comments = 'Zr3+Zr3 IP-Glasma'
        collision = Ru3Ru3200()
        scale_factor = 57.0

    elif system == 'Xe2Xe25440':
        comments = 'Xe2+Xe2'
        collision = Xe2Xe25440()
        scale_factor = 130
    elif system == 'OO6500':
        comments = 'O+O'
        cfg.Eta_gw = 2.0
        cfg.Eta_flat = 1.7
        collision = OO6500()
        scale_factor = 180.0
    elif system == 'ArAr5850':
        comments = 'Ar+Ar'
        cfg.Eta_gw = 2.0
        cfg.Eta_flat = 1.7
        collision = ArAr5850()
        scale_factor = 160.0
    # for pbpb
    else:
        cfg.Eta_gw = 1.8
        cfg.Eta_flat = 2.0
        comments = 'pb+pb IP-Glasma'
        if system == 'pbpb2760':
            collision = PbPb2760()
            scale_factor = 118.0
        elif system == 'pbpb5020':
            cfg.Eta_flat = 2.2
            collision = PbPb5020()
            scale_factor = 130.0

    grid_max = np.floor(cfg.NX/2) * cfg.DX
    eta_max = np.floor(cfg.NZ/2) * cfg.DZ

    fini = os.path.join(fout, 'trento_ini/')

    if os.path.exists(fini):
        call(['rm', '-r', fini])

    ev = np.zeros((cfg.NX*cfg.NY*cfg.NZ, 4), cfg.real)
    if boost_invariance:
        print ("################# run Trento 2D ###################")
        cwd = os.getcwd()
        os.chdir("../3rdparty/trento_with_participant_plane/build/src/")
        if oneshot:
            print ("################# oneshot ###################")
            collision.create_ini(cent, fini, num_of_events=100,
                             grid_max=grid_max, grid_step=cfg.DX,
                             one_shot_ini=oneshot)
            s = np.loadtxt(os.path.join(fini, 'one_shot_ini.dat'))
        else:
            collision.create_ini(cent, fini, num_of_events=1,
                             grid_max=grid_max, grid_step=cfg.DX,
                             one_shot_ini=oneshot)
            s = np.loadtxt(os.path.join(fini, '0.dat'))
         
        os.chdir(cwd)
        smax = s.max()
        s_scale = s * scale_factor
        t0 = time()

        visc = CLVisc(cfg, gpu_id=gpu_id)

        ed = from_sd_to_ed(s_scale, visc.ideal.eos)

        

        # repeat the ed(x,y) NZ times
        ev[:, 0] = np.repeat((ed.T).flatten(), cfg.NZ)

        eta_max = cfg.NZ//2 * cfg.DZ
        eta = np.linspace(-eta_max, eta_max, cfg.NZ)

        heta = np.ones(cfg.NZ)

        fall_off = np.abs(eta) > cfg.Eta_flat
        eta_fall = np.abs(eta[fall_off])
        heta[fall_off] = np.exp(-(eta_fall - cfg.Eta_flat)**2/(2.0*cfg.Eta_gw**2))

        # apply the heta longitudinal distribution
        ev[:, 0] *= np.tile(heta, cfg.NX * cfg.NY)
    else:

        print ("################# run Trento 3D ###################",system)
        cwd = os.getcwd()
        os.chdir("../3rdparty/trento3d-master/build/src/")
        if oneshot:
            print ("################# oneshot ###################")
            collision.create_ini3D(cent, fini, num_of_events=100,
                             grid_max=grid_max, grid_step=cfg.DX,
                             eta_max=eta_max, eta_step=cfg.DZ,
                             one_shot_ini=oneshot)
            s = np.loadtxt(os.path.join(fini, 'one_shot_ini.dat'))
        else:
            collision.create_ini3D(cent, fini, num_of_events=1,
                             grid_max=grid_max, grid_step=cfg.DX,
                             eta_max=eta_max, eta_step=cfg.DZ,
                             one_shot_ini=oneshot)
            s = np.loadtxt(os.path.join(fini, '0.dat'))
        os.chdir(cwd)
        smax = s.max()
        s_scale = s * scale_factor
        t0 = time()

        visc = CLVisc(cfg, gpu_id=gpu_id)

        ed = from_sd_to_ed(s_scale, visc.ideal.eos)
        ed = ed.reshape((cfg.NY,cfg.NX,cfg.NZ))
        ev[:, 0] = ed.transpose((1,0,2)).flatten() 

        



    visc.ideal.load_ini(ev)

    visc.evolve(max_loops=4000, save_hypersf=True, save_bulk=True, save_vorticity=False)

    write_config(cfg, comments)
    t1 = time()
    print('finished. Total time: {dtime}'.format(dtime = t1-t0))

    # get particle spectra from MC sampling and force decay
    #call(['python', 'spec.py', '--event_dir', cfg.fPathOut,
    #  '--viscous_on', "true", "--reso_decay", "true", "--nsampling", "2000",
    #  '--mode', 'mc'])

     # calc the smooth particle spectra
    #call(['python', 'spec.py', '--event_dir', cfg.fPathOut,
    #  '--viscous_on', "true", "--reso_decay", "true", 
    #  '--mode', 'smooth'])
 
def main(path, cent='0_5', gpu_id=0, event0=0, event1=1, system='pbpb2760'):
    fpath_out = os.path.abspath(path)
    for i in range(event0, event1):
        fout = os.path.join(fpath_out, 'event%s'%i)
        if not os.path.exists(fout):
            os.makedirs(fout)
        ebehydro(fout, cent, system = system)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 6:
        #path_base = '/lustre/nyx/hyihp/lpang/trento_ebe_hydro/results/'
        path_base = './results/'
        coll_sys = sys.argv[1]
        cent = sys.argv[2]
        gpu_id = int(sys.argv[3])
        path = os.path.join(path_base, coll_sys, cent)
        event0 = int(sys.argv[4])
        event1 = int(sys.argv[5])
        main(path, cent, gpu_id=gpu_id, event0=event0,event1=event1, system=coll_sys)
