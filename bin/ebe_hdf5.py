#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com

import numpy as np
from subprocess import call
import os
from time import time
from glob import glob
import pyopencl as cl
import matplotlib.pyplot as plt
import h5py

import os, sys
cwd, cwf = os.path.split(__file__)
sys.path.append(os.path.join(cwd, '../pyvisc'))
from config import cfg, write_config
from visc import CLVisc



def read_p4x4(cent='30_35', idx=0,
        fname='/lustre/nyx/hyihp/lpang/hdf5_data/auau200.h5'):
    '''read 4-momentum and 4-coordiantes from h5 file,
    return: np.array with shape (num_of_partons, 8)
    the first 4 columns store: E, px, py, pz
    the last 4 columns store: t, x, y, z'''
    with h5py.File(fname, 'r') as f:
        grp = f['cent']
        event_id = grp[cent][:, 0].astype(np.int)

        impact = grp[cent][:, 1]
        nw = grp[cent][:, 2]
        nparton = grp[cent][:, 3]
        key = 'event%s'%event_id[idx]
        #print key, nw[0], nparton[0]
        p4x4 = f[key]
        return p4x4[...], event_id[idx], impact[idx], nw[idx], nparton[idx]

def event_by_event(fout, cent='30_35', idx=0, etaos=0.0):
    ''' Run event_by_event hydro, with initial condition 
    from smearing on the particle list'''
    if not os.path.exists(fout):
        os.mkdir(fout)
    cfg.NX = 301
    cfg.NY = 301
    cfg.NZ = 101

    cfg.DT = 0.005
    cfg.DX = 0.1
    cfg.DY = 0.1
    cfg.DZ = 0.15
    cfg.IEOS = 4
    cfg.TFRZ = 0.136

    cfg.ntskip = 60
    cfg.nzskip = 2

    cfg.TAU0 = 0.4
    cfg.ETAOS = etaos
    cfg.fPathOut = fout

    t0 = time()
    visc = CLVisc(cfg, gpu_id=3)

    parton_list, eid, imp_b, nwound, npartons = read_p4x4(cent, idx)

    comments = 'cent=%s, eventid=%s, impact parameter=%s, nw=%s, npartons=%s'%(
            cent, eid, imp_b, nwound, npartons)

    write_config(cfg, comments)

    visc.smear_from_p4x4(parton_list, SIGR=0.6, SIGZ=0.6, KFACTOR=1.3)

    visc.evolve(max_loops=4000, save_hypersf=True, save_bulk=True, save_vorticity=True)

    # test whether queue.finish() fix the opencl memory leak problem
    visc.queue.finish()

    t1 = time()
    print('finished. Total time: {dtime}'.format(dtime = t1-t0))


if __name__ == '__main__':
    cent = '20_25'
    #path = '/lustre/nyx/hyihp/lpang/auau200_results/visc_etas0p08/cent20_25_etas0p08/'
    path = '/lustre/nyx/hyihp/lpang/auau200_results/ideal/cent20_25_etas0p00/'

    etaos = 0.08

    for idx in range(50, 100):
        fpath_out = path + 'cent%s_event%s'%(cent, idx)
        event_by_event(fpath_out, cent, idx, etaos=etaos)

        cwd = os.getcwd()
        os.chdir('../CLSmoothSpec/build')
        #os.system('cmake -D VISCOUS_ON=ON ..')
        #os.system('make')
        call(['./spec', fpath_out])
        os.chdir(cwd)
        call(['python', '../spec/main.py', fpath_out])
