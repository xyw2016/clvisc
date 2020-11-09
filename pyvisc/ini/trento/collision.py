#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fri 21 Apr 2017 01:27:02 AM CEST

from subprocess import call
import pandas as pd
import os
import numpy as np
import reader
from rotate import rotate

__cwd__, __cwf__ = os.path.split(__file__)

class Collision(object):
    def __init__(self, config):
        self.config = config
        centrality_file = os.path.join(__cwd__, config['centrality_file'])
        self.info = pd.read_csv(centrality_file)


    def get_smin_smax(self, cent='0_6'):
        '''get min/max initial total entropy for one
        centrality class, stored in auau200.csv or ...'''
        clow, chigh = cent.split('_')
        smin = self.entropy_bound(cent_bound = float(chigh))
        smax = self.entropy_bound(cent_bound = float(clow))
        return smin, smax

    def entropy_bound(self, cent_bound=5):
        '''get entropy value for one specific centrality bound'''
        self.info.set_index(['cent'])
        cents = self.info['cent']
        entropy = self.info['entropy']
        return np.interp(cent_bound, cents, entropy)

    def get_bmin_bmax(self, cent='0_6'):
        '''get min/max initial total entropy for one
        centrality class, stored in auau200.csv or ...'''
        clow, chigh = cent.split('_')
        bmin = self.b_bound(cent_bound = float(chigh))
        bmax = self.b_bound(cent_bound = float(clow))
        return bmin, bmax

    def b_bound(self, cent_bound=5):
        '''get entropy value for one specific centrality bound'''
        self.info_b.set_index(['cent'])
        cents = self.info_b['cent']
        b = self.info_b['b']
        return np.interp(cent_bound, cents, b)


    def create_ini(self, cent, output_path,
                   grid_max=15.0, grid_step=0.1, num_of_events=1,
                   one_shot_ini=False, align_for_oneshot=False):
        smin, smax = self.get_smin_smax(cent)
        call(['trento', self.config['projectile'],
              self.config['target'],
              '%s'%num_of_events,
              '-o', output_path,
              '-x', '%s'%self.config['cross_section'],
              '--s-min', '%s'%smin,
              '--s-max', '%s'%smax,
              '--grid-max', '%s'%grid_max,
              '--grid-step', '%s'%grid_step])

        if one_shot_ini:
            ngrid = int(2 * grid_max / grid_step)
            sxy = np.zeros((ngrid, ngrid), dtype=np.float32)
            events = os.listdir(output_path)
            print(events)
            num_of_events = 0
            for event in events:
                try:
                    fname = os.path.join(output_path, event)
                    dat = np.loadtxt(fname).reshape(ngrid, ngrid)
                    opt = reader.get_comments(fname)
                    sd_new = rotate(dat, opt['ixcm'], opt['iycm'], opt['phi_2'], ngrid, ngrid)
                    sxy += sd_new 
                    num_of_events += 1
                except:
                    print(fname, 'is not a trento event')
            np.savetxt(os.path.join(output_path, "one_shot_ini.dat"), sxy/num_of_events, header=cent)
    
    def create_ini3D(self, cent, output_path,num_of_events=1,
                   grid_max=12.0, grid_step=0.12, 
                   eta_max=9.9,eta_step=0.3,
                   one_shot_ini=False, align_for_oneshot=False):
        output_path = os.path.abspath(output_path)
        centrality_file = os.path.join(__cwd__, self.config['centrality_file_b'])
        self.info_b = pd.read_csv(centrality_file)
        bmin, bmax = self.get_bmin_bmax(cent)
        #cwd1 = os.getcwd()
        #os.chdir("../../../3rdparty/trento3d-master/build/src/")
        #print os.getcwd()
        #print output_path 
        call(['trento3d', self.config['projectile'],
              self.config['target'],
              '%s'%num_of_events,
              '-o', output_path,
              '-x', '%s'%self.config['cross_section'],
              '--b-min', '%s'%bmin,
              '--b-max', '%s'%bmax,
              '--xy-max', '%s'%grid_max,
              '--xy-step', '%s'%grid_step,
              '--eta-max','%s'%eta_max,
              '--eta-step','%s'%eta_step,
              '--mean-coeff',self.config['mean_coeff'],
              '--std-coeff',self.config['std_coeff'],
              '--skew-coeff',self.config['skew_coeff'],
              '--skew-type',self.config['skew_type'],
              '--jacobian',self.config['jacobian'],
              '--fluctuation',self.config['fluctuation'],
              '--nucleon-width',self.config['nucleon_width']])

        #os.chdir(cwd1)
        if one_shot_ini:
            ngridxy = int(2 * grid_max / grid_step)
            ngrideta = int(2*eta_max/eta_step)+1
            sxyz = np.zeros((ngridxy, ngridxy,ngrideta), dtype=np.float32)
            events = os.listdir(output_path)
            print(events)
            num_of_events = 0
            for event in events:
                #try:
                fname = os.path.join(output_path, event)
                dat = np.loadtxt(fname).reshape(ngridxy, ngridxy,ngrideta)
                opt = reader.get_comments(fname)
                sd_new = rotate(dat, opt['ixcm'], opt['iycm'], opt['phi_2'], ngridxy, ngridxy,ngrideta,is3D=True)
                sxyz += sd_new 
                num_of_events += 1
                #except:
                #    print(fname, 'is not a trento event')
            sxyz = sxyz.flatten()
            np.savetxt(os.path.join(output_path, "one_shot_ini.dat"), sxyz/num_of_events, header=cent)


class AuAu200(Collision):
    def __init__(self):
        config = {'projectile':'Au',
                  'target':'Au',
                  'cross_section':4.23,
                  'centrality_file':'auau200_cent.csv'}
        super(AuAu200, self).__init__(config)

       

class PbPb2760(Collision):
    def __init__(self):
        config = {'projectile':'Pb',
                  'target':'Pb',
                  'cross_section':6.4,
                  'centrality_file':'pbpb2760_cent.csv',
                  'centrality_file_b':'pbpb2760_cent_b.csv',
                  'mean_coeff':'0.0',
                  'std_coeff':'2.9',
                  'skew_coeff':'7.3',
                  'skew_type':'1',
                  'jacobian':'0.75',
                  'fluctuation':'2.0',
                  'nucleon_width':'0.59'}
        super(PbPb2760, self).__init__(config)

       

class PbPb5020(Collision):
    def __init__(self):
        config = {'projectile':'Pb',
                  'target':'Pb',
                  'cross_section':7.0,
                  'centrality_file':'pbpb5020_cent.csv',
                  'centrality_file_b':'pbpb5020_cent_b.csv',
                  'mean_coeff':'0.0',
                  'std_coeff':'2.9',
                  'skew_coeff':'7.3',
                  'skew_type':'1',
                  'jacobian':'0.75',
                  'fluctuation':'2.0',
                  'nucleon_width':'0.59'}
        super(PbPb5020, self).__init__(config)

class Xe2Xe25440(Collision):
    def __init__(self):
        config = {'projectile':'Xe2',
                  'target':'Xe2',
                  'cross_section':7.0,
                  'centrality_file':'xexe5440_cent.csv',
                  'centrality_file_b':'xe2xe25440_cent_b.csv',
                  'mean_coeff':'0.0',
                  'std_coeff':'2.9',
                  'skew_coeff':'7.3',
                  'skew_type':'1',
                  'jacobian':'0.75',
                  'fluctuation':'2.0',
                  'nucleon_width':'0.59'}
        super(Xe2Xe25440, self).__init__(config)

class XeXe5440(Collision):
    def __init__(self):
        config = {'projectile':'Xe',
                  'target':'Xe',
                  'cross_section':7.0,
                  'centrality_file':'xexe5440_cent.csv',
                  'centrality_file_b':'xexe5440_cent_b.csv',
                  'mean_coeff':'0.0',
                  'std_coeff':'2.9',
                  'skew_coeff':'7.3',
                  'skew_type':'1',
                  'jacobian':'0.75',
                  'fluctuation':'2.0',
                  'nucleon_width':'0.59'}
        super(XeXe5440, self).__init__(config)

if __name__=='__main__':
    xexe = Xe2Xe25440()
    xexe.create_ini3D('0_6', './dat', num_of_events=2, one_shot_ini=True)
    #pbpb = PbPb5020()
    #pbpb.create_ini3D('0_6','./dat1',num_of_events=10,one_shot_ini=True)
