#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fri 21 Apr 2017 01:27:02 AM CEST

from subprocess import call
import pandas as pd
import os
import numpy as np
import sys
__cwd__, __cwf__ = os.path.split(__file__)
sys.path.append(os.path.join(__cwd__, '../trento'))
import reader
from rotate import rotate


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
        call(['./trento', self.config['projectile'],
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
                    sxy += np.fabs(sd_new)
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
        call(['./trento3d', self.config['projectile'],
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
                sxyz += np.fabs(sd_new)
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
                  'centrality_file':'auau200_cent.csv',
                  'centrality_file_b':'auau200_cent_b.csv',
                  'mean_coeff':'0.0',
                  'std_coeff':'2.9',
                  'skew_coeff':'7.3',
                  'skew_type':'1',
                  'jacobian':'0.75',
                  'fluctuation':'2.0',
                  'nucleon_width':'0.59'}
        super(AuAu200, self).__init__(config)

class RuRu200(Collision):
    def __init__(self):
        config = {'projectile':'Ru',
                  'target':'Ru',
                  'cross_section':4.23,
                  'centrality_file':'ruru200_cent.csv',
                  'centrality_file_b':'ruru200_cent_b.csv',
                  'mean_coeff':'0.0',
                  'std_coeff':'2.9',
                  'skew_coeff':'7.3',
                  'skew_type':'1',
                  'jacobian':'0.75',
                  'fluctuation':'2.0',
                  'nucleon_width':'0.59'}
        super(RuRu200, self).__init__(config)

class Ru2Ru2200(Collision):
    def __init__(self):
        config = {'projectile':'Ru2',
                  'target':'Ru2',
                  'cross_section':4.23,
                  'centrality_file':'ru2ru2200_cent.csv',
                  'centrality_file_b':'ru2ru2200_cent_b.csv',
                  'mean_coeff':'0.0',
                  'std_coeff':'2.9',
                  'skew_coeff':'7.3',
                  'skew_type':'1',
                  'jacobian':'0.75',
                  'fluctuation':'2.0',
                  'nucleon_width':'0.59'}
        super(Ru2Ru2200, self).__init__(config)
       
class Ru3Ru3200(Collision):
    def __init__(self):
        config = {'projectile':'Ru3',
                  'target':'Ru3',
                  'cross_section':4.23,
                  'centrality_file':'ru3ru3200_cent.csv',
                  'centrality_file_b':'ru3ru3200_cent_b.csv',
                  'mean_coeff':'0.0',
                  'std_coeff':'2.9',
                  'skew_coeff':'7.3',
                  'skew_type':'1',
                  'jacobian':'0.75',
                  'fluctuation':'2.0',
                  'nucleon_width':'0.59'}
        super(Ru3Ru3200, self).__init__(config)


class ZrZr200(Collision):
    def __init__(self):
        config = {'projectile':'Zr',
                  'target':'Zr',
                  'cross_section':4.23,
                  'centrality_file':'ZrZr200_cent.csv',
                  'centrality_file_b':'ZrZr200_cent_b.csv',
                  'mean_coeff':'0.0',
                  'std_coeff':'2.9',
                  'skew_coeff':'7.3',
                  'skew_type':'1',
                  'jacobian':'0.75',
                  'fluctuation':'2.0',
                  'nucleon_width':'0.59'}
        super(ZrZr200, self).__init__(config)

class Zr2Zr2200(Collision):
    def __init__(self):
        config = {'projectile':'Zr2',
                  'target':'Zr2',
                  'cross_section':4.23,
                  'centrality_file':'zr2zr2200_cent.csv',
                  'centrality_file_b':'zr2zr2200_cent_b.csv',
                  'mean_coeff':'0.0',
                  'std_coeff':'2.9',
                  'skew_coeff':'7.3',
                  'skew_type':'1',
                  'jacobian':'0.75',
                  'fluctuation':'2.0',
                  'nucleon_width':'0.59'}
        super(Zr2Zr2200, self).__init__(config)
       
class Zr3Zr3200(Collision):
    def __init__(self):
        config = {'projectile':'Zr3',
                  'target':'Zr3',
                  'cross_section':4.23,
                  'centrality_file':'Zr3Zr3200_cent.csv',
                  'centrality_file_b':'Zr3Zr3200_cent_b.csv',
                  'mean_coeff':'0.0',
                  'std_coeff':'2.9',
                  'skew_coeff':'7.3',
                  'skew_type':'1',
                  'jacobian':'0.75',
                  'fluctuation':'2.0',
                  'nucleon_width':'0.59'}
        super(Zr3Zr3200, self).__init__(config)


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
class OO6500(Collision):
    def __init__(self):
        config = {'projectile':'O',
                  'target':'O',
                  'cross_section':7.25,
                  'centrality_file':'OO6500_cent.csv',
                  'centrality_file_b':'OO6500_cent_b.csv',
                  'mean_coeff':'0.0',
                  'std_coeff':'2.9',
                  'skew_coeff':'7.3',
                  'skew_type':'1',
                  'jacobian':'0.75',
                  'fluctuation':'2.0',
                  'nucleon_width':'0.59'}
        super(OO6500, self).__init__(config)

class ArAr5850(Collision):
    def __init__(self):
        config = {'projectile':'Ar',
                  'target':'Ar',
                  'cross_section':7.0,
                  'centrality_file':'ArAr5850_cent.csv',
                  'centrality_file_b':'ArAr5850_cent_b.csv',
                  'mean_coeff':'0.0',
                  'std_coeff':'2.9',
                  'skew_coeff':'7.3',
                  'skew_type':'1',
                  'jacobian':'0.75',
                  'fluctuation':'2.0',
                  'nucleon_width':'0.59'}
        super(ArAr5850, self).__init__(config)


if __name__=='__main__':
    xexe = Xe2Xe25440()
    xexe.create_ini3D('0_6', './dat', num_of_events=2, one_shot_ini=True)
    #pbpb = PbPb5020()
    #pbpb.create_ini3D('0_6','./dat1',num_of_events=10,one_shot_ini=True)
