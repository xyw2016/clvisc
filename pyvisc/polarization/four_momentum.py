#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Fri 24 Apr 2015 14:10:48 CEST

import numpy as np
from scipy.interpolate import interp1d, splrep, splint
from scipy.integrate import quad

HBARC = 0.19732

gaulep48 = np.array([
    0.998771007252426118601,	0.993530172266350757548,
    0.984124583722826857745,	0.970591592546247250461,
    0.952987703160430860723,	0.931386690706554333114,
    0.905879136715569672822,	0.876572020274247885906,
    0.843588261624393530711,	0.807066204029442627083,
    0.767159032515740339254,	0.724034130923814654674,
    0.677872379632663905212,	0.628867396776513623995,
    0.577224726083972703818,	0.523160974722233033678,
    0.466902904750958404545,	0.408686481990716729916,
    0.348755886292160738160,	0.287362487355455576736,
    0.224763790394689061225,	0.161222356068891718056,
    0.097004699209462698930,	0.032380170962869362033 ])

gaulew48 = np.array([
    0.003153346052305838633,	0.007327553901276262102,
    0.011477234579234539490,	0.015579315722943848728,
    0.019616160457355527814,	0.023570760839324379141,
    0.027426509708356948200,	0.031167227832798088902,
    0.034777222564770438893,	0.038241351065830706317,
    0.041545082943464749214,	0.044674560856694280419,
    0.047616658492490474826,	0.050359035553854474958,
    0.052890189485193667096,	0.055199503699984162868,
    0.057277292100403215705,	0.059114839698395635746,
    0.060704439165893880053,	0.062039423159892663904,
    0.063114192286254025657,	0.063924238584648186624,
    0.064466164435950082207,	0.064737696812683922503 ])

gala15x = np.array([	0.093307812017,         0.492691740302,
    1.215595412071,         2.269949526204,
    3.667622721751,         5.425336627414,
    7.565916226613,        10.120228568019, 
    13.130282482176,        16.654407708330,
    20.776478899449,        25.623894226729,
    31.407519169754,        38.530683306486,
    48.026085572686	])

gala15w = np.array([	0.239578170311,         0.560100842793,
    0.887008262919,         1.22366440215,
    1.57444872163,          1.94475197653,
    2.34150205664,          2.77404192683,
    3.25564334640,          3.80631171423,
    4.45847775384,          5.27001778443,
    6.35956346973,          8.03178763212,
    11.5277721009   ])

# NY = 41
# NPT = 15
# NPHI = 48
# INVP = 1/12.0

# # used as rapidity or pseudo-rapidity
# Y = np.linspace( -8, 8, NY, endpoint=True )

# # used as transverse momentum 
# PT = INVP * gala15x

# # used as azimuthal angle
# PHI = np.zeros( NPHI )
# PHI[0:NPHI/2] = np.pi*(1.0-gaulep48)
# PHI[NPHI-1:NPHI/2-1:-1] = np.pi*(1.0+gaulep48)


NY = 41
NPT = 61
NPHI = 48
INVP = 1/12.0

# used as rapidity or pseudo-rapidity
Y = np.linspace( -2, 2, NY, endpoint=True )

# used as transverse momentum 
PT = np.linspace( 0, 3, NPT, endpoint=True )
PHI = np.linspace( 0, 2*np.pi, NPHI, endpoint=True )

# used as azimuthal angle
#PHI = np.zeros( NPHI )
#PHI[0:NPHI/2] = np.pi*(1.0-gaulep48)
#PHI[NPHI-1:NPHI/2-1:-1] = np.pi*(1.0+gaulep48)


print("Y=", Y)

print("Pt=", PT)

print("Phi=", PHI)

# def pt_integral(spec_along_pt):
#     '''1D integration along transverse momentum'''
#     return (spec_along_pt*gala15w*INVP).sum()

# def phi_integral(spec_along_phi):
#     '''1D integration along azimuthal angle'''
#     return np.pi*((spec_along_phi[0:NPHI/2] + 
#         spec_along_phi[NPHI-1:NPHI/2-1:-1])*gaulew48).sum()

# def rapidity_integral(spec_along_y, ylo=-0.5, yhi=0.5):
#     '''1D integration along rapidity/pseudo-rapidity 
#     The spline interpolation and integration is much faster than
#     the interp1d() and quad combination'''
#     #f = interp1d(Y, spec_along_y, kind='cubic')
#     #return quad(f, ylo, yhi, epsrel=1.0E-5)[0]
#     tck = splrep(Y, spec_along_y)
#     return splint(ylo, yhi, tck)

# def pt_phi_integral(spec):
#     '''2D integration along pt and phi,
#     The spec is: dN/dYptdptdphi'''
#     spec_along_pt = np.empty(NPT)
#     for i in range(NPT):
#         spec_along_pt[i] = PT[i] * phi_integral(spec[i, :])
#     return pt_integral(spec_along_pt)

if __name__ == '__main__':
    print(phi_integral(np.sin(0.25*PHI)))
    print(pt_integral(np.exp(-PT)))
