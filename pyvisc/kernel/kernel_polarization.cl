# include <real_type.h>

// get the round_up value 
int round_up(int value, int multiple) {
    int rem = value % multiple;
    if ( rem != 0 ) {
        value += multiple - rem;
    }
    return value;
}

// calc the Lambda polarization on freeze out hyper surface
// Each local work group is used to calculate polarization for one rapidity
__kernel void polarization_on_sf(
        __global real4 * d_pol_lab,
        __global real * d_density,
        __global real4 * d_pol_lrf,
        __global const real8 * d_sf,
        __global const real * d_omega_sf,
        __global const real4 * d_momentum_list)
{
    __local real4 subpol[BSZ];
    __local real  subspec[BSZ];

    int lid = get_local_id(0);
    int gid = get_group_id(0);

    real4 mom = d_momentum_list[gid];
    real mt = mom.s0;
    real rapidity = mom.s1;
    real px = mom.s2;
    real py = mom.s3;
    real Tfrz = TFRZ;
    real mass = MASS;
    real mu = Mu;

    real8 sf;
    real omega_tx, omega_ty, omega_tz, omega_xy, omega_xz, omega_yz;

    //
    real4 pol_lab = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
    real density = 0.0f;

    // notice there if size_sf is not multiplies of BSZ, some threads can never
    // hit the barrier(CLK_LOCAL_MEM_FENCE), the program will abort
    int nsf = round_up(SIZE_SF, BSZ);


    for ( int i = get_local_id(0); i < nsf; i += get_local_size(0) ) {
        
        if ( i < SIZE_SF ) {
        sf = d_sf[i];
        omega_tx = d_omega_sf[6*i + 0];
        omega_ty = d_omega_sf[6*i + 1];
        omega_tz = d_omega_sf[6*i + 2];
        omega_xy = d_omega_sf[6*i + 3];
        omega_xz = d_omega_sf[6*i + 4];
        omega_yz = d_omega_sf[6*i + 5];
        
        real4 umu = (real4)(1.0f, sf.s4, sf.s5, sf.s6) * \
            1.0f/sqrt(max((real)1.0E-10f, \
            (real)(1.0f-sf.s4*sf.s4 - sf.s5*sf.s5 - sf.s6*sf.s6)));
        // u^{mu}
        real4 dsigma = sf.s0123;
        real etas = sf.s7;

        real Y = rapidity;
        real mtcosh = mt * cosh(Y - etas);
        real mtsinh = mt * sinh(Y - etas);
        real4 pmu = (real4)(mtcosh, -px, -py, -mtsinh); //p_mu
        
        real beta = 1.0/Tfrz;
        real pdotu = dot(umu, pmu);
        real tmp = exp(-pdotu*beta);
        real feq = tmp/(1.0f + tmp);
        //real tmp = (pdotu - mu )*beta;
        //real feq = 1.0/(exp(tmp) + 1);
        real volum = dot(dsigma, pmu);

        // pomega^a = Omega^{a b} p_b
        real pomega_0 =  omega_tx * pmu.s1 + omega_ty * pmu.s2 + omega_tz * pmu.s3;
        real pomega_1 = -omega_tx * pmu.s0 + omega_xy * pmu.s2 + omega_xz * pmu.s3;
        real pomega_2 = -omega_ty * pmu.s0 - omega_xy * pmu.s1 + omega_yz * pmu.s3;
        real pomega_3 = -omega_tz * pmu.s0 - omega_xz * pmu.s1 - omega_yz * pmu.s2;

        real4 pomega = (real4)(pomega_0, pomega_1, pomega_2, pomega_3);
        // pomega^a

        
            subspec[lid] = volum * feq;
            subpol[lid] = volum * pomega * feq * (1.0f - feq);
            real pol0 = subpol[lid].s0;
            real pol3 = subpol[lid].s3;
            subpol[lid].s0 = cosh(etas)*pol0 + sinh(etas)*pol3;
            subpol[lid].s3 = sinh(etas)*pol0 + cosh(etas)*pol3;
            

            // t-z coordinate  

        } else {
            subspec[lid] = 0.0f;
            subpol[lid] = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // parallel reduction 
        for ( unsigned int s = get_local_size(0) >> 1; s > 0; s >>= 1 ) {
            if ( lid < s ) {
                subspec[lid] += subspec[lid + s];
                subpol[lid] += subpol[lid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        if ( lid == 0 ) {
             density += subspec[0];
             pol_lab += subpol[0];
        }
    }
    
    // save the final results to global memory
    if ( lid == 0 ) {
        d_density[gid] = density;
        d_pol_lab[gid] = pol_lab/(4.0f*mass);
        //printf ("%.12f %.12f \n",density,pol_lab.s3/(4.0f*mass));
        
        
        

        // polarization at (mt, Y, px, py) in lab frame
        //real4 Pi_lab = pol_lab/(4.0f*mass*density);
        real4 Pi_lab = pol_lab/(4.0f*mass*density);

        // calc the polarization in lrf
        real p0 = mt * cosh(rapidity);
        real pz = mt * sinh(rapidity);
        real4 p_mu = (real4)(p0, -px, -py, -pz);

        real Pi0 = dot(p_mu, Pi_lab)/mass;

        real3 Pi123 = Pi_lab.s123 - dot(p_mu.s123, Pi_lab.s123)*p_mu.s123/(p0*(p0+mass));

        d_pol_lrf[gid] = (real4)(Pi0, Pi123);


    }
}


__kernel void polarization_shear_on_sf(
        __global real4 * d_pol_lab,
        __global real * d_density,
        __global real4 * d_pol_lrf,
        __global const real8 * d_sf,
        __global const real * d_omega_shear1_sf,
        __global const real * d_omega_shear2_sf,
        __global const real4 * d_momentum_list)
{
    __local real4 subpol[BSZ];
    __local real  subspec[BSZ];

    int lid = get_local_id(0);
    int gid = get_group_id(0);
    
    real hbarc = 0.19733f;
    real4 mom = d_momentum_list[gid];
    real mt = mom.s0;
    real rapidity = mom.s1;
    real px = mom.s2;
    real py = mom.s3;
    real Tfrz = TFRZ;
    real mass = MASS;
    real mu = Mu;

    real8 sf;
    real omega_tx, omega_ty, omega_tz, omega_xy, omega_xz, omega_yz;


    //
    real4 pol_lab = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
    real density = 0.0f;

    // notice there if size_sf is not multiplies of BSZ, some threads can never
    // hit the barrier(CLK_LOCAL_MEM_FENCE), the program will abort
    int nsf = round_up(SIZE_SF, BSZ);


    for ( int i = get_local_id(0); i < nsf; i += get_local_size(0) ) {
        
        if ( i < SIZE_SF ) {
        sf = d_sf[i];
        
        real4 umu = (real4)(1.0f, sf.s4, sf.s5, sf.s6) * \
            1.0f/sqrt(max((real)1.0E-10f, \
            (real)(1.0f-sf.s4*sf.s4 - sf.s5*sf.s5 - sf.s6*sf.s6)));
        // u^{mu}

        real4 umu1 = (real4)(1.0f, -sf.s4, -sf.s5, -sf.s6) * \
            1.0f/sqrt(max((real)1.0E-10f, \
            (real)(1.0f-sf.s4*sf.s4 - sf.s5*sf.s5 - sf.s6*sf.s6)));
        // u_{mu}
        real4 dsigma = sf.s0123;
        real etas = sf.s7;

        real Y = rapidity;
        real mtcosh = mt * cosh(Y - etas);
        real mtsinh = mt * sinh(Y - etas);
        real4 pmu1 = (real4)(mtcosh, -px, -py, -mtsinh); //p_mu
        real4 pmu2 = (real4)(mtcosh, px, py, mtsinh); //p^mu

        real4 dudx_sym[4] = { (real4)( d_omega_shear1_sf[16*i + 0]*2.0f, 
        d_omega_shear1_sf[16*i + 4] + d_omega_shear1_sf[16*i + 1],
        d_omega_shear1_sf[16*i + 8] + d_omega_shear1_sf[16*i + 2],
        d_omega_shear1_sf[16*i + 12] + d_omega_shear1_sf[16*i + 3]),
        (real4)( d_omega_shear1_sf[16*i + 1] + d_omega_shear1_sf[16*i + 4] , 
        d_omega_shear1_sf[16*i + 5] *2.0f,
        d_omega_shear1_sf[16*i + 9] + d_omega_shear1_sf[16*i + 6],
        d_omega_shear1_sf[16*i + 13] + d_omega_shear1_sf[16*i + 7]),
        (real4)( d_omega_shear1_sf[16*i + 2] + d_omega_shear1_sf[16*i + 8] , 
        d_omega_shear1_sf[16*i + 6] + d_omega_shear1_sf[16*i + 9] ,
        d_omega_shear1_sf[16*i + 10] * 2.0f,
        d_omega_shear1_sf[16*i + 14] + d_omega_shear1_sf[16*i + 11]),
        (real4)( d_omega_shear1_sf[16*i + 3] + d_omega_shear1_sf[16*i + 12] , 
        d_omega_shear1_sf[16*i + 7] + d_omega_shear1_sf[16*i + 13] ,
        d_omega_shear1_sf[16*i + 11] + d_omega_shear1_sf[16*i + 14] ,
        d_omega_shear1_sf[16*i + 15] *2.0f)
        } ;
        

        real pdotu = dot(umu, pmu1);
        real4 omega_tep = (real4) (0.0f, 0.0f, 0.0f, 0.0f);
        omega_tep.s0 = dot(pmu2,dudx_sym[0]) - pdotu*d_omega_shear2_sf[4*i + 0];
        omega_tep.s1 = dot(pmu2,dudx_sym[1]) - pdotu*d_omega_shear2_sf[4*i + 1];
        omega_tep.s2 = dot(pmu2,dudx_sym[2]) - pdotu*d_omega_shear2_sf[4*i + 2];
        omega_tep.s3 = dot(pmu2,dudx_sym[3]) - pdotu*d_omega_shear2_sf[4*i + 3];


        //printf("%.12f %.12f %.12f %.12f \n",omega_tep.s0,omega_tep.s1,omega_tep.s2,omega_tep.s3);


        // omega_tep.s0 = - pdotu*d_omega_shear2_sf[4*i + 0];
        // omega_tep.s1 = - pdotu*d_omega_shear2_sf[4*i + 1];
        // omega_tep.s2 = - pdotu*d_omega_shear2_sf[4*i + 2];
        // omega_tep.s3 = - pdotu*d_omega_shear2_sf[4*i + 3];

        
        omega_tx = pmu1.s2*umu1.s3 - pmu1.s3*umu1.s2;
        omega_ty = pmu1.s3*umu1.s1 - pmu1.s1*umu1.s3;
        omega_tz = pmu1.s1*umu1.s2 - pmu1.s2*umu1.s1;
        omega_xy = pmu1.s0*umu1.s3 - pmu1.s3*umu1.s0;
        omega_xz = pmu1.s2*umu1.s0 - pmu1.s0*umu1.s2;
        omega_yz = pmu1.s0*umu1.s1 - pmu1.s1*umu1.s0;



        


        
        real beta = 1.0/Tfrz;
        
        real tmp = exp(-pdotu*beta);
        real feq = tmp/(1.0f + tmp);
        //real tmp = (pdotu - mu )*beta;
        //real feq = 1.0/(exp(tmp) + 1);
        real volum = dot(dsigma, pmu1);

        // pomega^a = Omega^{a b} p_b
        real pomega_0 =  omega_tx * omega_tep.s1 + omega_ty * omega_tep.s2 + omega_tz * omega_tep.s3;
        real pomega_1 = -omega_tx * omega_tep.s0 + omega_xy * omega_tep.s2 + omega_xz * omega_tep.s3;
        real pomega_2 = -omega_ty * omega_tep.s0 - omega_xy * omega_tep.s1 + omega_yz * omega_tep.s3;
        real pomega_3 = -omega_tz * omega_tep.s0 - omega_xz * omega_tep.s1 - omega_yz * omega_tep.s2;

        real4 pomega = (real4)(pomega_0, pomega_1, pomega_2, pomega_3);
        // pomega^a

        
            subspec[lid] = volum * feq;
            subpol[lid] = -hbarc*volum * pomega * feq * (1.0f - feq)/(pdotu*Tfrz);
            real pol0 = subpol[lid].s0;
            real pol3 = subpol[lid].s3;
            subpol[lid].s0 = cosh(etas)*pol0 + sinh(etas)*pol3;
            subpol[lid].s3 = sinh(etas)*pol0 + cosh(etas)*pol3;
            

            // t-z coordinate  

        } else {
            subspec[lid] = 0.0f;
            subpol[lid] = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // parallel reduction 
        for ( unsigned int s = get_local_size(0) >> 1; s > 0; s >>= 1 ) {
            if ( lid < s ) {
                subspec[lid] += subspec[lid + s];
                subpol[lid] += subpol[lid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        if ( lid == 0 ) {
             density += subspec[0];
             pol_lab += subpol[0];
        }
    }
    
    // save the final results to global memory
    if ( lid == 0 ) {
        d_density[gid] = density;
        d_pol_lab[gid] = pol_lab/(4.0f*mass);
        //printf ("%.12f %.12f \n",density,pol_lab.s3/(4.0f*mass));
        
        
        

        // polarization at (mt, Y, px, py) in lab frame
        //real4 Pi_lab = pol_lab/(4.0f*mass*density);
        real4 Pi_lab = pol_lab/(4.0f*mass*density);

        // calc the polarization in lrf
        real p0 = mt * cosh(rapidity);
        real pz = mt * sinh(rapidity);
        real4 p_mu = (real4)(p0, -px, -py, -pz);

        real Pi0 = dot(p_mu, Pi_lab)/mass;

        real3 Pi123 = Pi_lab.s123 - dot(p_mu.s123, Pi_lab.s123)*p_mu.s123/(p0*(p0+mass));

        d_pol_lrf[gid] = (real4)(Pi0, Pi123);


    }
}



__kernel void polarization_accT_on_sf(
        __global real4 * d_pol_lab,
        __global real * d_density,
        __global real4 * d_pol_lrf,
        __global const real8 * d_sf,
        __global const real * d_omega_sf,
        __global const real4 * d_momentum_list)
{
    __local real4 subpol[BSZ];
    __local real  subspec[BSZ];

    int lid = get_local_id(0);
    int gid = get_group_id(0);
    
    real hbarc = 0.19733f;
    real4 mom = d_momentum_list[gid];
    real mt = mom.s0;
    real rapidity = mom.s1;
    real px = mom.s2;
    real py = mom.s3;
    real Tfrz = TFRZ;
    real mass = MASS;
    real mu = Mu;

    real8 sf;
    real omega_tx, omega_ty, omega_tz, omega_xy, omega_xz, omega_yz;


    //
    real4 pol_lab = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
    real density = 0.0f;

    // notice there if size_sf is not multiplies of BSZ, some threads can never
    // hit the barrier(CLK_LOCAL_MEM_FENCE), the program will abort
    int nsf = round_up(SIZE_SF, BSZ);


    for ( int i = get_local_id(0); i < nsf; i += get_local_size(0) ) {
        
        if ( i < SIZE_SF ) {
        sf = d_sf[i];
        
        real4 umu = (real4)(1.0f, sf.s4, sf.s5, sf.s6) * \
            1.0f/sqrt(max((real)1.0E-10f, \
            (real)(1.0f-sf.s4*sf.s4 - sf.s5*sf.s5 - sf.s6*sf.s6)));
        // u^{mu}

        real4 umu1 = (real4)(1.0f, -sf.s4, -sf.s5, -sf.s6) * \
            1.0f/sqrt(max((real)1.0E-10f, \
            (real)(1.0f-sf.s4*sf.s4 - sf.s5*sf.s5 - sf.s6*sf.s6)));
        // u_{mu}
        real4 dsigma = sf.s0123;
        real etas = sf.s7;

        real Y = rapidity;
        real mtcosh = mt * cosh(Y - etas);
        real mtsinh = mt * sinh(Y - etas);
        real4 pmu1 = (real4)(mtcosh, -px, -py, -mtsinh); //p_mu
        real4 pmu2 = (real4)(mtcosh, px, py, mtsinh); //p^mu


        
        omega_tx = d_omega_sf[6*i + 0];
        omega_ty = d_omega_sf[6*i + 1];
        omega_tz = d_omega_sf[6*i + 2];
        omega_xy = d_omega_sf[6*i + 3];
        omega_xz = d_omega_sf[6*i + 4];
        omega_yz = d_omega_sf[6*i + 5];



        


        
        real beta = 1.0/Tfrz;
        real pdotu = dot(umu, pmu1);
        real tmp = exp(-pdotu*beta);
        real feq = tmp/(1.0f + tmp);
        //real tmp = (pdotu - mu )*beta;
        //real feq = 1.0/(exp(tmp) + 1);
        real volum = dot(dsigma, pmu1);

        // pomega^a = Omega^{a b} p_b

        real pomega_0 =  omega_tx * pmu1.s1 + omega_ty * pmu1.s2 + omega_tz * pmu1.s3;
        real pomega_1 = -omega_tx * pmu1.s0 + omega_xy * pmu1.s2 + omega_xz * pmu1.s3;
        real pomega_2 = -omega_ty * pmu1.s0 - omega_xy * pmu1.s1 + omega_yz * pmu1.s3;
        real pomega_3 = -omega_tz * pmu1.s0 - omega_xz * pmu1.s1 - omega_yz * pmu1.s2;

    

        real4 pomega = (real4)(pomega_0, pomega_1, pomega_2, pomega_3);
        // pomega^a

        
            subspec[lid] = volum * feq;
            subpol[lid] = -hbarc*volum * pomega * feq * (1.0f - feq)/(Tfrz);
            real pol0 = subpol[lid].s0;
            real pol3 = subpol[lid].s3;
            subpol[lid].s0 = cosh(etas)*pol0 + sinh(etas)*pol3;
            subpol[lid].s3 = sinh(etas)*pol0 + cosh(etas)*pol3;
            

            // t-z coordinate  

        } else {
            subspec[lid] = 0.0f;
            subpol[lid] = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // parallel reduction 
        for ( unsigned int s = get_local_size(0) >> 1; s > 0; s >>= 1 ) {
            if ( lid < s ) {
                subspec[lid] += subspec[lid + s];
                subpol[lid] += subpol[lid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        if ( lid == 0 ) {
             density += subspec[0];
             pol_lab += subpol[0];
        }
    }
    
    // save the final results to global memory
    if ( lid == 0 ) {
        d_density[gid] = density;
        d_pol_lab[gid] = pol_lab/(4.0f*mass);
        //printf ("%.12f %.12f \n",density,pol_lab.s3/(4.0f*mass));
        
        
        

        // polarization at (mt, Y, px, py) in lab frame
        //real4 Pi_lab = pol_lab/(4.0f*mass*density);
        real4 Pi_lab = pol_lab/(4.0f*mass*density);

        // calc the polarization in lrf
        real p0 = mt * cosh(rapidity);
        real pz = mt * sinh(rapidity);
        real4 p_mu = (real4)(p0, -px, -py, -pz);

        real Pi0 = dot(p_mu, Pi_lab)/mass;

        real3 Pi123 = Pi_lab.s123 - dot(p_mu.s123, Pi_lab.s123)*p_mu.s123/(p0*(p0+mass));

        d_pol_lrf[gid] = (real4)(Pi0, Pi123);


    }
}
