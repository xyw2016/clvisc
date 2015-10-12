#include<helper.h>

/** First stage: calc sub spec for each particle */
__kernel void get_sub_dNdYPtdPtdPhi(  
            __global real8  * d_SF,            
            __global real  * d_SubSpec,            
            __constant real4 * d_HadronInfo,           
            __constant real * d_Y ,           
            __constant real * d_Pt,
            __constant real * d_CPhi,
            __constant real * d_SPhi,
            const int pid,
            const int id_Y_PT_PHI)
{
    int I = get_global_id(0);
    
    int tid = get_local_id(0) ;
    
    __local real subspec[ BSZ ];
    
    real4 HadronInfo = d_HadronInfo[ pid ];
    real mass = HadronInfo.s0;
    real gspin = HadronInfo.s1;
    real fermi_boson = HadronInfo.s2;
    real muB = HadronInfo.s3;
    real dof = gspin / pown(2.0*M_PI_F, 3);
    
    int k = id_Y_PT_PHI / NPT;
    int l = id_Y_PT_PHI - k*NPT;
    real rapidity = d_Y[k];
    real pt = d_Pt[l];
    real mt = sqrt(mass*mass + pt*pt); 
    
    //real dNdYPtdPtdPhi[NPHI] = {0.0f}; 
    real dNdYPtdPtdPhi[NPHI];
    for ( int m = 0; m < NPHI; m++ ) {
        dNdYPtdPtdPhi[m] = 0.0f;
    }
    
    while ( I < SizeSF ) {
        real8 SF = d_SF[I];
        
        real4 umu = (real4)(1.0f, SF.s4, SF.s5, SF.s6) * \
            1.0/sqrt(max((real)1.0E-15, \
            (real)(1.0-SF.s4*SF.s4 - SF.s5*SF.s5 - SF.s6*SF.s6)));
        
        real4 dsigma = SF.s0123;
        
        real mtcosh = mt*cosh(rapidity-SF.s7);
        real mtsinh = mt*sinh(rapidity-SF.s7);
        for(int m=0; m<NPHI; m++){
            real4 pmu = (real4)(mtcosh, -pt*d_CPhi[m], -pt*d_SPhi[m], -mtsinh);
        
            dNdYPtdPtdPhi[m] += dof * dot(pmu, dsigma) / \
                ( exp((dot(pmu, umu)-muB)/Tfrz) + fermi_boson );
        }
        
        /** in units of GeV.fm^3 */
        I += get_global_size(0);
    }
    
    for ( int m = 0; m < NPHI; m++ ) {
        subspec[tid] = dNdYPtdPtdPhi[m];
        barrier(CLK_LOCAL_MEM_FENCE);
    
        //// do reduction in shared mem
        for ( unsigned int s = get_local_size(0) >> 1; s > 0; s >>= 1 ) {
            if (tid < s) {
                subspec[tid] += subspec[tid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    
        if(tid == 0) d_SubSpec[k*NPT*NPHI*NBlocks + l*NPHI*NBlocks \
               + m*NBlocks + get_group_id(0) ] = subspec[0];
    }
}


/** The second stage on sum reduction 
*   For each Y,Pt,Phi we need to sum NBlocks dNdYptdptdphi from the first stage */
__kernel void get_dNdYPtdPtdPhi(  
                                  __global real  * d_SubSpec,            
                                  __global real  * d_Spec,
                                  const int pid)
{
    int I = get_global_id(0);
    int tid = get_local_id(0);
    int localSize = get_local_size(0);
    
    __local real spec[ NBlocks ];
    
    spec[ tid ] = d_SubSpec[ I ];
    
    barrier( CLK_LOCAL_MEM_FENCE );
    
    for( int s = localSize>>1; s>0; s >>= 1 ){
         if( tid < s ){
             spec[ tid ] += spec[ s + tid ];
         }
         barrier( CLK_LOCAL_MEM_FENCE );
    }

    /** unroll the last warp because they are synced automatically */
    /** \bug unroll gives out wrong results */
    //if( NBlocks >= 512 ){ if( tid < 256 ) spec[tid] += spec[ 256 + tid ]; barrier( CLK_LOCAL_MEM_FENCE ); };
    //if( NBlocks >= 256 ){ if( tid < 128 ) spec[tid] += spec[ 128 + tid ]; barrier( CLK_LOCAL_MEM_FENCE ); };
    //if( NBlocks >= 128 ){ if( tid < 64 )  spec[tid] += spec[ 64 + tid ]; barrier( CLK_LOCAL_MEM_FENCE ); };

    //if( tid < 32 ){
    //   if( NBlocks >= 64 ) spec[ tid ] += spec[ 32 + tid ];
    //   if( NBlocks >= 32 ) spec[ tid ] += spec[ 16 + tid ];
    //   if( NBlocks >= 16 ) spec[ tid ] += spec[ 8 + tid ];
    //   if( NBlocks >= 8 )  spec[ tid ] += spec[ 4 + tid ];
    //   if( NBlocks >= 4 )  spec[ tid ] += spec[ 2 + tid ];
    //   if( NBlocks >= 2 )  spec[ tid ] += spec[ 1 + tid ];
    //}

    if( tid == 0 ) d_Spec[ pid*NY*NPT*NPHI + get_group_id(0) ] = spec[ tid ];
}