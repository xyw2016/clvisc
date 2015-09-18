#include<helper.h>

#define one_o_2sigr2  1.0/(2.0*SIGR*SIGR)
#define one_o_2sigz2  1.0/(2.0*SIGZ*SIGZ)

#define w1  one_o_2sigr2/M_PI_F
#define w2  sqrt(one_o_2sigz2/M_PI_F)/TAU0

#define NPARTONS_PER_GROUP 512

__kernel void smearing(
    __global real8 * d_p4x4, \
    __global real4  * d_EdV, \
    const int npartons, \
    const int size )
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  if ( (i*NY*NZ+j*NZ+k) > size ) {
     return;
  }

  real4 grid = (real4)(TAU0, (i-NX/2)*DX, (j-NY/2)*DY, (k-NZ/2)*DZ);

  __local real8 local_p4x4[NPARTONS_PER_GROUP];

  real delta, etasi; 

  real4 position;
  real4 momentum;

  real4 Tm0 = (real4)(0.0f, 0.0f, 0.0f, 0.0f);

  int li = get_local_id(0);
  int lj = get_local_id(1);
  int lk = get_local_id(2);

  // local id in the workgroup
  int tid = li*5*5 + lj*5 + lk;


  int npartons_in_this_group;
  const int num_of_groups = npartons / NPARTONS_PER_GROUP;

  const int block_size = get_local_size(0) * get_local_size(1) * \
                         get_local_size(2);

  for ( int n = 0; n < npartons; n = n + NPARTONS_PER_GROUP ) {
    // each thread load more than 1 particle per loop
    for ( int m = tid; m < NPARTONS_PER_GROUP; m += block_size ) {
        if ( (n+m) < npartons ) {
          local_p4x4[m] = d_p4x4[n+m];
        }
    }

    // The last workgroup has different number of partons
    if ( n < num_of_groups * NPARTONS_PER_GROUP ) {
        npartons_in_this_group = NPARTONS_PER_GROUP;
    } else {
        npartons_in_this_group = npartons - num_of_groups * NPARTONS_PER_GROUP;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int m=0; m < npartons_in_this_group; m++) {
      momentum = local_p4x4[m].s0123;
      position = local_p4x4[m].s4567;

      etasi = 0.5f*log(max(position.s0+position.s3, acu)) \
        - 0.5f*log(max(position.s0-position.s3, acu));

      position.s3 = etasi;

      real4 d = grid - position;

      real distance_sqr = one_o_2sigr2*(d.s1*d.s1+d.s2*d.s2+d.s3*d.s3);

      // only do smearing for distance < 10*sigma
      if ( distance_sqr < 50 ) {
          delta = w1*w2*exp(- distance_sqr);
          Tm0 += delta * momentum;
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  /** KFACTOR=1.6 for LHC energy and 1.45 for RHIC energy */
  Tm0 = KFACTOR * Tm0;

  real mt = sqrt(max(Tm0.s0*Tm0.s0-Tm0.s3*Tm0.s3, \
                 Tm0.s1*Tm0.s1+Tm0.s2*Tm0.s2));

  real Y = 0.5f*(log(max(Tm0.s0+Tm0.s3, acu)) \
           - log(max(Tm0.s0-Tm0.s3, acu)));

  Tm0 = (real4)(mt*cosh(Y-grid.s3), Tm0.s1, Tm0.s2, mt*sinh(Y-grid.s3));

  real K2 = Tm0.s1*Tm0.s1 + Tm0.s2*Tm0.s2 + Tm0.s3*Tm0.s3;

  real K0 = Tm0.s0;
  if ( K0 < acu ) K0 = acu;

  real Ed = 0.0f;
  rootFinding( &Ed, K0, K2 );

  if ( Ed < 1.0E-15f ) Ed = 1.0E-15f;

  real EPV = max(acu, K0+P(Ed));

  d_EdV[get_global_id(0)*NY*NZ + get_global_id(1)*NZ + get_global_id(2)] = \
       (real4){Ed, Tm0.s1/EPV, Tm0.s2/EPV, Tm0.s3/EPV/TAU0};
  
}
