#ifndef _LBM_H_
#define _LBM_H_

#include <cmath>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Timer.hpp>
#include <mpi.h>
#include <cstring>
#include <stdexcept>
#include <iostream>

#define q 9
#define dim 2
#define ghost 3
#define eps 0.000001
#define delta 4.0
using namespace std;
using namespace Kokkos;

typedef TeamPolicy<Cuda> team_policy;
typedef TeamPolicy<Cuda>::member_type member_type;
typedef RangePolicy<> range_policy;
typedef MDRangePolicy<Rank<2>> mdrange_policy2;
typedef MDRangePolicy<Rank<3>> mdrange_policy3;
typedef MDRangePolicy<Rank<4>> mdrange_policy4;

using buffer_f = View<double ***,  CudaSpace>;
using buffer_u = View<double **, CudaSpace>;
using buffer_div = View<double **, CudaSpace>;
using buffer_pack_f = View<double ***, CudaHostPinnedSpace>;
using buffer_pack_u = View<double **, CudaHostPinnedSpace>;

struct LBM
{


    MPI_Comm comm;
    int rx, ry;
    int nranks;
    int me;
    int px, py;
    int l_s[2], l_e[2], l_l[2];

    int glx;
    int gly;

    int lx ;
    int ly ;
    //Compute c1(int lx,int ly);
    int x_lo=0, x_hi=0, y_lo=0, y_hi=0;
    double u0 = 0.01;
    double mu;
    double cs2=1.0/3.0;
    double r0;
    double beta, kappa;
    double sigma;
    double rho_l, rho_v;
    double tau0, tau1, taum;
    double rho0, rho1;


    int face_recv[4],face_send[4];
    // 12 edges
    int edge_recv[4],edge_send[4];


    buffer_pack_f m_left, m_right, m_down, m_up;
    buffer_pack_f m_leftout, m_rightout, m_downout, m_upout;
    buffer_pack_f m_leftup, m_rightup, m_leftdown, m_rightdown;
    buffer_pack_f m_leftupout, m_rightupout, m_leftdownout, m_rightdownout;

    buffer_pack_u u_left, u_right, u_down, u_up;
    buffer_pack_u u_leftout, u_rightout, u_downout, u_upout;
    buffer_pack_u u_leftup, u_rightup, u_leftdown, u_rightdown;
    buffer_pack_u u_leftupout, u_rightupout, u_leftdownout, u_rightdownout;

    buffer_f f,f_tem,fb,g,g_tem,gb;

    buffer_u ua, va, rho, cp, p, pp, phi, tau, nu;

    buffer_f drho, dphi, dp, dpp, du, dv;
    buffer_div div, divphix, divphiy;

    View<int **, CudaSpace> e;
    View<double *, CudaSpace> t;
    View<int **, CudaSpace> usr;
    View<int **, CudaSpace> ran;
    View<int *, CudaSpace> bb;


    LBM(MPI_Comm comm_, int sx, int sy, double &tau0, double &tau1, double &taum, double &rho0,double &rho1, double &sigma) : 
          comm(comm_), glx(sx), gly(sy), tau0(tau0), tau1(tau1), taum(taum), rho_l(rho0),rho_v(rho1), sigma(sigma){
        
                                                                 };


    void setup_Cartesian();
    void setup_Local();
    void setup_MPI();
    void setup_u();
    void setup_f();

    void Initialize();
    void Collision();
    void Streaming();
    void Update();
    void Output(int n);
    void MPIoutput(int n);

    void pack_f(buffer_f ff);
    void unpack_f(buffer_f ff);

    void pack_u(buffer_u c);
    void unpack_u(buffer_u c);

    void exchange_u();
    void exchange_f();

    void passf(buffer_f ff);
    void passu(buffer_u c);

};
#endif