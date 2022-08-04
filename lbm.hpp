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
struct CommHelper
{

    MPI_Comm comm;
    int rx, ry;
    int nranks;
    int me;
    int px, py;
    int up, down, left, right, leftup, leftdown, rightup, rightdown;

    CommHelper(MPI_Comm comm_)
    {
        comm = comm_;

        MPI_Comm_size(comm, &nranks);
        MPI_Comm_rank(comm, &me);

        ry = std::pow(1.0 * nranks, 1.0 / 2.0);
        while (nranks % ry != 0)
            ry++;

        rx = nranks / ry;

        px = me % rx;
        py = (me / rx) % ry;

        printf("rx=%d,ry=%d,px=%d,py=%d\n", rx, ry, px, py);

        left = px == 0 ? me+rx-1 : me - 1;
        right = px == rx - 1 ? me-rx+1 : me + 1;
        down = py == 0 ? px+(ry-1)*rx : me - rx;
        up = py == ry - 1 ? px : me + rx;
                
                
        leftup = (py == ry - 1) ? left%rx : left + rx;
        
        rightup = (py == ry - 1) ? right%rx : right + rx;

        leftdown = (py == 0) ? left%rx+(ry-1)*rx : left - rx;
        rightdown = ( py == 0) ? right%rx+(ry-1)*rx : right - rx;

        printf("Me:%i MyNeibors: %i %i %i %i %i %i %i %i\n", me, left, right, up, down, leftup, leftdown, rightup, rightdown);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    template <class ViewType>
    void isend_irecv(int partner, ViewType send_buffer, ViewType recv_buffer, MPI_Request *request_send, MPI_Request *request_recv)
    {
        MPI_Irecv(recv_buffer.data(), recv_buffer.size(), MPI_DOUBLE, partner, 1, comm, request_recv);
        MPI_Isend(send_buffer.data(), send_buffer.size(), MPI_DOUBLE, partner, 1, comm, request_send);
    }
};

struct LBM
{
    typedef RangePolicy<> range_policy;
    typedef MDRangePolicy<Rank<3>> mdrange_policy3;
    typedef MDRangePolicy<Rank<2>> mdrange_policy2;
    using buffer_ft = View<double ***, CudaSpace>;
    using buffer_tr = View<double ***, CudaSpace>;
    using buffer_t = View<double **, LayoutLeft, HostSpace>;
    using buffer_ut = View<double *, LayoutLeft, HostSpace>;

    CommHelper comm;
    
    MPI_Request mpi_requests_recv[8];
    MPI_Request mpi_requests_send[8];
    int mpi_active_requests;

    int glx;
    int gly;

    int lx = glx / comm.rx + 2 * ghost;
    int ly = gly / comm.ry + 2 * ghost;
    //Compute c1(int lx,int ly);
    int x_lo, x_hi, y_lo, y_hi;
    double u0 = 0.01;
    double mu;
    double cs2=1.0/3.0;
    double r0;
    double beta, kappa;
    double sigma;
    double rho_l, rho_v;
    double tau0, tau1, taum;
    double rho0, rho1;

    MPI_Datatype m_face1, m_face2, m_face3, m_line1, m_line2, m_line3, m_point;
    MPI_Datatype u_face1, u_face2, u_line1;
    View<double ***, CudaSpace> f,f_tem,fb,g,g_tem,gb;

    View<double **, CudaUVMSpace> ua, va, rho, cp, p, pp, phi, tau, nu;

    View<double ***, CudaUVMSpace> drho, dphi, dp, dpp, du, dv;
    View<double **, CudaUVMSpace> divphix, divphiy;

    View<double **, CudaUVMSpace> div;

    View<double ***, CudaUVMSpace> edc_cp, edc_rho, edm_cp, edm_rho;

    View<int **, CudaUVMSpace> e;
    View<double *, CudaUVMSpace> t;
    View<int **, CudaUVMSpace> usr;
    View<int **, CudaUVMSpace> ran;
    View<int *, CudaUVMSpace> bb;


    LBM(MPI_Comm comm_, int sx, int sy, double &tau0, double &tau1, double &taum, double &rho0,double &rho1, double &sigma) : 
          comm(comm_), glx(sx), gly(sy), tau0(tau0), tau1(tau1), taum(taum), rho_l(rho0),rho_v(rho1), sigma(sigma){
        
        MPI_Type_vector(ghost, q * lx, 0, MPI_DOUBLE, &m_face1);
        MPI_Type_commit(&m_face1);

        MPI_Type_vector((ly), q*ghost, lx*q, MPI_DOUBLE, &m_face2);
        MPI_Type_commit(&m_face2);

        MPI_Type_vector(ghost, ghost, lx*q, MPI_DOUBLE, &m_line1);
        MPI_Type_commit(&m_line1);

        MPI_Type_vector(ghost, lx, 0, MPI_DOUBLE, &u_face1);
        MPI_Type_commit(&u_face1);

        MPI_Type_vector(ly, ghost, lx, MPI_DOUBLE, &u_face2);
        MPI_Type_commit(&u_face2);

        MPI_Type_vector(ghost, ghost, ly, MPI_DOUBLE, &u_line1);
        MPI_Type_commit(&u_line1);
                                                                 };

    void Initialize();
    void Collision();
    void u_exchange(View<double **,CudaUVMSpace> c);
    void exchange(buffer_ft ff);
    void Streaming();
    void Update();
    void Output(int n);
    void MPIoutput(int n);


    View<double***,CudaUVMSpace> d_c(View<double**,CudaUVMSpace> c);
    View<double***,CudaUVMSpace> d_b(View<double**,CudaUVMSpace> c);
    View<double***,CudaUVMSpace> d_m(View<double**,CudaUVMSpace> c);
    View<double**,CudaUVMSpace> laplace(View<double**,CudaUVMSpace> c);
    View<double***,CudaUVMSpace> edc(View<double**,CudaUVMSpace> c);
    View<double***,CudaUVMSpace> edb(View<double**,CudaUVMSpace> c);
    View<double***,CudaUVMSpace> edm(View<double**,CudaUVMSpace> c);
    View<double **, CudaUVMSpace> div_c(View<double **, CudaUVMSpace> cx, View<double **, CudaUVMSpace> cy);
};
#endif