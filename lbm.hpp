#ifndef _LBM_H_
#define _LBM_H_

#include <cmath>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Timer.hpp>

#define q 27
#define dim 3
#define ghost 3
#define eps 1e-6
#define pi 3.1415926

typedef Kokkos::TeamPolicy<> team_policy;
typedef Kokkos::TeamPolicy<>::member_type member_type;
typedef Kokkos::RangePolicy<> range_policy;
typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy2;
typedef Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy3;
typedef Kokkos::MDRangePolicy<Kokkos::Rank<4>> mdrange_policy4;

using buffer_f = Kokkos::View<double ****,  Kokkos::CudaSpace>;
using buffer_u = Kokkos::View<double ***, Kokkos::CudaSpace>;
using buffer_div = Kokkos::View<double ***, Kokkos::CudaSpace>;
using buffer_pack_f = Kokkos::View<double ****, Kokkos::CudaHostPinnedSpace>;
using buffer_pack_u = Kokkos::View<double ***, Kokkos::CudaHostPinnedSpace>;



struct CommHelper
{

    MPI_Comm comm;
    // rank number for each dim
    int rx,ry,rz;
    // rank
    int me;
    // axis for each rank
    int px,py,pz;
    int nranks;
    // 6 faces
    int face_recv[2],face_send[2];

    CommHelper(MPI_Comm comm_)
    {
        comm = comm_;
        
        MPI_Comm_size(comm, &nranks);
        MPI_Comm_rank(comm, &me);

        rx = nranks;
        ry = 1;
        rz = 1;

        px = me % rx;
        py = 0;
        pz = 0;


        face_send[0] = px == 0 ? me + rx - 1 : me - 1;

        face_send[1] = px == rx - 1 ? me - rx + 1 : me + 1;

        face_recv[0] = face_send[1];
 
        face_recv[1] = face_send[0];

    
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

struct LBM
{

    CommHelper comm;
    MPI_Request mpi_requests_recv[2];
    MPI_Request mpi_requests_send[2];

    MPI_Datatype f_face_recv[2];
    MPI_Datatype f_face_send[2];

    MPI_Datatype u_face_recv[2];
    MPI_Datatype u_face_send[2];

    int glx, gly, glz;
    // include ghost nodes
    int lx, ly, lz;
    // local start, local end, local length
    int l_s[3], l_e[3], l_l[3];

    // local axis
    int x_lo = 0, x_hi = 0, y_lo = 0, y_hi = 0, z_lo = 0, z_hi = 0;
    double rho0, rho1, mu;
    double sigma;
    double taum, tau0, tau1, r_x, r_y, r_z;
    double r1, u0;
    double cs2 = 1.0 / 3.0;
    double delta = 4.0;

    buffer_pack_f f_in[2], f_out[2];
    buffer_pack_f f_in0,f_in1,f_out0,f_out1;
    buffer_pack_u u_in[2], u_out[2];
    buffer_f f, ft, fb, g, gt, gb;
    buffer_f dphi, drho, dp, dpp, du, dv, dw;

    buffer_u phi, ua, va, wa, rho, p, pp;
    buffer_div divphix, divphiy, divphiz, div, tau, nu;

    Kokkos::View<int ***, Kokkos::CudaUVMSpace> usr, ran;

    Kokkos::View<int *, Kokkos::CudaUVMSpace> bb;

    Kokkos::View<double *, Kokkos::CudaUVMSpace> t;

    Kokkos::View<int **, Kokkos::CudaUVMSpace> e;

    LBM(MPI_Comm comm_, int sx, int sy, int sz, double &tau0, double &tau1, double &taum, double &rho0, double &rho1, double &r_x, double &r_y, double &rz,double &u0)
        : comm(comm_), glx(sx), gly(sy), glz(sz), tau0(tau0), tau1(tau1), taum(taum), rho0(rho0),rho1(rho1), r_x(r_x), r_y(r_y), r_z(r_z),u0(u0)
    {
        // local length
        l_l[0] = (comm.px - glx % comm.rx >= 0) ? glx / comm.rx : glx / comm.rx + 1;
        l_l[1] = gly;
        l_l[2] = glz;
        // local length
        lx = l_l[0] + 2 * ghost;
        ly = l_l[1] + 2 * ghost;
        lz = l_l[2] + 2 * ghost;
        // local start
        l_s[0] = ghost;
        l_s[1] = ghost;
        l_s[2] = ghost;
        // local end
        l_e[0] = l_s[0] + l_l[0];
        l_e[1] = l_s[1] + l_l[1];
        l_e[2] = l_s[2] + l_l[2];

        int x_his[comm.nranks];
        int y_his[comm.nranks];
        int z_his[comm.nranks];
        int ax_his[comm.rx][comm.ry][comm.rz];
        int ay_his[comm.rx][comm.ry][comm.rz];
        int az_his[comm.rx][comm.ry][comm.rz];

        MPI_Allgather(l_l, 1, MPI_INT, x_his, 1, MPI_INT, comm.comm);
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allgather(l_l + 1, 1, MPI_INT, y_his, 1, MPI_INT, comm.comm);
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allgather(l_l + 2, 1, MPI_INT, z_his, 1, MPI_INT, comm.comm);
        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0; i < comm.rx; i++)
        {
            for (int j = 0; j < comm.ry; j++)
            {
                for (int k = 0; k < comm.rz; k++)
                {
                    ax_his[i][j][k] = x_his[i + j * (comm.rx) + k * (comm.rx) * (comm.ry)];
                    ay_his[i][j][k] = y_his[i + j * (comm.rx) + k * (comm.rx) * (comm.ry)];
                    az_his[i][j][k] = z_his[i + j * (comm.rx) + k * (comm.rx) * (comm.ry)];
                }
            }
        }

        for (int i = 0; i <= comm.px; i++)
        {
            x_hi += ax_his[i][0][0];
        }

        for (int j = 0; j <= comm.py; j++)
        {
            y_hi += ay_his[0][j][0];
        }

        for (int k = 0; k <= comm.pz; k++)
        {
            z_hi += az_his[0][0][k];
        }

        x_lo = x_hi - l_l[0];
        x_hi = x_hi - 1;

        y_lo = y_hi - l_l[1];
        y_hi = y_hi - 1;

        z_lo = z_hi - l_l[2];
        z_hi = z_hi - 1;

        MPI_Barrier(MPI_COMM_WORLD);

    };

    void Initialize();
    void Collision();
    void Streaming();
    void Boundary();
    void Update();
    void MPIoutput(int n);
    void Output(int n);

    void setup_u();
    void pack_u(buffer_u c);
    void unpack_u(buffer_u c);

    void setup_f();
    void pack(buffer_f ff);
    void unpack(buffer_f ff);

    void exchange_u();
    void exchange();

    void passf(buffer_f ff);
    void pass(buffer_u c);

    buffer_f d_c(buffer_u c);
    buffer_div div_c(buffer_div cx,buffer_div cy,buffer_div cz);
};
#endif