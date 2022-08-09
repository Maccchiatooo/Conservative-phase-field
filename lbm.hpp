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
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Timer.hpp>
#include <mpi.h>

#define q 27
#define dim 3
#define ghost 3

typedef Kokkos::RangePolicy<> range_policy;
typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy2;
typedef Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy3;
typedef Kokkos::MDRangePolicy<Kokkos::Rank<4>> mdrange_policy4;

using buffer_f = Kokkos::View<double ****,  Kokkos::CudaUVMSpace>;
using buffer_div = Kokkos::View<double ***, Kokkos::CudaUVMSpace>;
using buffer_u = Kokkos::View<double ***, Kokkos::CudaUVMSpace>;


struct CommHelper
{

    MPI_Comm comm;
    // rank number for each dim
    int rx, ry, rz;
    // rank
    int me;
    // axis for each rank
    int px, py, pz;
    int nranks;
    // 6 faces
    int face_recv[6],face_send[6];
    // 12 edges
    int edge_recv[12],edge_send[12];
    // 8 points
    int point_recv[8],point_send[8];


    CommHelper(MPI_Comm comm_)
    {
        comm = comm_;
        
        MPI_Comm_size(comm, &nranks);
        MPI_Comm_rank(comm, &me);

        rx = std::pow(1.0 * nranks, 1.0 / 3.0);
        while (nranks % rx != 0)
            rx++;

        rz = std::sqrt(1.0 * (nranks / rx));
        while ((nranks / rx) % rz != 0)
            rz++;

        ry = nranks / rx / rz;

        px = me % rx;
        pz = (me / rx) % rz;
        py = (me / rx / rz);

        // 6 faces
        //left
        face_send[0] = px == 0 ? me + rx - 1 : me - 1;
        //right
        face_send[1] = px == rx - 1 ? me - rx + 1 : me + 1;
        //down
        face_send[2] = pz == 0 ? me + rx * (rz - 1) : me - rx;
        //top
        face_send[3] = pz == rz - 1 ? me - rx * (rz - 1) : me + rx;
        //front
        face_send[4] = py == 0 ? me + rx * rz * (ry - 1) : me - rx * rz;
        //back
        face_send[5] = py == ry - 1 ? me - rx * rz * (ry - 1) : me + rx * rz;

        // 12 edges
        //left up
        edge_send[0] = (pz == rz - 1) ? face_send[0] - rx * (rz - 1) : face_send[0] + rx;
        //rightup
        edge_send[1] = (pz == rz - 1) ? face_send[1] - rx * (rz - 1) : face_send[1] + rx;
        //leftdown
        edge_send[2] = (pz == 0) ? face_send[0] + rx * (rz - 1) : face_send[0] - rx;
        //rightdown
        edge_send[3] = (pz == 0) ? face_send[1] + rx * (rz - 1) : face_send[1] - rx;
        //frontup
        edge_send[4] = (pz == rz - 1) ? face_send[4] - rx * (rz - 1) : face_send[4] + rx;
        //frontdown
        edge_send[5] = (pz == 0) ? face_send[4] + rx * (rz - 1) : face_send[4] - rx;
        //backup
        edge_send[6] = (pz == rz - 1) ? face_send[5] - rx * (rz - 1) : face_send[5] + rx;
        //backdown
        edge_send[7] = (pz == 0) ? face_send[5] + rx * (rz - 1) : face_send[5] - rx;
        //frontright
        edge_send[8] = (px == rx - 1) ? face_send[4] - rx + 1 : face_send[4] + 1;
        //frontleft
        edge_send[9] = (px == 0) ? face_send[4] + rx - 1 : face_send[4] - 1;
        //backright
        edge_send[10] = (px == rx - 1) ? face_send[5] - rx + 1 : face_send[5] + 1;
        //backleft
        edge_send[11] = (px == 0) ? face_send[5] + rx - 1 : face_send[5] - 1;

        // 8 points
        //backrightdown
        point_send[0] = (px == rx - 1) ? edge_send[7] - rx + 1 : edge_send[7] + 1;
        //frontrightdown
        point_send[1] = (px == rx - 1) ? edge_send[5] - rx + 1 : edge_send[5] + 1;
        //frontrightup
        point_send[2] = (px == rx - 1) ? edge_send[4] - rx + 1 : edge_send[4] + 1;
        //backrightup
        point_send[3] = (px == rx - 1) ? edge_send[6] - rx + 1 : edge_send[6] + 1;
        //frontleftup
        point_send[4] = (px == 0) ? edge_send[4] + rx - 1 : edge_send[4] - 1;
        //backleftdown
        point_send[5] = (px == 0) ? edge_send[7] + rx - 1 : edge_send[7] - 1;
        //frontleftdown
        point_send[6] = (px == 0) ? edge_send[5] + rx - 1 : edge_send[5] - 1;
        //backleftup
        point_send[7] = (px == 0) ? edge_send[6] + rx - 1 : edge_send[6] - 1;

        //recv
        //left
        face_recv[0] = face_send[1];
        //right
        face_recv[1] = face_send[0];
        //down
        face_recv[2] = face_send[3];
        //top
        face_recv[3] = face_send[2];
        //front
        face_recv[4] = face_send[5];
        //back
        face_recv[5] = face_send[4];

        // 12 edges
        //left up
        edge_recv[0] = edge_send[3];
        //rightup
        edge_recv[1] = edge_send[2];
        //leftdown
        edge_recv[2] = edge_send[1];
        //rightdown
        edge_recv[3] = edge_send[0];
        //frontup
        edge_recv[4] = edge_send[7];
        //frontdown
        edge_recv[5] = edge_send[6];
        //backup
        edge_recv[6] = edge_send[5];
        //backdown
        edge_recv[7] = edge_send[4];
        //frontright
        edge_recv[8] = edge_send[11];
        //frontleft
        edge_recv[9] = edge_send[10];
        //backright
        edge_recv[10] = edge_send[9];
        //backleft
        edge_recv[11] = edge_send[8];

        // 8 points
        //backrightdown
        point_recv[0] = point_send[4];
        //frontrightdown
        point_recv[1] = point_send[7];
        //frontrightup
        point_recv[2] = point_send[5];
        //backrightup
        point_recv[3] = point_send[6];
        //frontleftup
        point_recv[4] = point_send[0];
        //backleftdown
        point_recv[5] = point_send[2];
        //frontleftdown
        point_recv[6] = point_send[3];
        //backleftup
        point_recv[7] = point_send[1];
        

        MPI_Barrier(MPI_COMM_WORLD);
    }
};

struct LBM
{

    CommHelper comm;
    MPI_Request mpi_requests_recv[26];
    MPI_Request mpi_requests_send[26];
    int mpi_active_requests;

    MPI_Datatype f_face_recv[6], f_edge_recv[12], f_point_recv[8];
    MPI_Datatype f_face_send[6], f_edge_send[12], f_point_send[8];

    MPI_Datatype u_face_recv[6], u_edge_recv[12], u_point_recv[8];
    MPI_Datatype u_face_send[6], u_edge_send[12], u_point_send[8];

    MPI_Datatype m_face1, m_face2, m_face3;

    int glx, gly, glz;
    // include ghost nodes
    int lx, ly, lz;
    // local start, local end, local length
    int l_s[3], l_e[3], l_l[3];

    // local axis
    int x_lo = 0, x_hi = 0, y_lo = 0, y_hi = 0, z_lo = 0, z_hi = 0;
    double rho0,rho1, mu;
    double sigma;
    double taum,tau0,tau1, r_x, r_y, r_z;
    double r1,u0;
    double cs2 = 1.0 / 3.0;
    double delta = 4.0;
    double eps = 0.000001;

    // 6 faces


    buffer_f f, ft, fb, g, gt, gb;
    buffer_f dphi,drho,dp,dpp,du,dv,dw;

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
        l_l[1] = (comm.py - gly % comm.ry >= 0) ? gly / comm.ry : gly / comm.ry + 1;
        l_l[2] = (comm.pz - glz % comm.rz >= 0) ? glz / comm.rz : glz / comm.rz + 1;
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

        MPI_Type_vector(lz*ly, q * ghost, lx*q, MPI_DOUBLE, &m_face1);
        MPI_Type_commit(&m_face1);

        MPI_Type_vector(1, q*ghost*lx*ly, 1, MPI_DOUBLE, &m_face2);
        MPI_Type_commit(&m_face2);

        MPI_Type_vector(lz, q*ghost*lx, q*lx*ly, MPI_DOUBLE, &m_face3);
        MPI_Type_commit(&m_face3);

        int f_array[4] = {q,lx, ly, lz};
        int f_face_len[3][4] = {{q, ghost, ly, lz}, {q, lx, ly, ghost}, {q, lx, ghost, lz}};
        int f_edge_len[3][4] = {{q, ghost, ly, ghost}, {q, lx, ghost, ghost}, {q, ghost, ghost, lz}};
        int f_point_len[4] = {q, ghost, ghost, ghost};

        int f_facestart_send[6][4] = {{0, ghost, 0, 0}, {0, lx - 2 * ghost, 0, 0}, {0, 0, 0, ghost}, {0, 0, 0, lz - 2 * ghost}, {0, 0, ghost, 0}, {0, 0, ly - 2 * ghost, 0}};
        int f_facestart_recv[6][4] = {{0, lx - ghost, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, lz - ghost}, {0, 0, 0, 0}, {0, 0, ly - ghost, 0}, {0, 0, 0, 0}};

        int f_edgestart_send[12][4] = {{0, ghost, 0, lz - 2 * ghost}, {0, lx - 2 * ghost, 0, lz - 2 * ghost}, {0, ghost, 0, ghost}, {0, lx - 2 * ghost, 0, ghost}, 
                                        {0, 0, ghost, lz - 2 * ghost}, {0, 0, ghost, ghost}, {0, 0, ly - 2 * ghost, lz - 2 * ghost}, {0, 0, ly - 2 * ghost, ghost}, 
                                        {0, lx - 2 * ghost, ghost, 0}, {0, ghost, ghost, 0}, {0, lx - 2 * ghost, ly - 2 * ghost, 0}, {0, ghost, ly - 2 * ghost, 0}};

        int f_edgestart_recv[12][4] = {{0, lx - ghost, 0, 0}, {0, 0, 0, 0}, {0, lx - ghost, 0, lz - ghost}, {0, 0, 0, lz - ghost}, 
                                        {0, 0, ly - ghost, 0}, {0, 0, ly - ghost, lz - ghost}, {0, 0, 0, 0}, {0, 0, 0, lz - ghost}, 
                                        {0, 0, ly - ghost, 0}, {0, lx - ghost, ly - ghost, 0}, {0, 0, 0, 0}, {0, lx - ghost, 0, 0}};

        int f_pointstart_send[8][4] = {{0, lx - 2 * ghost, ly - 2 * ghost, ghost}, {0, lx - 2 * ghost, ghost, ghost}, {0, lx - 2 * ghost, ghost, lz - 2 * ghost}, {0, lx - 2 * ghost, ly - 2 * ghost, lz - 2 * ghost}, 
                                     {0, ghost, ghost, lz - 2 * ghost}, {0, ghost, ly - 2 * ghost, ghost}, {0, ghost, ghost, ghost}, {0, ghost, ly - 2 * ghost, lz - 2 * ghost}};
        int f_pointstart_recv[8][4] = {{0, 0, 0, lz - ghost}, {0, 0, ly - ghost, lz - ghost}, {0, 0, ly - ghost, 0}, {0, 0, 0, 0}, 
                                    {0, lx - ghost, ly - ghost, 0}, {0, lx - ghost, 0, lz - ghost}, {0, lx - ghost, ly - ghost, lz - ghost}, {0, lx - ghost, 0, 0}};

        int u_array[3] = {lx, ly, lz};
        int u_face_len[3][3] = {{ ghost, ly, lz}, { lx, ly, ghost}, { lx, ghost, lz}};
        int u_edge_len[3][3] = {{ ghost, ly, ghost}, { lx, ghost, ghost}, { ghost, ghost, lz}};
        int u_point_len[3] = { ghost, ghost, ghost};

        int u_facestart_send[6][3] = {{ghost, 0, 0}, {lx - 2 * ghost, 0, 0}, {0, 0, ghost}, {0, 0, lz - 2 * ghost}, {0, ghost, 0}, {0, ly - 2 * ghost, 0}};
        int u_facestart_recv[6][3] = {{lx - ghost, 0, 0}, {0, 0, 0}, {0, 0, lz - ghost}, {0, 0, 0}, {0, ly - ghost, 0}, {0, 0, 0}};

        int u_edgestart_send[12][3] = {{ghost, 0, lz - 2 * ghost}, {lx - 2 * ghost, 0, lz - 2 * ghost}, {ghost, 0, ghost}, {lx - 2 * ghost, 0, ghost}, {0, ghost, lz - 2 * ghost}, {0, ghost, ghost}, {0, ly - 2 * ghost, lz - 2 * ghost}, {0, ly - 2 * ghost, ghost}, {lx - 2 * ghost, ghost, 0}, {ghost, ghost, 0}, {lx - 2 * ghost, ly - 2 * ghost, 0}, {ghost, ly - 2 * ghost, 0}};
        int u_edgestart_recv[12][3] = {{lx - ghost, 0, 0}, {0, 0, 0}, { lx - ghost, 0, lz - ghost}, {0, 0, lz - ghost}, {0,ly - ghost, 0}, {0, ly - ghost, lz - ghost}, {0, 0, 0}, {0, 0, lz - ghost}, {0,ly - ghost, 0}, {lx - ghost, ly - ghost, 0}, {0, 0, 0}, {lx - ghost, 0, 0}};

        int u_pointstart_send[8][3] = {{lx - 2 * ghost, ly - 2 * ghost, ghost}, {lx - 2 * ghost, ghost, ghost}, {lx - 2 * ghost, ghost, lz - 2 * ghost}, {lx - 2 * ghost, ly - 2 * ghost, lz - 2 * ghost}, {ghost, ghost, lz - 2 * ghost}, {ghost, ly - 2 * ghost, ghost}, {ghost, ghost, ghost}, {ghost, ly - 2 * ghost, lz - 2 * ghost}};
        int u_pointstart_recv[8][3] = {{0, 0, lz - ghost}, {0, ly - ghost, lz - ghost}, {0, ly - ghost, 0}, {0, 0, 0}, {lx - ghost, ly - ghost, 0}, {lx - ghost, 0, lz - ghost}, {lx - ghost, ly - ghost, lz - ghost}, {lx - ghost, 0, 0}};

        for (int i = 0; i < 6;i++){
            int (*p1)[4];
            int (*p2)[4];
            int (*p3)[4];
            p1= f_face_len;
            p2= f_facestart_send;
            p3= f_facestart_recv;
            MPI_Type_create_subarray(4, f_array, *(p1 + i / 2), *(p2 + i), MPI_ORDER_FORTRAN, MPI_DOUBLE, &f_face_send[i]);
            MPI_Type_commit(&f_face_send[i]);
            MPI_Type_create_subarray(4, f_array, *(p1 + i / 2), *(p3 + i), MPI_ORDER_FORTRAN, MPI_DOUBLE, &f_face_recv[i]);
            MPI_Type_commit(&f_face_recv[i]);
        }



        for (int i = 0; i < 12;i++){
            int(*p1)[4];
            int(*p2)[4];
            int (*p3)[4];
            p1= f_edge_len;
            p2= f_edgestart_send;
            p3= f_edgestart_recv;
            MPI_Type_create_subarray(4, f_array, *(p1 + i / 4), *(p2 + i), MPI_ORDER_FORTRAN, MPI_DOUBLE, &f_edge_send[i]);
            MPI_Type_commit(&f_edge_send[i]);
            MPI_Type_create_subarray(4, f_array, *(p1 + i / 4), *(p3 + i), MPI_ORDER_FORTRAN, MPI_DOUBLE, &f_edge_recv[i]);
            MPI_Type_commit(&f_edge_recv[i]);
        }

        for (int i = 0; i < 8;i++){
            int(*p2)[4];
            int (*p3)[4];
            p2= f_pointstart_send;
            p3= f_pointstart_recv;
            MPI_Type_create_subarray(4, f_array, f_point_len, *(p2 + i), MPI_ORDER_FORTRAN, MPI_DOUBLE, &f_point_send[i]);
            MPI_Type_commit(&f_point_send[i]);
            MPI_Type_create_subarray(4, f_array, f_point_len, *(p3 + i), MPI_ORDER_FORTRAN, MPI_DOUBLE, &f_point_recv[i]);
            MPI_Type_commit(&f_point_recv[i]);
        }

        for (int i = 0; i < 6;i++){
            int(*p1)[3];
            int(*p2)[3];
            int(*p3)[3];
            p1 = u_face_len;
            p2 = u_facestart_send;
            p3 = u_facestart_recv;
            MPI_Type_create_subarray(3, u_array, *(p1 + i / 2), *(p2 + i), MPI_ORDER_FORTRAN, MPI_DOUBLE, &u_face_send[i]);
            MPI_Type_commit(&u_face_send[i]);
            MPI_Type_create_subarray(3, u_array, *(p1 + i / 2), *(p3 + i), MPI_ORDER_FORTRAN, MPI_DOUBLE, &u_face_recv[i]);
            MPI_Type_commit(&u_face_recv[i]);
               }

        for (int i = 0; i < 12;i++){
            int(*p1)[3];
            int(*p2)[3];
            int (*p3)[3];
            p1= u_edge_len;
            p2= u_edgestart_send;
            p3= u_edgestart_recv;
            MPI_Type_create_subarray(3, u_array, *(p1 + i / 4), *(p2 + i), MPI_ORDER_FORTRAN, MPI_DOUBLE, &u_edge_send[i]);
            MPI_Type_commit(&u_edge_send[i]);
            MPI_Type_create_subarray(3, u_array, *(p1 + i / 4), *(p3 + i), MPI_ORDER_FORTRAN, MPI_DOUBLE, &u_edge_recv[i]);
            MPI_Type_commit(&u_edge_recv[i]);
        }

        for (int i = 0; i < 8;i++){
            int(*p2)[3];
            int (*p3)[3];
            p2= u_pointstart_send;
            p3= u_pointstart_recv;
            MPI_Type_create_subarray(3, u_array, u_point_len, *(p2 + i), MPI_ORDER_FORTRAN, MPI_DOUBLE, &u_point_send[i]);
            MPI_Type_commit(&u_point_send[i]);
            MPI_Type_create_subarray(3, u_array, u_point_len, *(p3 + i), MPI_ORDER_FORTRAN, MPI_DOUBLE, &u_point_recv[i]);
            MPI_Type_commit(&u_point_recv[i]);
        }

    };

    void Initialize();
    void Collision();
    void exchange(buffer_f ff);

    void Streaming();
    void Boundary();
    void Update();
    void MPIoutput(int n);
    void Output(int n);

    void exchange_u(buffer_u u);



    buffer_f d_c(buffer_u c);
        
    buffer_div div_c(buffer_div cx,buffer_div cy,buffer_div cz);
};
#endif