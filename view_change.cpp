#include "lbm.hpp"
using namespace std;
using namespace Kokkos;
void LBM::setup_MPI(){

    MPI_Comm_size(comm, &nranks);
    MPI_Comm_rank(comm, &me);

    setup_Cartesian();
    
    setup_Local();

    setup_u();
    
    setup_f();

    MPI_Barrier(MPI_COMM_WORLD);
}
void LBM::setup_Cartesian(){

    int x_tem, y_tem;

    double min_e = 1e16;
    double u2 = 0;
    for (int i = 1; i < nranks + 1; i++)
    {
        if (nranks % i == 0)
        {
            x_tem = i;

            y_tem = nranks/i;

            double loc_x = glx / x_tem;
            double loc_y = gly / y_tem;

            double average = (loc_x + loc_y) / 2.0;

            u2 = pow((loc_x - average), 2) + pow((loc_y - average), 2);
            if (u2 < min_e)
            {
                min_e = u2;
                rx = x_tem;
                ry = y_tem;
                    }
                }
            }

    
    printf("rx=%d,ry=%d\n", rx, ry);

    px = me % rx;
    py = (me / rx) % ry;

    // 6 faces
    //left
    face_send[0] = px == 0 ? me + rx - 1 : me - 1;
    // right
    face_send[1] = px == rx - 1 ? me - rx + 1 : me + 1;
    // down
    face_send[2] = py == 0 ? px + rx  * (ry - 1) : me - rx ;
    // up
    face_send[3] = py == ry - 1 ? px : me + rx;

    // 12 edges
    // left up
    edge_send[0] = (py == ry - 1) ? face_send[0]%rx  : face_send[0] + rx;
    // rightup
    edge_send[1] = (py == ry - 1) ? face_send[1] %rx: face_send[1] + rx;
    // leftdown
    edge_send[2] = (py == 0) ? face_send[0]%rx + rx * (ry - 1) : face_send[0] - rx;
    // rightdown
    edge_send[3] = (py == 0) ? face_send[1]%rx + rx * (ry - 1) : face_send[1] - rx;

    //printf("rank=%d,leftup=%d,rightup=%d,leftdown=%d,rightdown=%d\n", me, edge_send[0], edge_send[1], edge_send[2], edge_send[3]);

    // recv
    // left
    face_recv[0] = face_send[1];
    // right
    face_recv[1] = face_send[0];
    // down
    face_recv[2] = face_send[3];
    // top
    face_recv[3] = face_send[2];


    // 12 edges
    // left up
    edge_recv[0] = edge_send[3];
    // rightup
    edge_recv[1] = edge_send[2];
    // leftdown
    edge_recv[2] = edge_send[1];
    // rightdown
    edge_recv[3] = edge_send[0];


    MPI_Barrier(MPI_COMM_WORLD);
}

void LBM::setup_Local(){

    l_l[0] = (px - glx % rx >= 0) ? glx / rx : glx / rx + 1;
    l_l[1] = (py - gly % ry >= 0) ? gly / ry : gly / ry + 1;

    // local length
    lx = l_l[0] + 2 * ghost;
    ly = l_l[1] + 2 * ghost;

    // local start
    l_s[0] = ghost;
    l_s[1] = ghost;

    // local end
    l_e[0] = l_s[0] + l_l[0];
    l_e[1] = l_s[1] + l_l[1];

    int x_his[nranks];
    int y_his[nranks];

    int ax_his[rx][ry];
    int ay_his[rx][ry];


    MPI_Allgather(l_l, 1, MPI_INT, x_his, 1, MPI_INT, comm);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Allgather(l_l + 1, 1, MPI_INT, y_his, 1, MPI_INT, comm);
    MPI_Barrier(MPI_COMM_WORLD);


    for (int i = 0; i < rx; i++)
    {
        for (int j = 0; j < ry; j++)
        {

                ax_his[i][j] = x_his[i + j * (rx) ];
                ay_his[i][j] = y_his[i + j * (rx) ];
  
            
        }
        }

        for (int i = 0; i <= px; i++)
        {
            x_hi += ax_his[i][0];
        }

        for (int j = 0; j <= py; j++)
        {
            y_hi += ay_his[0][j];
        }


        x_lo = x_hi - l_l[0];
        x_hi = x_hi - 1;

        y_lo = y_hi - l_l[1];
        y_hi = y_hi - 1;

}
void LBM::setup_f()
{
    
    m_left = buffer_pack_f("m_left", q, 1, ly);

    m_right = buffer_pack_f("m_right", q, 1, ly);

    m_down = buffer_pack_f("m_down", q, lx,  1);

    m_up = buffer_pack_f("m_up", q, lx,  1);

    // 12 lines

    m_leftup = buffer_pack_f("m_leftup", q, 1, 1);

    m_rightup = buffer_pack_f("m_rightup", q, 1,  1);

    m_leftdown = buffer_pack_f("m_leftdown", q, 1, 1);

    m_rightdown = buffer_pack_f("m_rightdown", q, 1, 1);


    // outdirection
    // 6 faces

    m_leftout = buffer_pack_f("m_leftout", q, 1, ly);

    m_rightout = buffer_pack_f("m_rightout", q, 1, ly);

    m_downout = buffer_pack_f("m_downout", q, lx,  1);

    m_upout = buffer_pack_f("m_upout", q, lx,  1);

   

    m_leftupout = buffer_pack_f("m_leftupout", q, 1,  1);

    m_rightupout = buffer_pack_f("m_rightupout", q, 1, 1);

    m_leftdownout = buffer_pack_f("m_leftdownout", q, 1,  1);

    m_rightdownout = buffer_pack_f("m_rightdownout", q, 1, 1);

   
}
void LBM::pack_f(buffer_f ff)
{
    // 6 faces

    Kokkos::deep_copy(m_leftout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_s[0], 1+ l_s[0]), Kokkos::ALL));

    Kokkos::deep_copy(m_rightout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_e[0]-1, l_e[0]), Kokkos::ALL));

    Kokkos::deep_copy(m_downout, Kokkos::subview(ff, Kokkos::ALL, Kokkos::ALL, std::make_pair(l_s[1], 1+ l_s[1])));

    Kokkos::deep_copy(m_upout, Kokkos::subview(ff, Kokkos::ALL, Kokkos::ALL, std::make_pair(l_e[1]-1, l_e[1])));


    Kokkos::deep_copy(m_leftupout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_s[0], 1+ l_s[0]), std::make_pair(l_e[1]-1, l_e[1])));

    Kokkos::deep_copy(m_rightupout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_e[0]-1, l_e[0]), std::make_pair(l_e[1]-1, l_e[1])));

    Kokkos::deep_copy(m_leftdownout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_s[0], 1+ l_s[0]), std::make_pair(l_s[1], l_s[1]+1)));

    Kokkos::deep_copy(m_rightdownout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_e[0]-1, l_e[0]), std::make_pair(l_s[1], l_s[1]+1)));

   

}
void LBM::exchange_f()
{
    MPI_Request mpi_requests_recv[8];
    MPI_Request mpi_requests_send[8];

    int mar = 1;
    MPI_Isend(m_leftout.data(), m_leftout.size(), MPI_DOUBLE, face_send[0], mar, comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(m_right.data(), m_right.size(), MPI_DOUBLE, face_recv[0], mar, comm, &mpi_requests_recv[mar - 1]);

    mar = 2;
    MPI_Isend(m_rightout.data(), m_rightout.size(), MPI_DOUBLE, face_send[1], mar, comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(m_left.data(), m_left.size(), MPI_DOUBLE, face_recv[1], mar, comm, &mpi_requests_recv[mar - 1]);

    mar = 3;
    MPI_Isend(m_downout.data(), m_downout.size(), MPI_DOUBLE, face_send[2], mar, comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(m_up.data(), m_up.size(), MPI_DOUBLE, face_recv[2], mar, comm, &mpi_requests_recv[mar - 1]);

    mar = 4;
    MPI_Isend(m_upout.data(), m_upout.size(), MPI_DOUBLE, face_send[3], mar, comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(m_down.data(), m_down.size(), MPI_DOUBLE, face_recv[3], mar, comm, &mpi_requests_recv[mar - 1]);

    mar = 5;
    MPI_Isend(m_leftupout.data(), m_leftupout.size(), MPI_DOUBLE, edge_send[0], mar, comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(m_rightdown.data(), m_rightdown.size(), MPI_DOUBLE, edge_recv[0], mar, comm, &mpi_requests_recv[mar - 1]);

    mar = 6;
    MPI_Isend(m_rightdownout.data(), m_rightdownout.size(), MPI_DOUBLE, edge_send[3], mar, comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(m_leftup.data(), m_leftup.size(), MPI_DOUBLE, edge_recv[3], mar, comm, &mpi_requests_recv[mar - 1]);

    mar = 7;
    MPI_Isend(m_leftdownout.data(), m_leftdownout.size(), MPI_DOUBLE, edge_send[2], mar, comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(m_rightup.data(), m_rightup.size(), MPI_DOUBLE, edge_recv[2], mar, comm, &mpi_requests_recv[mar - 1]);

    mar = 8;
    MPI_Isend(m_rightupout.data(), m_rightupout.size(), MPI_DOUBLE, edge_send[1], mar, comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(m_leftdown.data(), m_leftdown.size(), MPI_DOUBLE, edge_recv[1], mar, comm, &mpi_requests_recv[mar - 1]);

   

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Waitall(8, mpi_requests_send, MPI_STATUS_IGNORE);
    MPI_Waitall(8, mpi_requests_recv, MPI_STATUS_IGNORE);
}

void LBM::unpack_f(buffer_f ff)
{
    // 6 faces

    Kokkos::deep_copy(Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_s[0]-1, l_s[0]), Kokkos::ALL), m_left);

    Kokkos::deep_copy(Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_e[0], l_e[0]+1), Kokkos::ALL), m_right);

    Kokkos::deep_copy(Kokkos::subview(ff, Kokkos::ALL, Kokkos::ALL, std::make_pair(l_s[1]-1, l_s[1])), m_down);

    Kokkos::deep_copy(Kokkos::subview(ff, Kokkos::ALL, Kokkos::ALL,  std::make_pair(l_e[1], l_e[1]+1)), m_up);

    // 12 lines

    Kokkos::deep_copy(Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_s[0]-1, l_s[0]), std::make_pair(l_e[1], l_e[1]+1)), m_leftup);

    Kokkos::deep_copy(Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_e[0], l_e[0]+1),  std::make_pair(l_e[1], l_e[1]+1)), m_rightup);



    Kokkos::deep_copy(Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_s[0]-1, l_s[0]),  std::make_pair(l_s[1]-1, l_s[1])), m_leftdown);

    Kokkos::deep_copy(Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_e[0], l_e[0]+1),  std::make_pair(l_s[1]-1, l_s[1])), m_rightdown);

   

    Kokkos::fence();
};


void LBM::setup_u()
{
    
    u_left = buffer_pack_u("u_left",  1, ly);

    u_right = buffer_pack_u("u_right",  1, ly);

    u_down = buffer_pack_u("u_down",  lx,  1);

    u_up = buffer_pack_u("u_up",  lx,  1);

    // 12 lines

    u_leftup = buffer_pack_u("u_leftup",  1, 1);

    u_rightup = buffer_pack_u("u_rightup",  1,  1);

    u_leftdown = buffer_pack_u("u_leftdown",  1, 1);

    u_rightdown = buffer_pack_u("u_rightdown",  1, 1);


    // outdirection
    // 6 faces

    u_leftout = buffer_pack_u("u_leftout",  1, ly);

    u_rightout = buffer_pack_u("u_rightout",  1, ly);

    u_downout = buffer_pack_u("u_downout",  lx,  1);

    u_upout = buffer_pack_u("u_upout",  lx,  1);

   

    u_leftupout = buffer_pack_u("u_leftupout",  1,  1);

    u_rightupout = buffer_pack_u("u_rightupout",  1, 1);

    u_leftdownout = buffer_pack_u("u_leftdownout",  1,  1);

    u_rightdownout = buffer_pack_u("u_rightdownout",  1, 1);

   
}
void LBM::pack_u(buffer_u u)
{
    // 6 faces

    Kokkos::deep_copy(u_leftout, Kokkos::subview(u,  std::make_pair(l_s[0], 1+ l_s[0]), Kokkos::ALL));

    Kokkos::deep_copy(u_rightout, Kokkos::subview(u, std::make_pair(l_e[0]-1, l_e[0]), Kokkos::ALL));

    Kokkos::deep_copy(u_downout, Kokkos::subview(u, Kokkos::ALL, std::make_pair(l_s[1], 1+ l_s[1])));

    Kokkos::deep_copy(u_upout, Kokkos::subview(u, Kokkos::ALL, std::make_pair(l_e[1]-1, l_e[1])));


    Kokkos::deep_copy(u_leftupout, Kokkos::subview(u,  std::make_pair(l_s[0], 1+ l_s[0]), std::make_pair(l_e[1]-1, l_e[1])));

    Kokkos::deep_copy(u_rightupout, Kokkos::subview(u,  std::make_pair(l_e[0]-1, l_e[0]), std::make_pair(l_e[1]-1, l_e[1])));

    Kokkos::deep_copy(u_leftdownout, Kokkos::subview(u,  std::make_pair(l_s[0], 1+ l_s[0]), std::make_pair(l_s[1], l_s[1]+1)));

    Kokkos::deep_copy(u_rightdownout, Kokkos::subview(u,  std::make_pair(l_e[0]-1, l_e[0]), std::make_pair(l_s[1], l_s[1]+1)));

   

}
void LBM::exchange_u()
{
    MPI_Request mpi_requests_recv[8];
    MPI_Request mpi_requests_send[8];

    int mar = 1;
    MPI_Isend(u_leftout.data(), u_leftout.size(), MPI_DOUBLE, face_send[0], mar, comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(u_right.data(), u_right.size(), MPI_DOUBLE, face_recv[0], mar, comm, &mpi_requests_recv[mar - 1]);

    mar = 2;
    MPI_Isend(u_rightout.data(), u_rightout.size(), MPI_DOUBLE, face_send[1], mar, comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(u_left.data(), u_left.size(), MPI_DOUBLE, face_recv[1], mar, comm, &mpi_requests_recv[mar - 1]);

    mar = 3;
    MPI_Isend(u_downout.data(), u_downout.size(), MPI_DOUBLE, face_send[2], mar, comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(u_up.data(), u_up.size(), MPI_DOUBLE, face_recv[2], mar, comm, &mpi_requests_recv[mar - 1]);

    mar = 4;
    MPI_Isend(u_upout.data(), u_upout.size(), MPI_DOUBLE, face_send[3], mar, comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(u_down.data(), u_down.size(), MPI_DOUBLE, face_recv[3], mar, comm, &mpi_requests_recv[mar - 1]);

    mar = 5;
    MPI_Isend(u_leftupout.data(), u_leftupout.size(), MPI_DOUBLE, edge_send[0], mar, comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(u_rightdown.data(), u_rightdown.size(), MPI_DOUBLE, edge_recv[0], mar, comm, &mpi_requests_recv[mar - 1]);

    mar = 6;
    MPI_Isend(u_rightdownout.data(), u_rightdownout.size(), MPI_DOUBLE, edge_send[3], mar, comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(u_leftup.data(), u_leftup.size(), MPI_DOUBLE, edge_recv[3], mar, comm, &mpi_requests_recv[mar - 1]);

    mar = 7;
    MPI_Isend(u_leftdownout.data(), u_leftdownout.size(), MPI_DOUBLE, edge_send[2], mar, comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(u_rightup.data(), u_rightup.size(), MPI_DOUBLE, edge_recv[2], mar, comm, &mpi_requests_recv[mar - 1]);

    mar = 8;
    MPI_Isend(u_rightupout.data(), u_rightupout.size(), MPI_DOUBLE, edge_send[1], mar, comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(u_leftdown.data(), u_leftdown.size(), MPI_DOUBLE, edge_recv[1], mar, comm, &mpi_requests_recv[mar - 1]);

   

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Waitall(8, mpi_requests_send, MPI_STATUS_IGNORE);
    MPI_Waitall(8, mpi_requests_recv, MPI_STATUS_IGNORE);
}

void LBM::unpack_u(buffer_u u)
{
    // 6 faces

    Kokkos::deep_copy(Kokkos::subview(u,  std::make_pair(l_s[0]-1, l_s[0]), Kokkos::ALL), u_left);

    Kokkos::deep_copy(Kokkos::subview(u, std::make_pair(l_e[0], l_e[0]+1), Kokkos::ALL), u_right);

    Kokkos::deep_copy(Kokkos::subview(u,  Kokkos::ALL, std::make_pair(l_s[1]-1, l_s[1])), u_down);

    Kokkos::deep_copy(Kokkos::subview(u,  Kokkos::ALL,  std::make_pair(l_e[1], l_e[1]+1)), u_up);

    // 12 lines

    Kokkos::deep_copy(Kokkos::subview(u,  std::make_pair(l_s[0]-1, l_s[0]), std::make_pair(l_e[1], l_e[1]+1)), u_leftup);

    Kokkos::deep_copy(Kokkos::subview(u,  std::make_pair(l_e[0], l_e[0]+1),  std::make_pair(l_e[1], l_e[1]+1)), u_rightup);



    Kokkos::deep_copy(Kokkos::subview(u,  std::make_pair(l_s[0]-1, l_s[0]),  std::make_pair(l_s[1]-1, l_s[1])), u_leftdown);

    Kokkos::deep_copy(Kokkos::subview(u,  std::make_pair(l_e[0], l_e[0]+1),  std::make_pair(l_s[1]-1, l_s[1])), u_rightdown);

   

    Kokkos::fence();
};

void LBM::passu(buffer_u u){

    pack_u(u);
    exchange_u();
    unpack_u(u);


};

void LBM::passf(buffer_f ff){

    pack_f(ff);
    exchange_f();
    unpack_f(ff);
};








