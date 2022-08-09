#include "lbm.hpp"

using namespace Kokkos;
void LBM::exchange(buffer_f ff)
{
    /*for (int i = 0; i < 6;i++){
        MPI_Isend(ff.data(), 1, f_face_send[i], comm.face_send[i], i, comm.comm, &mpi_requests_send[i]);
        MPI_Irecv(ff.data(), 1, f_face_recv[i], comm.face_recv[i], i, comm.comm, &mpi_requests_recv[i]);
    }*/
    int mar = 1;
    MPI_Isend(subview(ff, ALL, std::make_pair(ghost, 2 * ghost), ALL, ALL).data(), 1,
              m_face1, comm.face_send[0], mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(ff, ALL, std::make_pair(lx - ghost, lx), ALL, ALL).data(), 1,
              m_face1, comm.face_recv[0], mar, comm.comm, &mpi_requests_recv[mar - 1]);

    /*mar = 2;
    MPI_Isend(subview(ff, ALL, std::make_pair(lx - 2 * ghost, lx - ghost), ALL, ALL).data(), 1, 
    m_face1, comm.face_send[mar-1], mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(ff, ALL, std::make_pair(0, ghost), ALL, ALL).data(), 1,
    m_face1, comm.face_recv[mar-1], mar, comm.comm, &mpi_requests_recv[mar - 1]);

    /*mar = 3;
    MPI_Isend(subview(ff, ALL, ALL, ALL, std::make_pair(ghost, 2 * ghost)).data(),1,
    m_face2, comm.face_send[mar-1], mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(ff, ALL, ALL, ALL, std::make_pair(lz-ghost, lz)).data(),1,
    m_face2, comm.face_recv[mar-1], mar, comm.comm, &mpi_requests_recv[mar - 1]);

    mar = 4;
    MPI_Isend(subview(ff, ALL, ALL, ALL, std::make_pair(lz - 2 * ghost, lz - ghost)).data(),1,
    m_face2, comm.face_send[mar-1], mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(ff, ALL, ALL, ALL, std::make_pair(0, ghost)).data(),1,
    m_face2, comm.face_recv[mar-1], mar, comm.comm, &mpi_requests_recv[mar - 1]);

    mar = 5;
    MPI_Isend(subview(ff, ALL, ALL, std::make_pair(ghost, 2 * ghost), ALL).data(),1,
    m_face3, comm.face_send[mar-1], mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(ff, ALL, ALL, std::make_pair(ly-ghost, ly), ALL).data(),1,
    m_face3, comm.face_recv[mar-1], mar, comm.comm, &mpi_requests_recv[mar - 1]);

    mar = 6;
    MPI_Isend(subview(ff, ALL, ALL, std::make_pair(ly - 2 * ghost, ly - ghost), ALL).data(),1,
    m_face3, comm.face_send[mar-1], mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(ff, ALL, ALL, std::make_pair(0, ghost), ALL).data(),1,
    m_face3, comm.face_recv[mar-1], mar, comm.comm, &mpi_requests_recv[mar - 1]);
*/


    for (int i = 0; i < 1; i++)
    {
        MPI_Wait(&mpi_requests_send[i], MPI_STATUS_IGNORE);
        MPI_Wait(&mpi_requests_recv[i], MPI_STATUS_IGNORE);
    }
    /*MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < 12;i++){
        MPI_Isend(ff.data(), 1, f_edge_send[i], comm.edge_send[i], i , comm.comm, &mpi_requests_send[i ]);
        MPI_Irecv(ff.data(), 1, f_edge_recv[i], comm.edge_recv[i], i , comm.comm, &mpi_requests_recv[i ]);
    }
    for (int i = 0; i < 12; i++)
    {
        MPI_Wait(&mpi_requests_send[i], MPI_STATUS_IGNORE);
        MPI_Wait(&mpi_requests_recv[i], MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < 8;i++){
        MPI_Isend(ff.data(), 1, f_point_send[i], comm.point_send[i], i, comm.comm, &mpi_requests_send[i ]);
        MPI_Irecv(ff.data(), 1, f_point_recv[i], comm.point_recv[i], i, comm.comm, &mpi_requests_recv[i ]);
    }
        for (int i = 0; i < 8; i++)
    {
        MPI_Wait(&mpi_requests_send[i], MPI_STATUS_IGNORE);
        MPI_Wait(&mpi_requests_recv[i], MPI_STATUS_IGNORE);
    }
*/
    MPI_Barrier(MPI_COMM_WORLD);

}


void LBM::exchange_u(buffer_u u)
{

    for (int i = 0; i < 6;i++){
        MPI_Isend(u.data(), 1, u_face_send[i], comm.face_send[i], i, comm.comm, &mpi_requests_send[i]);
        MPI_Irecv(u.data(), 1, u_face_recv[i], comm.face_recv[i], i, comm.comm, &mpi_requests_recv[i]);
    }

    for (int i = 0; i < 6; i++)
    {
        MPI_Wait(&mpi_requests_send[i], MPI_STATUS_IGNORE);
        MPI_Wait(&mpi_requests_recv[i], MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < 12;i++){
        MPI_Isend(u.data(), 1, u_edge_send[i], comm.edge_send[i], i , comm.comm, &mpi_requests_send[i ]);
        MPI_Irecv(u.data(), 1, u_edge_recv[i], comm.edge_recv[i], i , comm.comm, &mpi_requests_recv[i ]);
    }
    for (int i = 0; i < 12; i++)
    {
        MPI_Wait(&mpi_requests_send[i], MPI_STATUS_IGNORE);
        MPI_Wait(&mpi_requests_recv[i], MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < 8;i++){
        MPI_Isend(u.data(), 1, u_point_send[i], comm.point_send[i], i, comm.comm, &mpi_requests_send[i ]);
        MPI_Irecv(u.data(), 1, u_point_recv[i], comm.point_recv[i], i, comm.comm, &mpi_requests_recv[i ]);
    }
        for (int i = 0; i < 8; i++)
    {
        MPI_Wait(&mpi_requests_send[i], MPI_STATUS_IGNORE);
        MPI_Wait(&mpi_requests_recv[i], MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
};
