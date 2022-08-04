#include "lbm.hpp"
using namespace std;
using namespace Kokkos;
void LBM::exchange(View<double ***, CudaSpace> ff)
{
    int mar = 1;
    MPI_Isend(subview(ff, ALL, std::make_pair(ghost, 2 * ghost), ALL).data(), 1, 
    m_face2, comm.left, mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(ff, ALL, std::make_pair(lx - ghost, lx), ALL).data(), 1, 
    m_face2, comm.right, mar, comm.comm, &mpi_requests_recv[mar - 1]);

    mar = 2;
    MPI_Isend(subview(ff, ALL, std::make_pair(lx - 2*ghost, lx-ghost), ALL).data(), 1, 
    m_face2, comm.right, mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(ff, ALL, std::make_pair(0, ghost), ALL).data(), 1,
    m_face2, comm.left, mar, comm.comm, &mpi_requests_recv[mar - 1]);

    mar = 3;
    MPI_Isend(subview(ff, ALL, ALL, std::make_pair(ghost, 2 * ghost)).data(),1,
    m_face1, comm.down, mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(ff, ALL, ALL, std::make_pair(ly - ghost, ly)).data(),1,
    m_face1, comm.up, mar, comm.comm, &mpi_requests_recv[mar - 1]);

    mar = 4;
    MPI_Isend(subview(ff, ALL, ALL, std::make_pair(ly - 2 * ghost, ly - ghost)).data(),1,
    m_face1, comm.up, mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(ff, ALL, ALL,  std::make_pair(0, ghost)).data(),1,
    m_face1, comm.down, mar, comm.comm, &mpi_requests_recv[mar - 1]);

    mar = 5;
    MPI_Isend(subview(ff, ALL, std::make_pair(ghost, 2 * ghost), std::make_pair(ghost, 2 * ghost)).data(), 1,
    m_line1, comm.leftdown, mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(ff, ALL, std::make_pair(lx - ghost, lx),  std::make_pair(ly - ghost, ly)).data(),1,
    m_line1, comm.rightup, mar, comm.comm, &mpi_requests_recv[mar - 1]);

    mar = 6;
    MPI_Isend(subview(ff, ALL, std::make_pair(lx - 2 * ghost, lx - ghost), std::make_pair(ghost, 2 * ghost)).data(),1,
    m_line1, comm.rightdown, mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(ff, ALL, std::make_pair(0, ghost), std::make_pair(ly - ghost, ly)).data(),1,
    m_line1, comm.leftup, mar, comm.comm, &mpi_requests_recv[mar - 1]);

    mar = 7;
    MPI_Isend(subview(ff, ALL, std::make_pair(lx - 2 * ghost, lx - ghost),std::make_pair(ly-2*ghost,ly-ghost)).data(), 1,
    m_line1, comm.rightup, mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(ff, ALL, std::make_pair(0, ghost),  std::make_pair(0, ghost)).data(),1,
    m_line1, comm.leftdown, mar, comm.comm, &mpi_requests_recv[mar - 1]);

    mar = 8;
    MPI_Isend(subview(ff, ALL, std::make_pair(ghost, 2 * ghost), std::make_pair(ly - 2 * ghost, ly - ghost)).data(),1,
    m_line1, comm.leftup, mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(ff, ALL, std::make_pair(lx - ghost, lx),  std::make_pair(0, ghost)).data(),1,
    m_line1, comm.rightdown, mar, comm.comm, &mpi_requests_recv[mar - 1]);

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < q - 1; i++)
    {
        MPI_Wait(&mpi_requests_send[i], MPI_STATUS_IGNORE);
        MPI_Wait(&mpi_requests_recv[i], MPI_STATUS_IGNORE);
    }
};



void LBM::u_exchange(View<double **,CudaUVMSpace> u)
{
    int mar = 1;
    MPI_Isend(subview(u, std::make_pair(ghost, 2 * ghost), ALL).data(), 1, 
    u_face2, comm.left, mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(u, std::make_pair(lx - ghost, lx), ALL).data(), 1, 
    u_face2, comm.right, mar, comm.comm, &mpi_requests_recv[mar - 1]);

    mar = 2;
    MPI_Isend(subview(u, std::make_pair(lx - 2*ghost, lx-ghost), ALL).data(), 1, 
    u_face2, comm.right, mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(u, std::make_pair(0, ghost), ALL).data(), 1,
    u_face2, comm.left, mar, comm.comm, &mpi_requests_recv[mar - 1]);

    mar = 3;
    MPI_Isend(subview(u, ALL, std::make_pair(ghost, 2 * ghost)).data(),1,
    u_face1, comm.down, mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(u, ALL, std::make_pair(ly - ghost, ly)).data(),1,
    u_face1, comm.up, mar, comm.comm, &mpi_requests_recv[mar - 1]);

    mar = 4;
    MPI_Isend(subview(u, ALL, std::make_pair(ly - 2 * ghost, ly - ghost)).data(),1,
    u_face1, comm.up, mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(u, ALL,  std::make_pair(0, ghost)).data(),1,
    u_face1, comm.down, mar, comm.comm, &mpi_requests_recv[mar - 1]);

    mar = 5;
    MPI_Isend(subview(u, std::make_pair(ghost, 2 * ghost), std::make_pair(ghost, 2 * ghost)).data(), 1,
    u_line1, comm.leftdown, mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(u, std::make_pair(lx - ghost, lx),  std::make_pair(ly - ghost, ly)).data(),1,
    u_line1, comm.rightup, mar, comm.comm, &mpi_requests_recv[mar - 1]);

    mar = 6;
    MPI_Isend(subview(u, std::make_pair(lx - 2 * ghost, lx - ghost), std::make_pair(ghost, 2 * ghost)).data(),1,
    u_line1, comm.rightdown, mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(u, std::make_pair(0, ghost), std::make_pair(ly - ghost, ly)).data(),1,
    u_line1, comm.leftup, mar, comm.comm, &mpi_requests_recv[mar - 1]);

    mar = 7;
    MPI_Isend(subview(u, std::make_pair(lx - 2 * ghost, lx - ghost), std::make_pair(ly - 2 * ghost, ly - ghost)).data(),1,
    u_line1, comm.rightup, mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(u, std::make_pair(0, ghost),  std::make_pair(0, ghost)).data(),1,
    u_line1, comm.leftdown, mar, comm.comm, &mpi_requests_recv[mar - 1]);

    mar = 8;
    MPI_Isend(subview(u, std::make_pair(ghost, 2 * ghost), std::make_pair(ly - 2 * ghost, ly - ghost)).data(),1,
    u_line1, comm.leftup, mar, comm.comm, &mpi_requests_send[mar - 1]);

    MPI_Irecv(subview(u, std::make_pair(lx - ghost, lx),  std::make_pair(0, ghost)).data(),1,
    u_line1, comm.rightdown, mar, comm.comm, &mpi_requests_recv[mar - 1]);


    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < q - 1; i++)
    {
        MPI_Wait(&mpi_requests_send[i], MPI_STATUS_IGNORE);
        MPI_Wait(&mpi_requests_recv[i], MPI_STATUS_IGNORE);
        }
};




