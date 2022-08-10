#include "lbm.hpp"
using namespace Kokkos;

void LBM::setup_f()
{

    f_in[0] = buffer_pack_f("f_in0", q, 1, ly, lz);

    f_in[1] = buffer_pack_f("f_in1", q, 1, ly, lz);

    f_out[0] = buffer_pack_f("f_out0", q, 1, ly, lz);

    f_out[1] = buffer_pack_f("f_out1", q, 1, ly, lz);
}

void LBM::pack(buffer_f ff)
{

    // 6 faces
    Kokkos::deep_copy(f_out[0], Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost, ghost+1), Kokkos::ALL, Kokkos::ALL));

    Kokkos::deep_copy(f_out[1], Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx -  ghost-1, lx - ghost), Kokkos::ALL, Kokkos::ALL));

}
void LBM::exchange()
{

    MPI_Isend(f_out[0].data(), f_out[0].size(), MPI_DOUBLE, comm.face_send[0], 0, comm.comm, &mpi_requests_send[0]);

    MPI_Irecv(f_in[0].data(), f_in[0].size(), MPI_DOUBLE, comm.face_recv[0], 0, comm.comm, &mpi_requests_recv[0]);

    MPI_Isend(f_out[1].data(), f_out[1].size(), MPI_DOUBLE, comm.face_send[1], 1, comm.comm, &mpi_requests_send[1]);

    MPI_Irecv(f_in[1].data(), f_in[1].size(), MPI_DOUBLE, comm.face_recv[1], 1, comm.comm, &mpi_requests_recv[1]);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Waitall(2, mpi_requests_send, MPI_STATUS_IGNORE);
    MPI_Waitall(2, mpi_requests_recv, MPI_STATUS_IGNORE);
}

void LBM::unpack(buffer_f ff)
{

    Kokkos::deep_copy(Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost-1, ghost), Kokkos::ALL, Kokkos::ALL), f_in[1]);

    Kokkos::deep_copy(Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx-ghost, lx-ghost+1), Kokkos::ALL, Kokkos::ALL), f_in[0]);
};




/////////////////////////////////////////////////////////////////
void LBM::setup_u()
{

    u_in[0] = buffer_pack_u("u_in0",  ghost, ly, lz);

    u_in[1] = buffer_pack_u("u_in1",  ghost, ly, lz);

    u_out[0] = buffer_pack_u("u_out0",  ghost, ly, lz);

    u_out[1] = buffer_pack_u("u_out1",  ghost, ly, lz);

}

void LBM::pack_u(buffer_u u)
{
    // 6 faces
    Kokkos::deep_copy(u_out[0], Kokkos::subview(u,  std::make_pair(ghost, 2 * ghost), Kokkos::ALL, Kokkos::ALL));

    Kokkos::deep_copy(u_out[1], Kokkos::subview(u,  std::make_pair(lx - 2 * ghost, lx - ghost), Kokkos::ALL, Kokkos::ALL));

}
void LBM::exchange_u()
{

    MPI_Isend(u_out[0].data(), u_out[0].size(), MPI_DOUBLE, comm.face_send[0], 0, comm.comm, &mpi_requests_send[0]);

    MPI_Irecv(u_in[0].data(), u_in[0].size(), MPI_DOUBLE, comm.face_recv[0], 0, comm.comm, &mpi_requests_recv[0]);

    MPI_Isend(u_out[1].data(), u_out[1].size(), MPI_DOUBLE, comm.face_send[1], 1, comm.comm, &mpi_requests_send[1]);

    MPI_Irecv(u_in[1].data(), u_in[1].size(), MPI_DOUBLE, comm.face_recv[1], 1, comm.comm, &mpi_requests_recv[1]);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Waitall(2, mpi_requests_send, MPI_STATUS_IGNORE);
    MPI_Waitall(2, mpi_requests_recv, MPI_STATUS_IGNORE);
}

void LBM::unpack_u(buffer_u u)
{

    Kokkos::deep_copy(Kokkos::subview(u, std::make_pair(0, ghost), Kokkos::ALL, Kokkos::ALL), u_in[1]);

    Kokkos::deep_copy(Kokkos::subview(u, std::make_pair(lx - ghost, lx), Kokkos::ALL, Kokkos::ALL), u_in[0]);
};

void LBM::pass(buffer_u u){

    pack_u(u);
    exchange_u();
    unpack_u(u);
    MPI_Barrier(MPI_COMM_WORLD);

};

void LBM::passf(buffer_f ff){

    pack(ff);
    exchange();
    unpack(ff);
    MPI_Barrier(MPI_COMM_WORLD);

};
