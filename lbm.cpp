#include "lbm.hpp"
using namespace Kokkos;

void LBM::Initialize()
{

    x_lo = (lx - 2 * ghost) * comm.px;
    x_hi = (lx - 2 * ghost) * (comm.px + 1);
    y_lo = (ly - 2 * ghost) * comm.py;
    y_hi = (ly - 2 * ghost) * (comm.py + 1);

    f = View<double ***, CudaSpace>("f", q, lx, ly);
    f_tem = View<double ***, CudaSpace>("ft", q, lx, ly);
    fb = View<double ***, CudaSpace>("fb", q, lx, ly);
    g = View<double ***, CudaSpace>("g", q, lx, ly);
    g_tem = View<double ***, CudaSpace>("gf", q, lx, ly);
    gb = View<double ***, CudaSpace>("gb", q, lx, ly);

    ua = View<double **, CudaUVMSpace>("u", lx, ly);
    va = View<double **, CudaUVMSpace>("v", lx, ly);
    rho = View<double **, CudaUVMSpace>("rho", lx, ly);
    cp = View<double **, CudaUVMSpace>("mu", lx, ly);
    p = View<double **, CudaUVMSpace>("p", lx, ly);
    pp = View<double **, CudaUVMSpace>("pp", lx, ly);
    phi = View<double **, CudaUVMSpace>("phi", lx, ly);
    tau = View<double **, CudaUVMSpace>("tau", lx, ly);
    nu = View<double **, CudaUVMSpace>("nu", lx, ly);

    drho = View<double ***, CudaUVMSpace>("drho", dim, lx, ly);
    dphi = View<double ***, CudaUVMSpace>("dphi", dim, lx, ly);
    dp = View<double ***, CudaUVMSpace>("dp", dim, lx, ly);
    dpp = View<double ***, CudaUVMSpace>("dpp", dim, lx, ly);
    du = View<double ***, CudaUVMSpace>("du", dim, lx, ly);
    dv = View<double ***, CudaUVMSpace>("dv", dim, lx, ly);
    divphix = View<double **, CudaUVMSpace>("divphix", lx, ly);
    divphiy = View<double **, CudaUVMSpace>("divphiy", lx, ly);

    div = View<double **, CudaUVMSpace>("la", lx, ly);

    edc_cp = View<double ***, CudaUVMSpace>("edcmu", q, lx, ly);
    edc_rho = View<double ***, CudaUVMSpace>("edcrho", q, lx, ly);
    edm_cp = View<double ***, CudaUVMSpace>("edmmu", q, lx, ly);
    edm_rho = View<double ***, CudaUVMSpace>("edmrho", q, lx, ly);

    e = View<int **, CudaUVMSpace>("e", q, dim);
    t = View<double *, CudaUVMSpace>("t", q);
    usr = View<int **, CudaUVMSpace>("usr", lx, ly);
    ran = View<int **, CudaUVMSpace>("ran", lx, ly);
    bb = View<int *, CudaUVMSpace>("b", q);

    // weight and discrete velocity
    t(0) = 4.0 / 9.0;
    t(1) = 1.0 / 9.0;
    t(2) = 1.0 / 9.0;
    t(3) = 1.0 / 9.0;
    t(4) = 1.0 / 9.0;
    t(5) = 1.0 / 36.0;
    t(6) = 1.0 / 36.0;
    t(7) = 1.0 / 36.0;
    t(8) = 1.0 / 36.0;

    bb(0) = 0;
    bb(1) = 3;
    bb(3) = 1;
    bb(2) = 4;
    bb(4) = 2;
    bb(5) = 7;
    bb(7) = 5;
    bb(6) = 8;
    bb(8) = 6;

    e(0, 0) = 0;
    e(1, 0) = 1;
    e(2, 0) = 0;
    e(3, 0) = -1;
    e(4, 0) = 0;
    e(5, 0) = 1;
    e(6, 0) = -1;
    e(7, 0) = -1;
    e(8, 0) = 1;

    e(0, 1) = 0;
    e(1, 1) = 0;
    e(2, 1) = 1;
    e(3, 1) = 0;
    e(4, 1) = -1;
    e(5, 1) = 1;
    e(6, 1) = 1;
    e(7, 1) = -1;
    e(8, 1) = -1;

    parallel_for(
        "init", mdrange_policy2({0, 0}, {lx, ly}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            int global_x = x_lo + i - ghost;
            int global_y = y_lo + j - ghost;

            double dist = 2.0 * (sqrt(pow((global_x - 0.5 * glx), 2) + pow(global_y - (gly * 0.5 + 0.2 * glx), 2)) - 0.2 * glx) / delta;
            double dist1 = 2.0 * (sqrt(pow((global_x - 0.5 * glx), 2) + pow(global_y - (gly * 0.5 - 0.2 * glx), 2)) - 0.2 * glx) / delta;

            ua(i, j) = 0.0;
            va(i, j) = 0.0;
            p(i, j) = 0.0;
            pp(i, j) = 0.0;
            double phi1 = 0.5 - 0.5 * tanh(dist1);
            phi(i, j) = (0.5 - 0.5 * tanh(dist)) + phi1;

            rho(i, j) = rho_l * phi(i, j) + rho_v * (1.0 - phi(i, j));
            tau(i, j) = tau0 * phi(i, j) + tau1 * (1.0 - phi(i, j));
            nu(i, j) = tau(i, j) * cs2;
        });
    fence();

    u_exchange(phi);
    u_exchange(p);
    u_exchange(pp);
    u_exchange(ua);
    u_exchange(va);

    dphi = d_c(phi);
    dp = d_c(p);
    dpp = d_c(pp);
    du = d_c(ua);
    dv = d_c(va);

    parallel_for(
        "cp_init", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            double sqd = sqrt(pow(dphi(0, i, j), 2) + pow(dphi(1, i, j), 2)) + 0.000001;

            divphix(i, j) = dphi(0, i, j) / sqd;
            divphiy(i, j) = dphi(1, i, j) / sqd;

            drho(0, i, j) = dphi(0, i, j) * rho_l - dphi(0, i, j) * rho_v;
            drho(1, i, j) = dphi(1, i, j) * rho_l - dphi(1, i, j) * rho_v;
        });
    fence();

    u_exchange(divphix);
    u_exchange(divphiy);
    div = div_c(divphix, divphiy);

    parallel_for(
        "initf", mdrange_policy3({0, 0, 0}, {q, lx, ly}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            double sqd = sqrt(pow(dphi(0, i, j), 2) + pow(dphi(1, i, j), 2)) + eps;

            double gamma = t(ii) * (1.0 + 3.0 * (e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j)) +
                                    4.5 * (pow((e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j)), 2)) -
                                    1.5 * (pow(ua(i, j), 2) + pow(va(i, j), 2)));

            double forsx = -3.0 / 2.0 * sigma * delta * div(i, j) * sqd * dphi(0, i, j);
            double forsy = -3.0 / 2.0 * sigma * delta * div(i, j) * sqd * dphi(1, i, j);

            double forspx = -dp(0, i, j);
            double forspy = -dp(1, i, j);

            double forsppx = dpp(0, i, j);
            double forsppy = dpp(1, i, j);

            double forsnx = nu(i, j) * (2.0 * du(0, i, j) * drho(0, i, j) + (du(1, i, j) + dv(0, i, j)) * drho(1, i, j));
            double forsny = nu(i, j) * (2.0 * dv(1, i, j) * drho(1, i, j) + (du(1, i, j) + dv(0, i, j)) * drho(0, i, j));

            double fors = (e(ii, 0) - ua(i, j)) * gamma / rho(i, j) *
                              (forspx + forsnx + forsx) +
                          (e(ii, 1) - va(i, j)) * gamma / rho(i, j) *
                              (forspy + forsny + forsy) +
                          (e(ii, 0) - ua(i, j)) * t(ii) * forsppx +
                          (e(ii, 1) - va(i, j)) * t(ii) * forsppy;

            double forphi = gamma * ((e(ii, 0) - ua(i, j)) * 4.0 * phi(i, j) * (1.0 - phi(i, j)) / delta * divphix(i, j) +
                                     (e(ii, 1) - va(i, j)) * 4.0 * phi(i, j) * (1.0 - phi(i, j)) / delta * divphiy(i, j));

            f(ii, i, j) = t(ii) * pp(i, j) + (gamma - t(ii)) * cs2 - 0.50 * fors;
            g(ii, i, j) = gamma * phi(i, j) - 0.50 * forphi;

            f_tem(ii, i, j) = 0.0;
            g_tem(ii, i, j) = 0.0;
        });
    fence();
};
void LBM::Collision()
{

    parallel_for(
        "collision", mdrange_policy3({0, ghost, ghost}, {q, lx - ghost, ly - ghost}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            double sqd = sqrt(pow(dphi(0, i, j), 2) + pow(dphi(1, i, j), 2)) + eps;

            double gamma = t(ii) * (1.0 + 3.0 * (e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j)) +
                                    4.5 * (pow((e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j)), 2)) -
                                    1.5 * (pow(ua(i, j), 2) + pow(va(i, j), 2)));

            double forsx = -3.0 / 2.0 * sigma * delta * div(i, j) * sqd * dphi(0, i, j);
            double forsy = -3.0 / 2.0 * sigma * delta * div(i, j) * sqd * dphi(1, i, j);

            double forspx = -dp(0, i, j);
            double forspy = -dp(1, i, j);

            double forsppx = dpp(0, i, j);
            double forsppy = dpp(1, i, j);

            double forsnx = nu(i, j) * (2.0 * du(0, i, j) * drho(0, i, j) + (du(1, i, j) + dv(0, i, j)) * drho(1, i, j));
            double forsny = nu(i, j) * (2.0 * dv(1, i, j) * drho(1, i, j) + (du(1, i, j) + dv(0, i, j)) * drho(0, i, j));

            double fors = (e(ii, 0) - ua(i, j)) * gamma / rho(i, j) *
                              (forspx + forsnx + forsx) +
                          (e(ii, 1) - va(i, j)) * gamma / rho(i, j) *
                              (forspy + forsny + forsy) +
                          (e(ii, 0) - ua(i, j)) * t(ii) * forsppx +
                          (e(ii, 1) - va(i, j)) * t(ii) * forsppy;

            double forphi = gamma * ((e(ii, 0) - ua(i, j)) * 4.0 * phi(i, j) * (1.0 - phi(i, j)) / delta * divphix(i, j) +
                                     (e(ii, 1) - va(i, j)) * 4.0 * phi(i, j) * (1.0 - phi(i, j)) / delta * divphiy(i, j));

            double feq = t(ii) * pp(i, j) + (gamma - t(ii)) * cs2 - 0.50 * fors;
            double geq = gamma * phi(i, j) - 0.50 * forphi;

            f(ii, i, j) = f(ii, i, j) - (f(ii, i, j) - feq) / (tau(i, j) + 0.5) + fors;
            g(ii, i, j) = g(ii, i, j) - (g(ii, i, j) - geq) / (taum + 0.5) + forphi;
        });
    fence();
};

void LBM::Streaming()
{
    exchange(f);
    exchange(g);

    fence();
    parallel_for(
        "stream1", mdrange_policy3({0, ghost, ghost}, {q, lx - ghost, ly - ghost}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            f_tem(ii, i, j) = f(ii, i - e(ii, 0), j - e(ii, 1));
            g_tem(ii, i, j) = g(ii, i - e(ii, 0), j - e(ii, 1));
        });

    parallel_for(
        "stream2", mdrange_policy3({0, ghost, ghost}, {q, lx - ghost, ly - ghost}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            f(ii, i, j) = f_tem(ii, i, j);
            g(ii, i, j) = g_tem(ii, i, j);
        });
    fence();
};

void LBM::Update()
{
    typedef TeamPolicy<> team_policy;
    typedef TeamPolicy<>::member_type member_type;
    parallel_for(
        "update", team_policy(ly , AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int j = team_member.league_rank();
            parallel_for(
                TeamThreadRange(team_member, 0, lx ), [&](const int &i)
                {
                    phi(i, j) = 0.0;
                    parallel_reduce(
                        ThreadVectorRange(team_member, q), [&](const int &ii, double &phi_tem)
                        { phi_tem += g(ii, i, j); },
                        phi(i, j));
                }); });
    fence();
    parallel_for(
        "initialize", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            rho(i, j) = rho_l * phi(i, j) + rho_v * (1.0 - phi(i, j));
            tau(i, j) = tau0 * phi(i, j) + tau1 * (1.0 - phi(i, j));
            nu(i, j) = tau(i, j) * cs2;
        });

    parallel_for(
        "update", team_policy(ly , AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int j = team_member.league_rank();
            parallel_for(
                TeamThreadRange(team_member, 0, lx ), [&](const int &i)
                {
                    pp(i, j) = 0.0;

                    parallel_reduce(
                        ThreadVectorRange(team_member, q), [&](const int &ii, double &pp_tem)
                        { pp_tem += f(ii, i, j); },
                        pp(i, j));

                    pp(i, j) = pp(i, j) - (ua(i, j) * dpp(0, i, j) + va(i, j) * dpp(1, i, j)) / 2.0;
                    p(i, j) = pp(i, j) * rho(i, j); }); });
    u_exchange(phi);
    u_exchange(p);
    u_exchange(pp);
    u_exchange(ua);
    u_exchange(va);

    dphi = d_c(phi);
    dp = d_c(p);
    dpp = d_c(pp);
    du = d_c(ua);
    dv = d_c(va);

    parallel_for(
        "cp_init", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            double sqd = sqrt(pow(dphi(0, i, j), 2) + pow(dphi(1, i, j), 2)) + 0.000001;

            divphix(i, j) = dphi(0, i, j) / sqd;
            divphiy(i, j) = dphi(1, i, j) / sqd;

            drho(0,i, j) = dphi(0, i, j) * rho_l  - dphi(0, i, j) * rho_v;
            drho(1, i, j) = dphi(1, i, j) * rho_l - dphi(1, i, j) * rho_v;
        });
    fence();

    u_exchange(divphix);
    u_exchange(divphiy);
    div = div_c(divphix, divphiy);

    parallel_for(
        "update", team_policy(ly, AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int j = team_member.league_rank();
            parallel_for(
                TeamThreadRange(team_member, 0, lx ), [&](const int &i)
                {
                    ua(i, j) = 0.0;
                    va(i, j) = 0.0;

                    parallel_reduce(
                        ThreadVectorRange(team_member, q), [&](const int &ii, double &u_tem)
                        { u_tem += f(ii, i, j) * e(ii, 0); },
                        ua(i, j));

                    parallel_reduce(
                        ThreadVectorRange(team_member, q), [&](const int &ii, double &v_tem)
                        { v_tem += f(ii, i, j) * e(ii, 1); },
                        va(i, j));
                }); });

    fence();
    parallel_for(
        "stream1", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            double sqd = sqrt(pow(dphi(0, i, j), 2) + pow(dphi(1, i, j), 2)) + 0.000001;

            double forsx = -3.0 / 2.0 * sigma * delta * div(i, j) * sqd * dphi(0, i, j);
            double forsy = -3.0 / 2.0 * sigma * delta * div(i, j) * sqd * dphi(1, i, j);

            double forspx = -dp(0, i, j);
            double forspy = -dp(1, i, j);

            double forsppx = dpp(0, i, j) * rho(i, j);
            double forsppy = dpp(1, i, j) * rho(i, j);

            double forsnx = nu(i, j) * (2.0 * du(0, i, j) * drho(0, i, j) + (du(1, i, j) + dv(0, i, j)) * drho(1, i, j));
            double forsny = nu(i, j) * (2.0 * dv(1, i, j) * drho(1, i, j) + (du(1, i, j) + dv(0, i, j)) * drho(0, i, j));

            ua(i, j) = ua(i, j) / cs2 + (forspx + forsppx + forsnx + forsx) / 2.0 / rho(i, j);
            va(i, j) = va(i, j) / cs2 + (forspy + forsppy + forsny + forsy) / 2.0 / rho(i, j);
        });
    fence();
};


void LBM::MPIoutput(int n)
{
    // MPI_IO
    MPI_File fh;
    MPI_Status status;
    MPI_Offset offset = 0;

    MPI_Datatype FILETYPE, DATATYPE;
    // buffer
    int tp;
    float ttp;
    double fp;
    // min max
    double umin, umax, vmin, vmax, pmin, pmax,rhomin,rhomax;
    double uumin, uumax, vvmin, vvmax, ppmin, ppmax,rhomin_,rhomax_;
    // transfer
    double *uu, *vv, *pp, *xx, *yy, *rr;
    uu = (double *)malloc((lx - 2 * ghost) * (ly - 2 * ghost) * sizeof(double));
    vv = (double *)malloc((lx - 2 * ghost) * (ly - 2 * ghost) * sizeof(double));
    rr = (double *)malloc((lx - 2 * ghost) * (ly - 2 * ghost) * sizeof(double));
    xx = (double *)malloc((lx - 2 * ghost) * (ly - 2 * ghost) * sizeof(double));
    yy = (double *)malloc((lx - 2 * ghost) * (ly - 2 * ghost) * sizeof(double));

    for (int j = 0; j < (ly - 2 * ghost); j++)
    {
        for (int i = 0; i < (lx - 2 * ghost); i++)
        {

            uu[i + j * (lx - 2 * ghost)] = ua(i + ghost, j + ghost);
            vv[i + j * (lx - 2 * ghost)] = va(i + ghost, j + ghost);
            rr[i + j * (lx - 2 * ghost)] = rho(i + ghost, j + ghost);
            xx[i + j * (lx - 2 * ghost)] = 2.0 * (double)(x_lo + i) / (glx - 1);
            yy[i + j * (lx - 2 * ghost)] = 3.0 * (double)(y_lo + j) / (gly - 1);
        }
    }

    parallel_reduce(
        " Label", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value = ua(i,j);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(umax));
    fence();
    parallel_reduce(
        " Label", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value = va(i,j);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(vmax));
    fence();

    parallel_reduce(
        " Label", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value =rho(i,j);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(rhomax));
    fence();
    parallel_reduce(
        " Label", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value = ua(i,j);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(umin));
    fence();
    parallel_reduce(
        " Label", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value = va(i,j);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(vmin));
    fence();

    parallel_reduce(
        " Label", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, double &valueToUpdate) {
         double my_value = rho(i,j);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(rhomin));
    fence();
    std::string str1 = "output" + std::to_string(n) + ".plt";
    const char *na = str1.c_str();
    std::string str2 = "#!TDV112";
    const char *version = str2.c_str();
    MPI_File_open(MPI_COMM_WORLD, na, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    MPI_Reduce(&umin, &uumin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&umax, &uumax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&vmin, &vvmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&vmax, &vvmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&rhomin, &rhomin_, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rhomax, &rhomax_, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (comm.me == 0)
    {

        MPI_File_seek(fh, offset, MPI_SEEK_SET);
        // header !version number
        MPI_File_write(fh, version, 8, MPI_CHAR, &status);
        // INTEGER 1
        tp = 1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 3*4+8=20
        // variable name
        tp = 5;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 120;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 121;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 117;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 118;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 112;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 20+11*4=64
        // Zone Marker
        ttp = 299.0;
        MPI_File_write(fh, &ttp, 1, MPI_REAL, &status);
        // Zone Name
        tp = 90;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 79;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 78;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 69;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 32;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 48;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 48;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 49;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 64 + 10 * 4 = 104

        // paraents
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // strendid
        tp = -2;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // SOLUTION TIME
        double nn = (double)n;
        fp = nn;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        // zone color
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ZONE type
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // specify var location
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // are raw local
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // number of miscellaneous
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ordered zone
        tp = 0;

        tp = glx;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = gly;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // AUXILIARY
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // 104 + 13 * 4 = 156
        // EOHMARKER
        ttp = 357.0;
        MPI_File_write(fh, &ttp, 1, MPI_REAL, &status);
        // DATA SECTION
        ttp = 299.0;
        MPI_File_write(fh, &ttp, 1, MPI_REAL, &status);
        // VARIABLE DATA FORMAT
        tp = 2;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // PASSIVE VARIABLE
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // SHARING VARIABLE
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ZONE NUMBER
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // 156 + 10 * 4 = 196
        // MIN AND MAX VALUE FLOAT 64
        fp = 0.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 2.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 0.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 3.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = uumin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = uumax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = vvmin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = vvmax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = rhomin_;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = rhomax_;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);

        // 196 + 10 * 8 = 276
    }

    offset = 276;

    int glolen[2] = {glx, gly};
    int localstart[2] = {x_lo, y_lo};
    int l_l[2] = {lx - 2 * ghost, ly - 2 * ghost};
    MPI_Type_create_subarray(dim, glolen, l_l, localstart, MPI_ORDER_FORTRAN, MPI_DOUBLE, &DATATYPE);

    MPI_Type_commit(&DATATYPE);

    MPI_Type_contiguous(5, DATATYPE, &FILETYPE);

    MPI_Type_commit(&FILETYPE);

    MPI_File_set_view(fh, offset, MPI_DOUBLE, FILETYPE, "native", MPI_INFO_NULL);

    MPI_File_write_all(fh, xx, (lx - 2 * ghost) * (ly - 2 * ghost), MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, yy, (lx - 2 * ghost) * (ly - 2 * ghost), MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, uu, (lx - 2 * ghost) * (ly - 2 * ghost), MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, vv, (lx - 2 * ghost) * (ly - 2 * ghost), MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, rr, (lx - 2 * ghost) * (ly - 2 * ghost), MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);

    free(uu);
    free(vv);
    free(rr);
    free(xx);
    free(yy);

    MPI_Barrier(MPI_COMM_WORLD);
};
void LBM::Output(int n)
{
    std::ofstream outfile;
    std::string str = "output" + std::to_string(n) + std::to_string(comm.me);
    outfile << std::setiosflags(std::ios::fixed);
    outfile.open(str + ".dat", std::ios::out);

    outfile << "variables=x,y,u,v,p" << std::endl;
    outfile << "zone I=" << lx  << ",J=" << ly  << std::endl;

    for (int j = 0; j < ly ; j++)
    {
        for (int i = 0; i < lx ; i++)
        {

            outfile << std::setprecision(8) << setiosflags(std::ios::left) << (i - ghost + x_lo) / (glx - 1.0) << " " << (j - ghost + y_lo) / (gly - 1.0) << " " << ua(i, j) << " " << va(i, j) << " " << rho(i, j) << std::endl;
        }
    }

    outfile.close();
    if (comm.me == 0)
    {
        printf("\n");
        printf("The result %d is writen\n", n);
        printf("\n");
        printf("============================\n");
    }
};


View<double**,CudaUVMSpace> LBM::laplace(View<double**,CudaUVMSpace> c)
{

   View<double **, CudaUVMSpace> la_ = View<double **, CudaUVMSpace>("la_", lx, ly);

    typedef TeamPolicy<> team_policy;
    typedef TeamPolicy<>::member_type member_type;

    parallel_for(
        "laplace", team_policy(ly-2*ghost, AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int j = team_member.league_rank()+ghost;
            parallel_for(
            TeamThreadRange(team_member, ghost,lx-ghost), [&](const int &i)
            {
                            la_(i, j) = 0.0;


                        parallel_reduce(
                                         ThreadVectorRange(team_member, q),[&](const int& ii, double &la_tem) {
                        la_tem += t(ii) * (c(i +  e(ii, 0), j +  e(ii, 1)) + c(i -  e(ii, 0), j -  e(ii, 1)) - 2 * c(i, j)) / 2.0 /cs2;},
                        la_(i, j));


             }); });


    fence();
    return la_;
};

View<double***,CudaUVMSpace> LBM::d_c(View<double**,CudaUVMSpace> c)
{
  View<double ***, CudaUVMSpace> dc= View<double ***, CudaUVMSpace>("dc_", dim, lx, ly);
    typedef TeamPolicy<> team_policy;
    typedef TeamPolicy<>::member_type member_type;

    parallel_for(
        "dc", team_policy(ly-2*ghost, AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int j = team_member.league_rank()+ghost;
            parallel_for(
            TeamThreadRange(team_member, ghost,lx-ghost), [&](const int &i)
            {
                            dc(0, i, j) = 0.0;
                            dc(1, i, j) = 0.0;

                        parallel_reduce(
                                         ThreadVectorRange(team_member, q),[&](const int& ii, double &dc0_tem) {
                        dc0_tem += t(ii) *  e(ii, 0) * (c(i + e(ii, 0), j + e(ii, 1)) - c(i - e(ii, 0), j - e(ii, 1))) / 2.0 / cs2;},
                        dc(0,i,j));

                        parallel_reduce(
                                         ThreadVectorRange(team_member, q),[&](const int& ii, double &dc1_tem) {
                        dc1_tem += t(ii) *  e(ii, 1) * (c(i + e(ii, 0), j + e(ii, 1)) - c(i - e(ii, 0), j - e(ii, 1))) / 2.0 / cs2;},
                        dc(1,i,j));


             }); });

fence();
    return dc;
};

View<double***,CudaUVMSpace> LBM::d_m(View<double**,CudaUVMSpace> c)
{

   View<double ***, CudaUVMSpace> dm = View<double ***, CudaUVMSpace>("dm_", dim, lx, ly);

    typedef TeamPolicy<> team_policy;
    typedef TeamPolicy<>::member_type member_type;


    parallel_for(
        "dm", team_policy(ly-2*ghost, AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int j = team_member.league_rank()+ghost;
            parallel_for(
            TeamThreadRange(team_member, ghost,lx-ghost), [&](const int &i)
            {
                            dm(0, i, j) = 0.0;
                            dm(1, i, j) = 0.0;

                            parallel_reduce(
                                ThreadVectorRange(team_member, q), [&](const int &ii, double &db0_tem)
                                { double temp = 0.25 * (5.0 * c(i + e(ii, 0), j + e(ii, 1)) - 3.0 * c(i, j) - c(i - e(ii, 0), j - e(ii, 1)) - c(i + 2 * e(ii, 0), j + 2 * e(ii, 1)));
                                    db0_tem += t(ii) * e(ii, 0) * temp / cs2; },
                                dm(0, i, j));

                            parallel_reduce(
                                ThreadVectorRange(team_member, q), [&](const int &ii, double &db1_tem)
                                { double temp = 0.25 * (5.0 * c(i + e(ii, 0), j + e(ii, 1)) - 3.0 * c(i, j) - c(i - e(ii, 0), j - e(ii, 1)) - c(i + 2 * e(ii, 0), j + 2 * e(ii, 1)));
                                    db1_tem += t(ii) * e(ii, 1) * temp / cs2; },
                                dm(1, i, j));


             }); });

    fence();
    return dm;
};


View<double**,CudaUVMSpace> LBM::div_c(View<double**,CudaUVMSpace> cx,View<double**,CudaUVMSpace> cy)
{
  View<double **, CudaUVMSpace> divc= View<double **, CudaUVMSpace>("divc", lx, ly);
    typedef TeamPolicy<> team_policy;
    typedef TeamPolicy<>::member_type member_type;

        parallel_for(
        "dm", team_policy(ly-2*ghost, AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int j = team_member.league_rank()+ghost;
            parallel_for(
            TeamThreadRange(team_member, ghost,lx-ghost), [&](const int &i)
            {

                    double divx = 0.0;
                    double divy = 0.0;

                    parallel_reduce(
                        ThreadVectorRange(team_member, q), [&](const int &ii, double &divc0_tem)
                        { divc0_tem += t(ii) * e(ii, 0) * (cx(i + e(ii, 0), j + e(ii, 1)) - cx(i - e(ii, 0), j - e(ii, 1))) / 2.0 * 3.0; },
                        divx);

                    parallel_reduce(
                        ThreadVectorRange(team_member, q), [&](const int &ii, double &divc1_tem)
                        { divc1_tem += t(ii) * e(ii, 1) * (cy(i + e(ii, 0), j + e(ii, 1)) - cy(i - e(ii, 0), j - e(ii, 1))) / 2.0 * 3.0; },
                        divy);

                    divc(i, j) = divx + divy;
             }); });


    fence();
    return divc;
};