#include "lbm.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>
#define pi 3.1415926

using namespace std;
void LBM::Initialize()
{

    f = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("f", q, lx, ly, lz);
    ft = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("ft", q, lx, ly, lz);
    fb = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("fb", q, lx, ly, lz);

    g = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("g", q, lx, ly, lz);
    gt = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("gt", q, lx, ly, lz);
    gb = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("gb", q, lx, ly, lz);

    ua = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("u", lx, ly, lz);
    va = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("v", lx, ly, lz);
    wa = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("v", lx, ly, lz);
    rho = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("rho", lx, ly, lz);
    p = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("p", lx, ly, lz);
    pp = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("pp", lx, ly, lz);
    phi = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("phi", lx, ly, lz);
    tau = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("tau", lx, ly, lz);
    nu = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("nu", lx, ly, lz);

    e = Kokkos::View<int **, Kokkos::CudaUVMSpace>("e", q, dim);
    t = Kokkos::View<double *, Kokkos::CudaUVMSpace>("t", q);
    usr = Kokkos::View<int ***, Kokkos::CudaUVMSpace>("usr", lx, ly, lz);
    ran = Kokkos::View<int ***, Kokkos::CudaUVMSpace>("ran", lx, ly, lz);
    bb = Kokkos::View<int *, Kokkos::CudaUVMSpace>("b", q);

    du = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("du", dim,lx, ly, lz);

    dv = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("dv", dim,lx, ly, lz);

    dw = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("dw", dim,lx, ly, lz);

    dphi = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("dphi", dim,lx, ly, lz);

    drho = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("drho", dim,lx, ly, lz);

    dp = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("dp", dim,lx, ly, lz);

    dpp = Kokkos::View<double ****, Kokkos::CudaUVMSpace>("dpp", dim,lx, ly, lz);

    divphix = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("divphix", lx, ly, lz);
    divphiy = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("divphiy", lx, ly, lz);
    divphiz = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("divphiz", lx, ly, lz);
    div = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("div", lx, ly, lz);

    //  weight function
    t(0) = 8.0 / 27.0;
    t(1) = 2.0 / 27.0;
    t(2) = 2.0 / 27.0;
    t(3) = 2.0 / 27.0;
    t(4) = 2.0 / 27.0;
    t(5) = 2.0 / 27.0;
    t(6) = 2.0 / 27.0;
    t(7) = 1.0 / 54.0;
    t(8) = 1.0 / 54.0;
    t(9) = 1.0 / 54.0;
    t(10) = 1.0 / 54.0;
    t(11) = 1.0 / 54.0;
    t(12) = 1.0 / 54.0;
    t(13) = 1.0 / 54.0;
    t(14) = 1.0 / 54.0;
    t(15) = 1.0 / 54.0;
    t(16) = 1.0 / 54.0;
    t(17) = 1.0 / 54.0;
    t(18) = 1.0 / 54.0;
    t(19) = 1.0 / 216.0;
    t(20) = 1.0 / 216.0;
    t(21) = 1.0 / 216.0;
    t(22) = 1.0 / 216.0;
    t(23) = 1.0 / 216.0;
    t(24) = 1.0 / 216.0;
    t(25) = 1.0 / 216.0;
    t(26) = 1.0 / 216.0;
    // bounce back directions
    bb(0) = 0;
    bb(1) = 2;
    bb(2) = 1;
    bb(3) = 4;
    bb(4) = 3;
    bb(5) = 6;
    bb(6) = 5;
    bb(7) = 8;
    bb(8) = 7;
    bb(9) = 10;
    bb(10) = 9;
    bb(11) = 12;
    bb(12) = 11;
    bb(13) = 14;
    bb(14) = 13;
    bb(15) = 16;
    bb(16) = 15;
    bb(17) = 18;
    bb(18) = 17;
    bb(19) = 20;
    bb(20) = 19;
    bb(21) = 22;
    bb(22) = 21;
    bb(23) = 24;
    bb(24) = 23;
    bb(25) = 26;
    bb(26) = 25;

    // discrete velocity
    e(0, 0) = 0;
    e(0, 1) = 0;
    e(0, 2) = 0;

    e(1, 0) = 1;
    e(1, 1) = 0;
    e(1, 2) = 0;

    e(2, 0) = -1;
    e(2, 1) = 0;
    e(2, 2) = 0;

    e(3, 0) = 0;
    e(3, 1) = 1;
    e(3, 2) = 0;

    e(4, 0) = 0;
    e(4, 1) = -1;
    e(4, 2) = 0;

    e(5, 0) = 0;
    e(5, 1) = 0;
    e(5, 2) = 1;

    e(6, 0) = 0;
    e(6, 1) = 0;
    e(6, 2) = -1;

    e(7, 0) = 1;
    e(7, 1) = 1;
    e(7, 2) = 0;

    e(8, 0) = -1;
    e(8, 1) = -1;
    e(8, 2) = 0;

    e(9, 0) = 1;
    e(9, 1) = -1;
    e(9, 2) = 0;

    e(10, 0) = -1;
    e(10, 1) = 1;
    e(10, 2) = 0;

    e(11, 0) = 1;
    e(11, 1) = 0;
    e(11, 2) = 1;

    e(12, 0) = -1;
    e(12, 1) = 0;
    e(12, 2) = -1;

    e(13, 0) = 1;
    e(13, 1) = 0;
    e(13, 2) = -1;

    e(14, 0) = -1;
    e(14, 1) = 0;
    e(14, 2) = 1;

    e(15, 0) = 0;
    e(15, 1) = 1;
    e(15, 2) = 1;

    e(16, 0) = 0;
    e(16, 1) = -1;
    e(16, 2) = -1;

    e(17, 0) = 0;
    e(17, 1) = 1;
    e(17, 2) = -1;

    e(18, 0) = 0;
    e(18, 1) = -1;
    e(18, 2) = 1;

    e(19, 0) = 1;
    e(19, 1) = 1;
    e(19, 2) = 1;

    e(20, 0) = -1;
    e(20, 1) = -1;
    e(20, 2) = -1;

    e(21, 0) = 1;
    e(21, 1) = -1;
    e(21, 2) = 1;

    e(22, 0) = -1;
    e(22, 1) = 1;
    e(22, 2) = -1;

    e(23, 0) = 1;
    e(23, 1) = 1;
    e(23, 2) = -1;

    e(24, 0) = -1;
    e(24, 1) = -1;
    e(24, 2) = 1;

    e(25, 0) = 1;
    e(25, 1) = -1;
    e(25, 2) = -1;

    e(26, 0) = -1;
    e(26, 1) = 1;
    e(26, 2) = 1;

    setup_u();
    // macroscopic value initialization

    // macroscopic value initialization
    Kokkos::parallel_for(
        "initialize", mdrange_policy3({0, 0, 0}, {lx, ly, lz}), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            ua(i, j, k) =  u0 * sin((double)((double)(i - ghost + x_lo) / (double) glx * 2.0 * pi)) * cos((double)((double)(j - ghost + y_lo) / (double) gly * 2.0 * pi))
                              * cos((double)((double)(k - ghost + z_lo) / (double) glz * 2.0 * pi));
            va(i, j, k) = -u0 * cos((double)((double)(i - ghost + x_lo) / (double) glx * 2.0 * pi)) * sin((double)((double)(j - ghost + y_lo) / (double) gly * 2.0 * pi)) 
                              * cos((double)((double)(k - ghost + z_lo) / (double) glz * 2.0 * pi));
            wa(i, j, k) = 0.0;
            p(i, j, k)  = 0.0;
            pp(i, j, k) = 0.0;

            double dist = 2.0 * (pow((pow((i - ghost + x_lo - 0.45 * glx), 2) + pow((j - ghost + y_lo - 0.45 * gly), 2) + pow((k - ghost + z_lo - 0.45 * glz), 2)), 0.5) - 0.2 * glx) / delta;

            phi(i, j, k) = 0.5 - 0.5 * tanh(dist);

            rho(i, j, k) = rho0 * phi(i, j, k) + rho1 * (1.0 - phi(i, j, k));
            tau(i, j, k) = tau0 * phi(i, j, k) + tau1 * (1.0 - phi(i, j, k));
            nu(i, j, k) = tau(i, j, k) * cs2;
        });

    Kokkos::fence();
    pass(phi);
    pass(p);
    pass(pp);
    pass(ua);
    pass(va);
    pass(wa);

    dphi = d_c(phi);
    dp = d_c(p);
    dpp = d_c(pp);
    du = d_c(ua);
    dv = d_c(va);
    dw = d_c(wa);

    Kokkos::parallel_for(
        "initialize", mdrange_policy3({l_s[0], l_s[1], l_s[2]}, {l_e[0], l_e[1], l_e[2]}), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            double sqd = pow(pow(dphi(0, i, j, k), 2) + pow(dphi(1, i, j, k), 2) + pow(dphi(2, i, j, k), 2), 0.5) + eps;

            divphix(i, j, k) = dphi(0, i, j, k) / sqd;
            divphiy(i, j, k) = dphi(1, i, j, k) / sqd;
            divphiz(i, j, k) = dphi(2, i, j, k) / sqd;

            drho(0,i, j, k) = dphi(0, i, j, k) * rho0 + (1.0 - dphi(0, i, j, k)) * rho1;
            drho(1,i, j, k) = dphi(1, i, j, k) * rho0 + (1.0 - dphi(1, i, j, k)) * rho1;
            drho(2,i, j, k) = dphi(2, i, j, k) * rho0 + (1.0 - dphi(2, i, j, k)) * rho1;
        });

    pass(divphix);
    pass(divphiy);
    pass(divphiz);
    div = div_c(divphix,divphiy,divphiz);

    // distribution function initialization
    Kokkos::parallel_for(
        "initf", mdrange_policy4({0, 0, 0, 0}, {q, lx, ly, lz}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j, const int k) {

            double sqd = sqrt(pow(dphi(0, i, j, k), 2) + pow(dphi(1, i, j, k), 2) + pow(dphi(2, i, j, k), 2)) + eps;

            double gamma = t(ii) * (1.0 + 3.0 * (e(ii, 0) * ua(i, j, k) + e(ii, 1) * va(i, j, k) + e(ii, 2) * wa(i, j, k)) +
                                          4.5 * (pow((e(ii, 0) * ua(i, j, k) + e(ii, 1) * va(i, j, k) + e(ii, 2) * wa(i, j, k)), 2)) -
                                          1.5 * (pow(ua(i, j, k), 2) + pow(va(i, j, k), 2) + pow(wa(i, j, k), 2)));

            double forsx = -3.0 / 2.0 * sigma * delta * div(i, j, k) * sqd * dphi(0, i, j, k);
            double forsy = -3.0 / 2.0 * sigma * delta * div(i, j, k) * sqd * dphi(1, i, j, k);
            double forsz = -3.0 / 2.0 * sigma * delta * div(i, j, k) * sqd * dphi(2, i, j, k);

            double forspx = -dp(0, i, j, k);
            double forspy = -dp(1, i, j, k);
            double forspz = -dp(2, i, j, k);

            double forsppx = dpp(0, i, j, k);
            double forsppy = dpp(1, i, j, k);
            double forsppz = dpp(2, i, j, k);

            double forsnx = nu(i, j, k) * (2.0 * du(0, i, j, k) * drho(0, i, j, k) + (du(1, i,j, k) + dv(0, i, j, k)) * drho(1, i, j, k) + (du(2, i, j, k) + dw(0, i, j, k)) * drho(2, i, j, k));
            double forsny = nu(i, j, k) * (2.0 * dv(1, i, j, k) * drho(1, i, j, k) + (du(1, i,j, k) + dv(0, i, j, k)) * drho(0, i, j, k) + (dv(2, i, j, k) + dw(1, i, j, k)) * drho(2, i, j, k));
            double forsnz = nu(i, j, k) * (2.0 * dw(2, i, j, k) * drho(2, i, j, k) + (du(2, i,j, k) + dw(0, i, j, k)) * drho(0, i, j, k) + (dw(1, i, j, k) + dv(2, i, j, k)) * drho(1, i, j, k));

            double fors = (e(ii, 0) - ua(i, j, k)) * gamma / rho(i, j, k) *
                              (forspx + forsnx + forsx) +
                          (e(ii, 1) - va(i, j, k)) * gamma / rho(i, j, k) *
                              (forspy + forsny + forsy) +
                          (e(ii, 2) - wa(i, j, k)) * gamma / rho(i, j, k) *
                              (forspz + forsnz + forsz) +
                          (e(ii, 0) - ua(i, j, k)) * t(ii) * forsppx +
                          (e(ii, 1) - va(i, j, k)) * t(ii) * forsppy +
                          (e(ii, 2) - wa(i, j, k)) * t(ii) * forsppz;

            double forphi = gamma * ((e(ii, 0) - ua(i, j, k)) * 4.0 * phi(i, j, k) * (1.0 - phi(i, j, k)) / delta * divphix(i, j, k) +
                                     (e(ii, 1) - va(i, j, k)) * 4.0 * phi(i, j, k) * (1.0 - phi(i, j, k)) / delta * divphiy(i, j, k) +
                                     (e(ii, 2) - wa(i, j, k)) * 4.0 * phi(i, j, k) * (1.0 - phi(i, j, k)) / delta * divphiz(i, j, k));

            f(ii, i, j, k) = t(ii) * pp(i, j, k) + (gamma - t(ii)) * cs2 - 0.50 * fors;
            g(ii, i, j, k) = gamma * phi(i, j, k) - 0.50 * forphi;
        });

    Kokkos::fence();
};
void LBM::Collision()
{
    // collision

    Kokkos::parallel_for(
        "collision", mdrange_policy4({0, l_s[0], l_s[1], l_s[2]}, {q, l_e[0], l_e[1], l_e[2]}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j, const int k) {

            double sqd = sqrt(pow(dphi(0, i, j, k), 2) + pow(dphi(1, i, j, k), 2) + pow(dphi(2, i, j, k), 2)) + eps;

            double gamma = t(ii) * (1.0 + 3.0 * (e(ii, 0) * ua(i, j, k) + e(ii, 1) * va(i, j, k) + e(ii, 2) * wa(i, j, k)) +
                                          4.5 * (pow((e(ii, 0) * ua(i, j, k) + e(ii, 1) * va(i, j, k) + e(ii, 2) * wa(i, j, k)), 2)) -
                                          1.5 * (pow(ua(i, j, k), 2) + pow(va(i, j, k), 2) + pow(wa(i, j, k), 2)));

            double forsx = -3.0 / 2.0 * sigma * delta * div(i, j, k) * sqd * dphi(0, i, j, k);
            double forsy = -3.0 / 2.0 * sigma * delta * div(i, j, k) * sqd * dphi(1, i, j, k);
            double forsz = -3.0 / 2.0 * sigma * delta * div(i, j, k) * sqd * dphi(2, i, j, k);

            double forspx = -dp(0, i, j, k);
            double forspy = -dp(1, i, j, k);
            double forspz = -dp(2, i, j, k);

            double forsppx = dpp(0, i, j, k);
            double forsppy = dpp(1, i, j, k);
            double forsppz = dpp(2, i, j, k);

            double forsnx = nu(i, j, k) * (2.0 * du(0, i, j, k) * drho(0, i, j, k) 
                                               + (du(1, i,j, k) + dv(0, i, j, k)) * drho(1, i, j, k) 
                                               + (du(2, i, j, k) + dw(0, i, j, k)) * drho(2, i, j, k));

            double forsny = nu(i, j, k) * (2.0 * dv(1, i, j, k) * drho(1, i, j, k) 
                                               + (du(1, i,j, k)+ dv(0, i, j, k)) * drho(0, i, j, k) 
                                               + (dv(2, i, j, k) + dw(1, i, j, k)) * drho(2, i, j, k));

            double forsnz = nu(i, j, k) * (2.0 * dw(2, i, j, k) * drho(2, i, j, k) 
                                               + (du(2, i,j, k) + dw(0, i, j, k)) * drho(0, i, j, k) 
                                               + (dw(1, i, j, k) + dv(2, i, j, k)) * drho(1, i, j, k));

            double fors = (e(ii, 0) - ua(i, j, k)) * gamma / rho(i, j, k) * (forspx + forsnx + forsx) +
                          (e(ii, 1) - va(i, j, k)) * gamma / rho(i, j, k) * (forspy + forsny + forsy) +
                          (e(ii, 2) - wa(i, j, k)) * gamma / rho(i, j, k) * (forspz + forsnz + forsz) +
                          (e(ii, 0) - ua(i, j, k)) * t(ii) * forsppx +
                          (e(ii, 1) - va(i, j, k)) * t(ii) * forsppy +
                          (e(ii, 2) - wa(i, j, k)) * t(ii) * forsppz;

            double forphi = gamma * ((e(ii, 0) - ua(i, j, k)) * 4.0 * phi(i, j, k) * (1.0 - phi(i, j, k)) / delta * divphix(i, j, k) +
                                     (e(ii, 1) - va(i, j, k)) * 4.0 * phi(i, j, k) * (1.0 - phi(i, j, k)) / delta * divphiy(i, j, k) +
                                     (e(ii, 2) - wa(i, j, k)) * 4.0 * phi(i, j, k) * (1.0 - phi(i, j, k)) / delta * divphiz(i, j, k));

            double feq = t(ii) * pp(i, j, k) + (gamma - t(ii)) * cs2 - 0.50 * fors;

            double geq = gamma * phi(i, j, k)  - 0.50 * forphi;

            f(ii, i, j, k) = f(ii, i, j, k) - (f(ii, i, j, k) - feq) / (tau(i, j, k) + 0.5) + fors;
            g(ii, i, j, k) = g(ii, i, j, k) - (g(ii, i, j, k) - geq) / (taum + 0.5) + forphi;
        });
    Kokkos::fence();
};

void LBM::Streaming()
{
    passf(f);
    passf(g);

    // streaming process
    Kokkos::parallel_for(
        "stream1", mdrange_policy4({0, ghost, ghost, ghost}, {q, lx - ghost, ly - ghost, lz - ghost}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j, const int k) {
            ft(ii, i, j, k) = f(ii, i - e(ii, 0), j - e(ii, 1), k - e(ii, 2));
            gt(ii, i, j, k) = g(ii, i - e(ii, 0), j - e(ii, 1), k - e(ii, 2));
        });

    Kokkos::fence();

    Kokkos::parallel_for(
        "stream2", mdrange_policy4({0, ghost, ghost, ghost}, {q, lx - ghost, ly - ghost, lz - ghost}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j, const int k) {
            f(ii, i, j, k) = ft(ii, i, j, k);
            g(ii, i, j, k) = gt(ii, i, j, k);
        });

    Kokkos::fence();
};

void LBM::Update()
{
    typedef Kokkos::TeamPolicy<> team_policy;
    typedef Kokkos::TeamPolicy<>::member_type member_type;
    Kokkos::parallel_for(
        "update", team_policy(lz-2*ghost, Kokkos::AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int k = team_member.league_rank()+ghost;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, (lx - 2 * ghost) * (ly - 2 * ghost)), [&](const int &ij)
                {
                    const int i = ij % (lx - 2 * ghost) + ghost;
                    const int j = ij / (lx - 2 * ghost) + ghost;
                    phi(i, j, k) = 0.0;

                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &phim)
                        { phim += g(ii, i, j, k) ; },
                        phi(i, j, k)); });
        });

    Kokkos::parallel_for(
        "initialize", mdrange_policy3({ghost, ghost, ghost}, {lx-ghost, ly-ghost, lz-ghost}), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {

            rho(i, j, k) = rho0 * phi(i, j, k) + rho1 * (1.0 - phi(i, j, k));
            tau(i, j, k) = tau0 * phi(i, j, k) + tau1 * (1.0 - phi(i, j, k));
            nu(i, j, k) = tau(i, j, k) * cs2;
        });




    Kokkos::parallel_for(
        "update", team_policy(lz-2*ghost, Kokkos::AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int k = team_member.league_rank()+ghost;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, (lx - 2 * ghost) * (ly - 2 * ghost)), [&](const int &ij)
                {
                    const int i = ij % (lx - 2 * ghost) + ghost;
                    const int j = ij / (lx - 2 * ghost) + ghost;

                    pp(i, j, k) = 0.0;

                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &ppm)
                        { ppm += f(ii, i, j, k) ; },
                        pp(i, j, k));


                    pp(i, j, k) = pp(i, j, k) - (ua(i, j, k)  * dpp(0, i, j, k) 
                                              +  va(i, j, k)  * dpp(1, i, j, k) 
                                              +  wa(i, j, k)  * dpp(2, i, j, k))/2.0;
                    p(i,j,k)  = pp(i,j,k)*rho(i,j,k); });
        });

    pass(phi);
    pass(p);
    pass(pp);
    pass(ua);
    pass(va);
    pass(wa);

    dphi = d_c(phi);
    dp = d_c(p);
    dpp = d_c(pp);
    du = d_c(ua);
    dv = d_c(va);
    dw = d_c(wa);

    Kokkos::parallel_for(
        "initialize", mdrange_policy3({l_s[0], l_s[1], l_s[2]}, {l_e[0], l_e[1], l_e[2]}), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            double sqd = pow(pow(dphi(0, i, j, k), 2) + pow(dphi(1, i, j, k), 2) + pow(dphi(2, i, j, k), 2), 0.5) + eps;

            divphix(i, j, k) = dphi(0, i, j, k) / sqd;
            divphiy(i, j, k) = dphi(1, i, j, k) / sqd;
            divphiz(i, j, k) = dphi(2, i, j, k) / sqd;

            drho(0,i, j, k) = dphi(0, i, j, k) * rho0 + (1.0 - dphi(0, i, j, k)) * rho1;
            drho(1,i, j, k) = dphi(1, i, j, k) * rho0 + (1.0 - dphi(1, i, j, k)) * rho1;
            drho(2,i, j, k) = dphi(2, i, j, k) * rho0 + (1.0 - dphi(2, i, j, k)) * rho1;
        });
    
    Kokkos::fence();
    pass(divphix);
    pass(divphiy);
    pass(divphiz);
    div = div_c(divphix,divphiy,divphiz);

    Kokkos::parallel_for(
        "update", team_policy(lz - 2 * ghost, Kokkos::AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int k = team_member.league_rank()+ghost;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, (lx - 2 * ghost) * (ly - 2 * ghost)), [&](const int &ij)
                {
                    const int i = ij % (lx - 2 * ghost) + ghost;
                    const int j = ij / (lx - 2 * ghost) + ghost;

                    ua(i, j, k) = 0.0;
                    va(i, j, k) = 0.0;
                    wa(i, j, k) = 0.0;

                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &um)
                        { um += f(ii, i, j, k) * e(ii, 0); },
                        ua(i, j, k));

                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &vm)
                        { vm += f(ii, i, j, k) * e(ii, 1); },
                        va(i, j, k));

                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &wm)
                        { wm += f(ii, i, j, k) * e(ii, 2); },
                        wa(i, j, k)); });
        });
    Kokkos::fence();



    Kokkos::parallel_for(
        "initialize", mdrange_policy3({l_s[0], l_s[1], l_s[2]}, {l_e[0], l_e[1], l_e[2]}), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            double sqd = pow(pow(dphi(0, i, j, k), 2) + pow(dphi(1, i, j, k), 2) + pow(dphi(2, i, j, k), 2), 0.5) + eps;

            double forsx = -3.0 / 2.0 * sigma * delta * div(i, j, k) * sqd * dphi(0, i, j, k);
            double forsy = -3.0 / 2.0 * sigma * delta * div(i, j, k) * sqd * dphi(1, i, j, k);
            double forsz = -3.0 / 2.0 * sigma * delta * div(i, j, k) * sqd * dphi(2, i, j, k);

            double forspx = -dp(0, i, j, k);
            double forspy = -dp(1, i, j, k);
            double forspz = -dp(2, i, j, k);

            double forsppx = dpp(0, i, j, k) * rho(i, j, k);
            double forsppy = dpp(1, i, j, k) * rho(i, j, k);
            double forsppz = dpp(2, i, j, k) * rho(i, j, k);

            double forsnx = nu(i, j, k) * (2.0 * du(0, i, j, k) * drho(0, i, j, k) + 
                                                (du(1, i, j, k) + dv(0, i, j, k)) * drho(1, i, j, k) + 
                                                (du(2, i, j, k) + dw(0, i, j, k)) * drho(2, i, j, k));
            double forsny = nu(i, j, k) * (2.0 * dv(1, i, j, k) * drho(1, i, j, k) + 
                                                (du(1, i, j, k) + dv(0, i, j, k)) * drho(0, i, j, k) + 
                                                (dv(2, i, j, k) + dw(1, i, j, k)) * drho(2, i, j, k));
            double forsnz = nu(i, j, k) * (2.0 * dw(2, i, j, k) * drho(2, i, j, k) + 
                                                (du(2, i, j, k) + dw(0, i, j, k)) * drho(0, i, j, k) + 
                                                (dw(1, i, j, k) + dv(2, i, j, k)) * drho(1, i, j, k));

            ua(i, j, k) = ua(i, j, k) / cs2 + (forspx + forsppx + forsnx + forsx) / 2.0 / rho(i, j, k);
            va(i, j, k) = va(i, j, k) / cs2 + (forspy + forsppy + forsny + forsy) / 2.0 / rho(i, j, k);
            wa(i, j, k) = wa(i, j, k) / cs2 + (forspz + forsppz + forsnz + forsz) / 2.0 / rho(i, j, k);
        });

    Kokkos::fence();
    double energytot = 0.0;
    double energy = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
  /*
    for (int i = ghost; i < l_e[0]; i++)
    {
        for (int j = ghost; j < l_e[1]; j++)
        {
            for (int k = ghost; k < l_e[2]; k++)
            {

                energy += (pow(ua(i, j, k), 2) + pow(va(i, j, k), 2) + pow(wa(i, j, k), 2)) / 2.0;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&energy, &energytot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

        if (comm.me == 0)
        {
                ofstream outfile;
                outfile.open("test.dat", ios::out | ios::app);
                outfile << std::fixed << std::setprecision(16) << energytot << endl;
                outfile.close();
        }
*/
    Kokkos::fence();
};

void LBM::MPIoutput(int n)
{
    // MPI_IO
    MPI_File fh;
    //MPIO_Request request;
    MPI_Status status;
    MPI_Offset offset = 0;

    MPI_Datatype FILETYPE, DATATYPE;
    // buffer
    int tp;
    float ttp;
    double fp;
    // min max
    double umin, umax, wmin, wmax, vmin, vmax, pmin, pmax,phimin,phimax;
    double uumin, uumax, wwmin, wwmax, vvmin, vvmax, ppmin, ppmax,pphimin,pphimax;
    // transfer
    double *uu, *vv, *ww, *pp, *xx, *yy, *zz, *phio, *ddivphix;
    // int start[3];
    uu = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    vv = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    ww = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    pp = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    phio = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    xx = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    yy = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    zz = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));

    for (int k = 0; k < l_l[2]; k++)
    {
        for (int j = 0; j < l_l[1]; j++)
        {
            for (int i = 0; i < l_l[0]; i++)
            {

                uu[i + j * l_l[0] + k * l_l[1] * l_l[0]] = ua(i + ghost, j + ghost, k + ghost);
                vv[i + j * l_l[0] + k * l_l[1] * l_l[0]] = va(i + ghost, j + ghost, k + ghost);
                ww[i + j * l_l[0] + k * l_l[1] * l_l[0]] = wa(i + ghost, j + ghost, k + ghost);
                pp[i + j * l_l[0] + k * l_l[1] * l_l[0]] = p(i + ghost, j + ghost, k + ghost);
                phio[i + j * l_l[0] + k * l_l[1] * l_l[0]] = phi(i + ghost, j + ghost, k + ghost);                
                xx[i + j * l_l[0] + k * l_l[1] * l_l[0]] = (double)(x_lo + i) / (glx - 1);
                yy[i + j * l_l[0] + k * l_l[1] * l_l[0]] = (double)(y_lo + j) / (gly - 1);
                zz[i + j * l_l[0] + k * l_l[1] * l_l[0]] = (double)(z_lo + k) / (glz - 1);
            }
        }
    }

        parallel_reduce(
            " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = ua(i,j,k);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(umax));
        Kokkos::fence();
        parallel_reduce(
            " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = va(i,j,k);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(vmax));
        Kokkos::fence();
        parallel_reduce(
            " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = wa(i,j,k);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(wmax));
        Kokkos::fence();
        parallel_reduce(
            " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = p(i,j,k);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(pmax));
        Kokkos::fence();

        parallel_reduce(
            " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = phi(i,j,k);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(phimax));
        Kokkos::fence();
        parallel_reduce(
            " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = ua(i,j,k);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(umin));
        Kokkos::fence();
        parallel_reduce(
            " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = va(i,j,k);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(vmin));
        Kokkos::fence();
        parallel_reduce(
            " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = wa(i,j,k);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(wmin));
        Kokkos::fence();
        parallel_reduce(
            " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = p(i,j,k);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(pmin));
        Kokkos::fence();
        parallel_reduce(
            " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = phi(i,j,k);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(phimin));
        Kokkos::fence();
        std::string str1 = "output" + std::to_string(n) + ".plt";
        const char *na = str1.c_str();
        std::string str2 = "#!TDV112";
        const char *version = str2.c_str();
        MPI_File_open(MPI_COMM_WORLD, na, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

        MPI_Reduce(&umin, &uumin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&umax, &uumax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        MPI_Reduce(&vmin, &vvmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&vmax, &vvmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        MPI_Reduce(&wmin, &wwmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&wmax, &wwmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        MPI_Reduce(&pmin, &ppmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&pmax, &ppmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&phimin, &pphimin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&phimax, &pphimax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

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
            tp = 7;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            tp = 120;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            tp = 0;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            tp = 121;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            tp = 0;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            tp = 122;
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
            tp = 119;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            tp = 0;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            tp = 112;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            tp = 0;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);

            // 20+15*4=80
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

            // 80 + 10 * 4 = 120

            // Strand id
            tp = -1;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            // SOLUTION TIME
            double nn = (double)n;
            fp = nn;
            MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
            tp = 0;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            // ZONE COLOR
            tp = -1;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            // ZONE TYPE
            tp = 0;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            // SPECIFY VAR LOCATION
            tp = 0;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            // ARE RAW LOCAL
            tp = 0;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            // NUMBER OF MISCELLANEOUS
            tp = 0;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            // ORDERED ZONE
            tp = glx;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            tp = gly;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            tp = glz;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            // AUXILIARY
            tp = 0;
            MPI_File_write(fh, &tp, 1, MPI_INT, &status);
            // 120 + 13 * 4 = 172
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
            // 172 + 12 * 4 = 220
            // MIN AND MAX VALUE FLOAT 64
            fp = 0.0;
            MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
            fp = 1.0;
            MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
            fp = 0.0;
            MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
            fp = 1.0;
            MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
            fp = 0.0;
            MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
            fp = 1.0;
            MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
            fp = uumin;
            MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
            fp = uumax;
            MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
            fp = vvmin;
            MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
            fp = vvmax;
            MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
            fp = wwmin;
            MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
            fp = wwmax;
            MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
            fp = pphimin;
            MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
            fp = pphimax;
            MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);

            // 220 + 14 * 8 = 332
        }

        offset = 332;

        int glolen[3] = {glx, gly, glz};
        //int iniarr[3] = {0, 0, 0};
        int localstart[3] = {x_lo, y_lo, z_lo};
        MPI_Type_create_subarray(dim, glolen, l_l, localstart, MPI_ORDER_FORTRAN, MPI_DOUBLE, &DATATYPE);

        MPI_Type_commit(&DATATYPE);

        MPI_Type_contiguous(7, DATATYPE, &FILETYPE);

        MPI_Type_commit(&FILETYPE);

        MPI_File_set_view(fh, offset, MPI_DOUBLE, FILETYPE, "native", MPI_INFO_NULL);

        MPI_File_write_all(fh, xx, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

        MPI_File_write_all(fh, yy, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

        MPI_File_write_all(fh, zz, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

        MPI_File_write_all(fh, uu, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

        MPI_File_write_all(fh, vv, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

        MPI_File_write_all(fh, ww, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

        MPI_File_write_all(fh, phio, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

        MPI_File_close(&fh);


        if (comm.me == 0)
        {

        
        printf("\n");
        printf("The result %d is writen\n", n);
        printf("\n");
        printf("============================\n");
        }

        free(uu);
        free(vv);
        free(ww);
        free(pp);
        free(xx);
        free(yy);
        free(zz);
        free(phio);

        MPI_Barrier(MPI_COMM_WORLD);
};

void LBM::Output(int n)
{
    std::ofstream outfile;
    std::string str = "output" + std::to_string(n) + std::to_string(comm.me);
    outfile << std::setiosflags(std::ios::fixed);
    outfile.open(str + ".dat", std::ios::out);

    outfile << "variables=x,y,z,f" << std::endl;
    outfile << "zone I=" << lx - 6 << ",J=" << ly - 6 << ",K=" << lz - 6 << std::endl;

    for (int k = 3; k < lz - 3; k++)
    {
        for (int j = 3; j < ly - 3; j++)
        {
            for (int i = 3; i < lx - 3; i++)
            {

                outfile << std::setprecision(8) << setiosflags(std::ios::left) << x_lo + i - 3 << " " << y_lo + j - 3 << " " << z_lo + k - 3 << " " << f(0, i, j, k) << std::endl;
            }
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

Kokkos::View<double****,Kokkos::CudaUVMSpace> LBM::d_c(Kokkos::View<double***,Kokkos::CudaUVMSpace> c)
{
    Kokkos::View<double ****, Kokkos::CudaUVMSpace> dc= Kokkos::View<double ****, Kokkos::CudaUVMSpace>("dc_", dim, lx, ly, lz);
    typedef Kokkos::TeamPolicy<> team_policy;
    typedef Kokkos::TeamPolicy<>::member_type member_type;



        Kokkos::parallel_for(
            "dc", team_policy(lz - 2 * ghost, Kokkos::AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
                const int k = team_member.league_rank() + ghost;

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team_member, (lx - 2 * ghost) * (ly - 2 * ghost)), [&](const int &ij)
                    {
                    const int i = ij % (lx-2*ghost)+ghost;
                    const int j = ij / (lx-2*ghost)+ghost;

                    dc(0, i, j, k) = 0.0;
                    dc(1, i, j, k) = 0.0;
                    dc(2, i, j, k) = 0.0;

                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &dc0_tem)
                        { dc0_tem += t(ii)* e(ii, 0) * (c(i + e(ii, 0), j + e(ii, 1), k + e(ii, 2)) - c(i - e(ii, 0), j - e(ii, 1), k - e(ii, 2)))/2.0 *3.0;},
                        dc(0, i, j, k));

                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &dc1_tem)
                        { dc1_tem += t(ii) * e(ii, 1) * (c(i + e(ii, 0), j + e(ii, 1), k + e(ii, 2)) - c(i - e(ii, 0), j - e(ii, 1), k - e(ii, 2))) / 2.0 *3.0; },
                        dc(1, i, j, k));

                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &dc2_tem)
                        { dc2_tem += t(ii) * e(ii, 2) * (c(i + e(ii, 0), j + e(ii, 1), k + e(ii, 2)) - c(i - e(ii, 0), j - e(ii, 1), k - e(ii, 2))) / 2.0 *3.0; },
                        dc(2, i, j, k)); });
            });

    Kokkos::fence();
    return dc;
};

Kokkos::View<double***,Kokkos::CudaUVMSpace> LBM::div_c(Kokkos::View<double***,Kokkos::CudaUVMSpace> cx,Kokkos::View<double***,Kokkos::CudaUVMSpace> cy,Kokkos::View<double***,Kokkos::CudaUVMSpace> cz)
{
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> divc= Kokkos::View<double ***, Kokkos::CudaUVMSpace>("divc", lx, ly, lz);
    typedef Kokkos::TeamPolicy<> team_policy;
    typedef Kokkos::TeamPolicy<>::member_type member_type;

    Kokkos::parallel_for(
        "divc", team_policy(lz-2*ghost, Kokkos::AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int k = team_member.league_rank()+ghost;

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, (lx -2*ghost)* (ly-2*ghost)), [&](const int &ij)
                {
                    const int i = ij % (lx-2*ghost)+ghost;
                    const int j = ij / (lx-2*ghost)+ghost;

                    double divx = 0.0;
                    double divy = 0.0;
                    double divz = 0.0;
                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &divc0_tem)
                        { divc0_tem += t(ii) * e(ii, 0) * (cx(i + e(ii, 0), j + e(ii, 1), k + e(ii, 2)) - cx(i - e(ii, 0), j - e(ii, 1), k - e(ii, 2))) / 2.0 *3.0; },
                        divx);

                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &divc1_tem)
                        { divc1_tem += t(ii) * e(ii, 1) * (cy( i + e(ii, 0), j + e(ii, 1), k + e(ii, 2)) - cy( i - e(ii, 0), j - e(ii, 1), k - e(ii, 2))) / 2.0 *3.0; },
                        divy);

                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &divc2_tem)
                        { divc2_tem += t(ii) * e(ii, 2) * (cz( i + e(ii, 0), j + e(ii, 1), k + e(ii, 2)) - cz( i - e(ii, 0), j - e(ii, 1), k - e(ii, 2))) / 2.0 *3.0; },
                        divz);

                    divc(i, j, k) = divx + divy + divz;
                });
        });

    Kokkos::fence();
    return divc;
};