#include "lbm.hpp"
using namespace Kokkos;

void LBM::Initialize()
{

    setup_MPI();

    f = buffer_f("f", q, lx, ly);
    f_tem = buffer_f("f_tem", q, lx, ly);
    fb = buffer_f("fb", q, lx, ly);
    g = buffer_f("g", q, lx, ly);
    g_tem = buffer_f("g_tem", q, lx, ly);
    gb = buffer_f("gb", q, lx, ly);

    ua = buffer_u("u", lx, ly);
    va =  buffer_u("v", lx, ly);
    rho =  buffer_u("rho", lx, ly);
    cp =  buffer_u("cp", lx, ly);
    p =  buffer_u("p", lx, ly);
    pp =  buffer_u("pp", lx, ly);
    phi = buffer_u("phi", lx, ly);
    tau = buffer_u("tau", lx, ly);
    nu =  buffer_u("nu", lx, ly);

    drho = buffer_f("drho", dim,lx, ly);
    dphi = buffer_f("dphi", dim,lx, ly);
    dp = buffer_f("dp", dim,lx, ly);
    dpp = buffer_f("dpp", dim,lx, ly);
    du =buffer_f("du", dim,lx, ly);
    dv =buffer_f("dv", dim,lx, ly);
    divphix = buffer_div("divphix", lx, ly);
    divphiy = buffer_div("divphiy", lx, ly);
    div = buffer_div("div", lx, ly);

    e = Kokkos::View<int **, Kokkos::CudaSpace>("e", q, dim);
    t = Kokkos::View<double *, Kokkos::CudaSpace>("t", q);
    usr = Kokkos::View<int **, Kokkos::CudaSpace>("usr", lx, ly);
    ran = Kokkos::View<int **, Kokkos::CudaSpace>("ran", lx, ly);
    bb = Kokkos::View<int *, Kokkos::CudaSpace>("b", q);

    View<int *, CudaSpace>::HostMirror bb_mirror=Kokkos::create_mirror_view(Kokkos::HostSpace(), bb);
    View<double *, CudaSpace>::HostMirror t_mirror=Kokkos::create_mirror_view(Kokkos::HostSpace(),t);
    View<int **, CudaSpace>::HostMirror e_mirror=Kokkos::create_mirror_view(Kokkos::HostSpace(),e);


    // weight and discrete velocity
    t_mirror(0) = 4.0 / 9.0;
    t_mirror(1) = 1.0 / 9.0;
    t_mirror(2) = 1.0 / 9.0;
    t_mirror(3) = 1.0 / 9.0;
    t_mirror(4) = 1.0 / 9.0;
    t_mirror(5) = 1.0 / 36.0;
    t_mirror(6) = 1.0 / 36.0;
    t_mirror(7) = 1.0 / 36.0;
    t_mirror(8) = 1.0 / 36.0;

    bb_mirror(0) = 0;
    bb_mirror(1) = 3;
    bb_mirror(3) = 1;
    bb_mirror(2) = 4;
    bb_mirror(4) = 2;
    bb_mirror(5) = 7;
    bb_mirror(7) = 5;
    bb_mirror(6) = 8;
    bb_mirror(8) = 6;

    e_mirror(0, 0) = 0;
    e_mirror(1, 0) = 1;
    e_mirror(2, 0) = 0;
    e_mirror(3, 0) = -1;
    e_mirror(4, 0) = 0;
    e_mirror(5, 0) = 1;
    e_mirror(6, 0) = -1;
    e_mirror(7, 0) = -1;
    e_mirror(8, 0) = 1;

    e_mirror(0, 1) = 0;
    e_mirror(1, 1) = 0;
    e_mirror(2, 1) = 1;
    e_mirror(3, 1) = 0;
    e_mirror(4, 1) = -1;
    e_mirror(5, 1) = 1;
    e_mirror(6, 1) = 1;
    e_mirror(7, 1) = -1;
    e_mirror(8, 1) = -1;

    Kokkos::deep_copy(t, t_mirror);
    Kokkos::deep_copy(e, e_mirror);
    Kokkos::deep_copy(bb, bb_mirror);

    parallel_for(
        "init", mdrange_policy2({0, 0}, {lx, ly}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            int global_x = x_lo + i - ghost;
            int global_y = y_lo + j - ghost;

            double dist = 2.0 * (sqrt(pow((global_x - 0.5 * glx), 2) + pow(global_y - 0.5*glx, 2)) - 0.25 * glx) / delta;


            ua(i, j) = 0.0;
            va(i, j) = 0.0;
            p(i, j) = 0.0;
            pp(i, j) = 0.0;

            phi(i, j) = (0.5 - 0.5 * tanh(dist));

            rho(i, j) = 1.0;
            tau(i, j) = tau0;
        });
    fence();

    Kokkos::parallel_for(
        "update", team_policy(ly, Kokkos::AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int j = team_member.league_rank();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, 0, lx), [&](const int &i)
                {
                                dphi(0, i, j) = 0.0;
                                dphi(1, i, j) = 0.0;
                                Kokkos::parallel_reduce(
                                    Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &dphi0)
                                    { dphi0 += g(ii, i, j) * (e(ii, 0) - ua(i, j)); },
                                    dphi(0, i, j));

                                Kokkos::parallel_reduce(
                                    Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &dphi1)
                                    { dphi1 += g(ii, i, j) * (e(ii, 1) - va(i, j)); },
                                    dphi(1, i, j)); }); });

    parallel_for(
        "cp_init", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            double sqd = sqrt(pow(dphi(0, i, j), 2) + pow(dphi(1, i, j), 2)) + 0.000001;

            divphix(i, j) = -dphi(0, i, j) / sqd;
            divphiy(i, j) = -dphi(1, i, j) / sqd;

        });
    fence();

    parallel_for(
        "initf", mdrange_policy3({0, 0, 0}, {q, lx, ly}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            double sqd = sqrt(pow(dphi(0, i, j), 2) + pow(dphi(1, i, j), 2)) + eps;

            double gamma = t(ii) * (1.0 + 3.0 * (e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j)) +
                                    4.5 * (pow((e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j)), 2)) -
                                    1.5 * (pow(ua(i, j), 2) + pow(va(i, j), 2)));


            double forphi = gamma * ((e(ii, 0) - ua(i, j)) * 4.0 * phi(i, j) * (1.0 - phi(i, j)) / delta * divphix(i, j) +
                                     (e(ii, 1) - va(i, j)) * 4.0 * phi(i, j) * (1.0 - phi(i, j)) / delta * divphiy(i, j));

            double f0 = 3.0 * pow(sqd, 2) * delta * sigma / 2.0 * (pow(e(ii, 0), 2) + pow(e(ii, 1), 2) - cs2 - pow(divphix(i, j) * e(ii, 0) + divphiy(i, j) * e(ii, 1), 2));

            f(ii, i, j) = t(ii) * pp(i, j) + (gamma - t(ii)) * cs2 +t(ii)/2.0/cs2*f0;
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


            double forphi = gamma * ((e(ii, 0) - ua(i, j)) * 4.0 * phi(i, j) * (1.0 - phi(i, j)) / delta * divphix(i, j) +
                                     (e(ii, 1) - va(i, j)) * 4.0 * phi(i, j) * (1.0 - phi(i, j)) / delta * divphiy(i, j));
            double f0 = 3.0 * pow(sqd, 2) * delta * sigma / 2.0 * (pow(e(ii, 0), 2) + pow(e(ii, 1), 2) - cs2 - pow(divphix(i, j) * e(ii, 0) + divphiy(i, j) * e(ii, 1), 2));

            double feq = t(ii) * pp(i, j) + (gamma - t(ii)) * cs2+t(ii)/2.0/cs2*f0;
            double geq = gamma * phi(i, j) - 0.50 * forphi;

            f(ii, i, j) = f(ii, i, j) - (f(ii, i, j) - feq) / (tau(i, j) + 0.5);
            g(ii, i, j) = g(ii, i, j) - (g(ii, i, j) - geq) / (taum + 0.5) + forphi;
        });
    fence();
};

void LBM::Streaming()
{
    passf(f);
    passf(g);

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

    parallel_for(
        "update", team_policy(ly , AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int j = team_member.league_rank();
            parallel_for(
                TeamThreadRange(team_member, 0, lx ), [&](const int &i)
                {
                    phi(i, j) = 0.0;
                    pp(i, j) = 0.0;
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

                    parallel_reduce(
                        ThreadVectorRange(team_member, q), [&](const int &ii, double &phi_tem)
                        { phi_tem += g(ii, i, j); },
                        phi(i, j));

                    parallel_reduce(
                        ThreadVectorRange(team_member, q), [&](const int &ii, double &pp_tem)
                        { pp_tem += f(ii, i, j); },
                        pp(i, j)); });
                });
    fence();

    Kokkos::parallel_for(
        "update", team_policy(ly, Kokkos::AUTO), KOKKOS_CLASS_LAMBDA(const member_type &team_member) {
            const int j = team_member.league_rank();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, 0, lx), [&](const int &i)
                {
                                dphi(0, i, j) = 0.0;
                                dphi(1, i, j) = 0.0;
                                Kokkos::parallel_reduce(
                                    Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &dphi0)
                                    { dphi0 += g(ii, i, j) * (e(ii, 0) - ua(i, j)); },
                                    dphi(0, i, j));

                                Kokkos::parallel_reduce(
                                    Kokkos::ThreadVectorRange(team_member, q), [&](const int &ii, double &dphi1)
                                    { dphi1 += g(ii, i, j) * (e(ii, 1) - va(i, j)); },
                                    dphi(1, i, j)); }); });

    parallel_for(
        "cp_init", mdrange_policy2({ghost, ghost}, {lx - ghost, ly - ghost}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            double sqd = sqrt(pow(dphi(0, i, j), 2) + pow(dphi(1, i, j), 2)) + 0.000001;

            divphix(i, j) = -dphi(0, i, j) / sqd;
            divphiy(i, j) = -dphi(1, i, j) / sqd;

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


    buffer_u::HostMirror ua_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ua);
    buffer_u::HostMirror va_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), va);

    buffer_u::HostMirror p_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), p);
    buffer_u::HostMirror phi_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), phi);

    for (int j = 0; j < (ly - 2 * ghost); j++)
    {
        for (int i = 0; i < (lx - 2 * ghost); i++)
        {

            uu[i + j * (lx - 2 * ghost)] = ua_mirror(i + ghost, j + ghost);
            vv[i + j * (lx - 2 * ghost)] = va_mirror(i + ghost, j + ghost);
            rr[i + j * (lx - 2 * ghost)] = phi_mirror(i + ghost, j + ghost);
            xx[i + j * (lx - 2 * ghost)] = 1.0 * (double)(x_lo + i) / (glx - 1);
            yy[i + j * (lx - 2 * ghost)] = 1.0 * (double)(y_lo + j) / (gly - 1);
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
         double my_value =phi(i,j);
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
         double my_value = phi(i,j);
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

    if (me == 0)
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

        if (me == 0)
    {
        printf("\n");
        printf("The result %d is writen\n", n);
        printf("\n");
        printf("============================\n");
    }

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
    std::string str = "output" + std::to_string(n) + std::to_string(me);
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
    if (me == 0)
    {
        printf("\n");
        printf("The result %d is writen\n", n);
        printf("\n");
        printf("============================\n");
    }
};
