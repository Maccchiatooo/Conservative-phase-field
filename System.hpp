#ifndef _SYSTEM_H_
#define _SYSTEM_H_
#include "lbm.hpp"
#include <cmath>
#include <iostream>

class System
{

public:
    System();
    void Monitor();

    int sx, sy, sz;
    int Time, inter;
    double miu, u0, R, rho0,rho1, Ma, tau0,tau1;
    double r_x, r_y, r_z;
    double taum;
    double sigma;
    double Re;
    double cs2, cs;
};
#endif