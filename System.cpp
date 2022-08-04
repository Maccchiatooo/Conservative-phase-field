#include "System.hpp"

#include "iostream"

using namespace std;
System::System()
{


    fstream input;
    input.open("input.in");
    input >> this->rho0 >> this->rho1;
    input >> this->taum >> this->tau0 >> this->tau1;
    input >> this->Time >> this->inter;
    input >> this->Oh;
    input >> this->sx >> this->sy;
    this->cs2 = 1.0 / 3.0;
    this->cs = sqrt(cs2);
    double r = 0.2 * this->sx;
    double mu0 = rho0 * tau0 * cs2;

    this->sigma = pow((mu0 / this->Oh), 2) / rho0 / r;
    this->Time = 100;//mu0 * r / this->sigma;
    this->inter = 100;//this->Time / 20;
}


void System::Monitor()
{

    std::cout << "3D conservative phase field" << std::endl
              << "rho0   =" << this->rho0 << std::endl
              << "rho1   =" << this->rho1 << std::endl
              << "Oh     =" << this->Oh << std::endl
              << "tau0    =" << this->tau0 << std::endl
              << "tau1    =" << this->tau1 << std::endl
              << "taum    =" << this->taum << std::endl
              << "Time   =" << this->Time << std::endl
              << "inter  =" << this->inter << std::endl
              << "============================" << std::endl;
};
