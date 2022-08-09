#include "System.hpp"

#include "iostream"

using namespace std;
System::System()
{

    fstream input;
    input.open("input.in");
    input >> this->rho0 >> this->rho1 >>this->Re;
    input >> this->taum >> this->tau0 >> this->tau1;
    input >> this->Time >> this->inter;
    input >> this->u0;
    input >> this->sx >> this->sy >> this->sz;

    this->cs2 = 1.0 / 3.0;
    this->cs = sqrt(cs2);
    this->Ma = this->u0 / this->cs;
    this->tau0=this->u0*this->sx/Re/cs2;
    this->tau1=this->tau0;
    this->Time=100;//10*this->sx/this->u0/2/3.1415926;
    this->inter=100;//this->Time/10;
}


void System::Monitor()
{

    std::cout << "3D conservative phase field" << std::endl
              << "rho0   =" << this->rho0 << std::endl
              << "rho1   =" << this->rho1 << std::endl
              << "Ma     =" << this->Ma << std::endl
              << "tau0    =" << this->tau0 << std::endl
              << "tau1    =" << this->tau1 << std::endl
              << "taum    =" << this->taum << std::endl
              << "Time   =" << this->Time << std::endl
              << "inter  =" << this->inter << std::endl
              << "============================" << std::endl;
};
