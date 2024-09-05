 /*icpc -O2 -std=c++11 -O3 -DNDEBUG -funroll-loops -ffast-math -fopenmp fftwpp/fftw++.cc -lfftw3 -lfftw3_omp t_i_d_se_fast_vg.cpp*/
#include <iostream>
#include <vector>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <stdio.h>
#include <string>
#include <cmath>
#include <omp.h>
#include <iomanip> //this is for setprecision
#include "EbDawson.hpp"
#include "bessj.h"
using namespace std;

#define CHUNK 10
#define pi 3.1415926535897932384626433

typedef complex<double> dcompx;
typedef complex<int> icompx;
typedef std::vector<double> dvec;
typedef std::vector<std::vector<double> > ddvec;
typedef std::vector<dcompx> dcopxvec;
typedef std::vector<std::vector<dcompx> > ddcopxvec;

constexpr dcompx I{0.0, 1.0};
constexpr dcompx zero{0.0,0.0};
constexpr dcompx one{1.0,0.0};
constexpr dcompx two{2.0,0.0};
constexpr dcompx three{3.0,0.0};

void getEnergy_dipolematrix(dvec &E, ddcopxvec &dipole);
double occupation_svea_sinusodal_pulse(dcopxvec &d_j, dvec &E, ddcopxvec &Intgral_simps, double w0, double A_0,double T, double eta);
double fft_fw(dvec &pulse, double w);
double sinusodal_pulse(double t, double w0,double T);
//the number of OpenMP threads 
int NTHREADS = omp_get_max_threads();

double xhbox = 1000.; // min must be 50
double e_up = 1.5;


double t_0 = -5000.0;
double t_m = -t_0;
double dt = 0.1;//(t_m - t_0)/N_time;
int N_time = int((t_m - t_0)/dt);


template <class T>
std::string strcpp(T value){
    ostringstream out;
    out << value;
    return out.str();
}
double K_Gaussian(double E, double dE){
    double norm_n = (1./(sqrt(pi)*abs(dE)));
    return norm_n*exp(-(E*E)/(dE*dE));
}

int main(){
    
    double w0 = 0.314;
    
    double beta = 1., beta2=1.0;//4.5;//1.8;
    double A0 = 0.5;
    double T = 2.0*41.341;
    
    
    dvec E;
    ddcopxvec Intgral_simps;
    getEnergy_dipolematrix(E, Intgral_simps);

    dcopxvec d_jkl,d_jk,d_j;
    double eta2 = 0.017;//0.479;
    double eta = 0.065;//0.479;
    double eta3 = 0.03;//0.479;
    
    d_jkl.resize(E.size());
    d_jk.resize(E.size());
    d_j.resize(E.size());
    omp_set_num_threads(NTHREADS);
    #pragma omp parallel for default(shared) schedule(static, CHUNK)
    for(int j = 0 ; j < E.size(); j++){
        dcompx sum_dip = 0.0,last_sum = 0.0,sum_d = 0.0;
        for(int k = 0; k < E.size(); k++){
                
                //one
                dcompx part1 =  -1.0/8.0 * A0*A0*pi*T*T*(Intgral_simps[k][0]*Intgral_simps[j][k])\
                        *exp(-(E[k]-E[0]-w0)*(E[k]-E[0]-w0)*T*T/4.0)*exp(-(E[j]-E[k]-w0)*(E[j]-E[k]-w0)*T*T/4.0);
                dcompx part1_2 =  -1.0/8.0 * A0*A0*pi*T*T*(Intgral_simps[k][0]*Intgral_simps[j][k])\
                        *exp(-(E[k]-E[0]-w0)*(E[k]-E[0]-w0)*T*T/4.0)*exp(-(E[j]-E[k]+w0)*(E[j]-E[k]+w0)*T*T/4.0);
                dcompx part1_3 =  -1.0/8.0 * A0*A0*pi*T*T*(Intgral_simps[k][0]*Intgral_simps[j][k])\
                        *exp(-(E[k]-E[0]+w0)*(E[k]-E[0]+w0)*T*T/4.0)*exp(-(E[j]-E[k]-w0)*(E[j]-E[k]-w0)*T*T/4.0);
                        
                
                dcompx part2_1 = -I/4.0 * A0*A0*sqrt(pi)*T*T*(Intgral_simps[k][0]*Intgral_simps[j][k])*DawsonByMcCabeCF( ( (E[j]-E[k]) - (E[k]-E[0]) ) *T/sqrt(8.0) )\
                        *exp(-(E[j]-E[0]-2.0*w0)*(E[j]-E[0]-2.0*w0)*T*T/8.0);
                dcompx part2_2 = -I/4.0 * A0*A0*sqrt(pi)*T*T*(Intgral_simps[k][0]*Intgral_simps[j][k])*DawsonByMcCabeCF( ( (E[j]-E[k]) - (E[k]-E[0]) + 2*w0 ) *T/sqrt(8.0) )\
                        *exp(-(E[j]-E[0])*(E[j]-E[0])*T*T/8.0);
                dcompx part2_3 = -I/4.0 * A0*A0*sqrt(pi)*T*T*(Intgral_simps[k][0]*Intgral_simps[j][k])*DawsonByMcCabeCF( ( (E[j]-E[k]) - (E[k]-E[0]) - 2*w0 ) *T/sqrt(8.0) )\
                        *exp(-(E[j]-E[0])*(E[j]-E[0])*T*T/8.0);
                        


                sum_dip +=  (part1 + part1_2 + part1_3 ) + (part2_1 + part2_2 + part2_3 );
                
        }
        d_jkl[j] =sum_dip;
    }
    
    //slowly-varying envelope
    for(int j = 0 ; j < E.size(); j++){
        dcompx sum_d = 0.0;
        for(int k = 0; k < E.size(); k++){
                
                dcompx part_1 = I/4.0 * A0*A0*sqrt(pi/2)*T*(Intgral_simps[k][0]*Intgral_simps[j][k])*real( 1.0/( (E[j]-E[k]) - (E[k]-E[0]) + I*eta) )\
                        *exp(-(E[j]-E[0]-2.0*w0)*(E[j]-E[0]-2.0*w0)*T*T/8.0);
                dcompx part_2 = I/4.0 * A0*A0*sqrt(pi/2)*T*(Intgral_simps[k][0]*Intgral_simps[j][k])*real( 1.0/( (E[j]-E[k]) - (E[k]-E[0]) + 2*w0 + I*eta) )\
                        *exp(-(E[j]-E[0])*(E[j]-E[0])*T*T/8.0);
                dcompx part_3 = I/4.0 * A0*A0*sqrt(pi/2)*T*(Intgral_simps[k][0]*Intgral_simps[j][k])*real( 1.0/( (E[j]-E[k]) - (E[k]-E[0]) - 2*w0 + I*eta) )\
                        *exp(-(E[j]-E[0])*(E[j]-E[0])*T*T/8.0);
                        


                sum_d +=  (part_1 + part_2 + part_3 );
                
        }
        d_j[j] = sum_d;
    }
    
    occupation_svea_sinusodal_pulse(d_j,E,Intgral_simps,w0,A0,T,eta);

    ofstream dout("dipole_sinu.dat");
    for(int j = 2 ; j < E.size(); j+=2)
        dout<<E[j]<<"\t"<<abs(d_jkl[j])*abs(d_jkl[j])<<"\t"<<abs(d_j[j])*abs(d_j[j])<<endl;
    
    
    ofstream pout("spectrumANA_sinu.dat");
    double P_E_sum_emit,p_sum_obsr,p_sum,p_sum_inter, dE;
    for(double E_spectrum = 0.0; E_spectrum < e_up; E_spectrum += 0.0001){
        dE = 0.005;
        P_E_sum_emit = 0.0,p_sum_obsr=0.0,p_sum=0.0,p_sum_inter = 0.0;
        for(int j = 2 ; j < E.size(); j+=2){
            if(E[j] > 0.0)
                p_sum += abs(d_jkl[j])*abs(d_jkl[j]) *K_Gaussian(E_spectrum - E[j], dE);
        }
        for(int j = 2 ; j < E.size(); j+=2){
            if(E[j] > 0.0)
                p_sum_obsr += abs(d_j[j])*abs(d_j[j]) *K_Gaussian(E_spectrum - E[j], dE);
        }
        pout<<E_spectrum<<"\t"<<p_sum<<"\t"<<p_sum_obsr <<'\n';
    }
    
    
    return 0;
}


double occupation_svea_sinusodal_pulse(dcopxvec &d_j, dvec &E, ddcopxvec &Intgral_simps, double w0, double A_0, double T, double eta){
    
    dvec pulse(N_time,0.0);
    for(int i =0; i < N_time; i++){
        double t = t_0 + i*dt;
        pulse[i] = sinusodal_pulse(t,w0,T);
    }
    
    cout << "pulse done\n";
    for(int j = 0 ; j < E.size(); j++){
        double fft_ft = fft_fw(pulse,E[j]-E[0]);
        dcompx sum_d = 0.0;
        for(int k = 0; k < E.size(); k++){
                
                dcompx part_1 = I/4.0 * A_0*A_0*(Intgral_simps[k][0]*Intgral_simps[j][k])\
                            *real( 1.0/( (E[j]-E[k]) - (E[k]-E[0]) + 2*w0 + I*eta) + 1.0/( (E[j]-E[k]) - (E[k]-E[0]) - 2*w0 + I*eta) )*fft_ft;
                sum_d +=  (part_1);
                
        }
        d_j[j] = sum_d;
    }
    
}



double fft_fw(dvec &pulse, double w){
    
    dcompx sum = 0.0;
    for(int i =0; i < N_time; i++){
        double t = t_0 + i*dt;
        double A_t_squar = pulse[i]*pulse[i];
        if(i%2==0 && i!=0 && i!=(N_time - 1)){
            sum += 2.* A_t_squar*exp(I*w*t);
        }
        else if(i%2!=0 && i!=0 && i!=(N_time - 1)){
            sum += 4.* A_t_squar*exp(I*w*t);
        }
        else{
            sum += A_t_squar*exp(I*w*t);
        }
    }
    return (dt/3.)*sum.real()/(2.0*pi);
    
}

double sinusodal_pulse(double t, double w0,double T){
    
    double tau = 4*T;
    double a = 1.0;
    int Nacc = 15;
    double phase = 0.0;
    double sumj = 0.0;
    for(int k = -Nacc; k <=Nacc; k++){
        if(k < 0)
            sumj += pow(-1,abs(k))*bessj(abs(k),a)*exp(-(t - k*tau)*(t - k*tau)/(T*T))*cos(w0*(t-k*tau)-k*phase);
        else
            sumj += bessj(abs(k),a)*exp(-(t - k*tau)*(t - k*tau)/(T*T))*cos(w0*(t-k*tau)-k*phase);
    }
    return sumj;
}





void getEnergy_dipolematrix(dvec &E, ddcopxvec &dipole){
    
    std::ifstream Ein("eigenvalue_"+strcpp(xhbox)+"_"+strcpp(e_up)+".dat");
    if(Ein.is_open()){
        double en;
        int jj = 0;
        while(Ein>>en){
            E.push_back(en);
            jj++;
        }
    }
    else{cout <<"Can't find the file!\n";}
    
    dipole.resize(E.size());
    for(int i = 0; i < E.size(); i++)dipole[i].resize(E.size());
    
    
    std::ifstream din("Integraldipole_"+strcpp(xhbox)+"_"+strcpp(e_up)+".dat");
    if(din.is_open()){
        double dip;
        for(int i = 0; i < E.size(); i++){
            for(int j = 0; j < E.size(); j++){
                din >> dip;
                dipole[i][j] = (dip);
            }
        }
    }
    else{cout <<"Can't find the file!\n";}
    
    //make it just positive or negative as it should be    
    for(int i = 1; i < E.size(); i+=4){
        for(int k = 0; k < E.size(); k++){
            if(k%2==0){
                dipole[i][k] = -dipole[i][k];
                dipole[k][i] = -dipole[k][i];
            }
        }
    }
    for(int i = 3; i < E.size(); i+=4){
        for(int k = 0; k < E.size(); k++){
            if(k%2==0){
                dipole[i][k] = dipole[i][k];
                dipole[k][i] = dipole[k][i];
            }
        }
    }

    for(int i = 0; i < E.size(); i+=4){
        for(int k = 0; k < E.size(); k++){
            if(k%2!=0){
                dipole[i][k] = -dipole[i][k];
                dipole[k][i] = -dipole[k][i];
            }
        }
    }
    for(int i = 2; i < E.size(); i+=4){
        for(int k = 0; k < E.size(); k++){
            if(k%2!=0){
                dipole[i][k] = dipole[i][k];
                dipole[k][i] = dipole[k][i];
            }
        }
    }

    for(int i = 0; i < E.size(); i++){
        for(int j = 0; j < E.size(); j++){
            dipole[i][j] = -I*dipole[i][j];
        }
    }
}

