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
#include "fftwpp/fftw++.h"
using namespace std;
using namespace utils;
using namespace fftwpp;



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

class TDSE{
    public:
        TDSE();
        ~TDSE();

        //functions
        void propagator_method(ddcopxvec &H, dcopxvec &C, int N_E);
        void Hamiltonian(ddcopxvec &H, ddvec &Intgral_simps, dvec &E, double Tc, int t_index, int N_E);
        void Hamiltonian_VG(ddcopxvec &H, ddvec &Intgral_simps, Complex *pulse, dvec &E, double Tc, int t_index, int N_E);
        void Numerical5p_differentiation(ddvec &phi, ddvec &dphi, int N_E);
        void SimpsonRule_overAlltimes(ddvec &phi, ddvec &Intgral_simps, int N_E);
        void SimpsonRule_overAlltimes_VG(ddvec &phi, ddvec &dphi, ddvec &Intgral_simps, int N_E);
        void MV_MultiTS(ddcopxvec &H, ddcopxvec &r, int size, int l);
        void Creat_Wavefunction(dcopxvec &Psi, ddvec &phi, dcopxvec &C, int N_E);
        string string_precision(int value, int n_digits);
        template <class T>
        string strcpp(T value);
        double Normalization(dcopxvec &psi, int);
        bool Kdelta(int i, int j){return(i==j);}
        int factor(int i){if(i==0){return (1);}else{return (i*factor(i-1));}};
        double t_step(int);
        double E_f(int t_index, double Tc);
        double A_f(int t_index, double Tc);
        double phase_function(dcopxvec &C,int indx);
        void tdse();

        //function for getting fourier transform
        double phase_fourierFunction(double w, double T, double A);
        void Creating_Boundpulse(Complex *pulse,double F_0,double T,double w0, double A);
        void Creating_FElpulse(Complex *pulse,double F_0,double tau,double T,double w0, double A);
        void pulse_generating(Complex *pulse, double A);

        //functions of numerov's method for solving time-independent SE
        double Potential(char, int);
        void Setk2(double *k2, double E, int N, char lr, int i_save);
        void Numerov(double *psi, double *k2, int N, int i_save);
        double diff_slope(double *psi_r, double *psi_l, double *k2_r, double *k2_l, double e_tr, int N_L, int N_R, int i_save);
        void Numerov_Solver(ddvec &wavfun, dvec &eignVal);
        void time_independent_wavefunction(ddvec &wavfun, dvec &eignVal, int N_E);
        void Normalizer(double *psi,int N);
        double x(char lr, int index);
        void id_tise();

        //vectors and Matricse
        ddvec phi;
        dvec E;
        ddcopxvec H;
        dcopxvec C;
        
        int N_graid = 160000;
        double xhbox = 1500.; // min must be 50
        double x_min = -xhbox;
        double x_max = xhbox;
        double dx = (x_max - x_min)/(N_graid-1);

        //set trial energy
        double e_tr = -0.03;
        double de = 0.000001;
        double e_up = 4.5;
        int N_E;

        //constant
        const double m = 1.;
        const double h_bar = 1.;
        const double c = 0.000000001; // second point of the wavefunctions

        //int N_time = 40000;
        double t_0 = -12000;
        double t_m = -t_0;
        double dt = 0.1;//(t_m - t_0)/N_time;
        int N_time = int((t_m - t_0)/dt);
        
        // fourier
        double dw = 2.0*pi/(N_time*dt);
        double w_0 = -0.5*N_time*dw;// w=[-n/2,n/2]*dw
        
        //pulse
        double Tc = 0.0;
        
        //the number of OpenMP threads 
        int NTHREADS = omp_get_max_threads();
};

TDSE::TDSE(){
}

TDSE::~TDSE(){
}


// phase of the frequency space function
double TDSE::phase_fourierFunction(double w, double T, double A){

    double tau = 4*T;
    double ampl = 0.0;
    return ampl*sin(w*tau + 0.);
}

//Vector potential
void TDSE::pulse_generating(Complex *pulse, double A){

    double k = pow(10.0,10.0/4.0)*0.1;
    double F_0 = sqrt(k/3.51);
    double w0 = 0.314;//or 16eV

    double T = 2.0*41.341;
    double A_0 = A*0.1;//0.318471338;//(F_0/w0);

    //for(int i = 0; i < N_time; i++){
    //    double t = t_0 + i*dt;
     //   pulse[i] = A_0*exp(-(t*t/(T*T)))*cos(w0*t); 
    //}
    
    //bound pulse
    Creating_Boundpulse(pulse,A_0,T,w0,A);
    //FEL pulse
    double tau=0.5*41.341;// a parameter in FEL pulse
    // tau=T this means furier limited pulse in FEL pulses
    //Creating_FElpulse(pulse,A_0,tau,T,w0,A);
}

void TDSE::tdse(){

    
    Complex *pulse=new Complex[N_time];
    ifstream fin("A_ini.dat");
    fin >> Tc;

    pulse_generating(pulse,Tc);
    int i_save = 0;
    for(int i = 0; i < N_time; i++){
        if(pulse[i].real() > 5.0e-5){
            i_save = i;
            break;
        }
    }
    t_0 = t_0 + i_save*dt;
    N_time = int( (-2.0*t_0)/dt  );
    t_0 = this->t_0;
    N_time = this->N_time;
    
    dw = 2.0*pi/(N_time*dt);
    w_0 = -0.5*N_time*dw;// w=[-n/2,n/2]*dw
    dw = this->dw;
    w_0 = this->w_0;

    cout << "new N_time = " << N_time <<" &  New t_min = "<<t_0 << endl;
    
    cout << "Step 1: finding eigenvalues and wavefunctions for Time-independent SE(numerove's Method) ....\n\n";
    ifstream Iin("../Integraldipole_"+strcpp(xhbox)+"_"+strcpp(e_up)+ ".dat", ios::in);
    ifstream fein("../eigenvalue_"+strcpp(xhbox)+"_"+strcpp(e_up)+ ".dat");
    if(!fein.is_open()){
        Numerov_Solver(phi,E);
        //save eigenvalues
        ofstream eout("eigenvalue_"+strcpp(xhbox)+"_"+strcpp(e_up)+ ".dat");
        for(int i = 0; i < N_E; i++)
            eout << E[i] << endl;
        eout.close();
    }
    else{
        // read a file
        double Eign;
        int jj = 0;
        while(fein >> Eign){
            E.push_back(Eign);
            jj++;
        }
        N_E = jj;
        
        if(!Iin.is_open()){
            time_independent_wavefunction(phi,E,N_E); // when we already have eigenvalues by this function we can find corresponding wavefunction
        }
    }

    cout <<"Number of EigenValues = " << N_E << endl;

    //exit(1);
    E.resize(N_E);
    phi.resize(N_graid);
    for (int i = 0; i < N_graid; ++i)
        phi[i].resize(N_E);

    ddvec Intgral_simps;
    Intgral_simps.resize(N_E);
    for(int i = 0; i < N_E; ++i)
        Intgral_simps[i].resize(N_E);


    dcopxvec Psi;
    dvec occpy;
    dvec phaseC;
    double norm;

    cout << "Step 2: calculating Integral, the dipole part(simpson rule) ....\n\n";

    if(!Iin.is_open()){
        
        ddvec dphi;
        dphi.resize(N_graid);
        for(int i = 0; i < N_graid; ++i)
            dphi[i].resize(N_E);
    
        //Integral over all EigenValues
        Numerical5p_differentiation(phi,dphi,N_E);
        cout << "here!"<<endl;
        SimpsonRule_overAlltimes_VG(phi,dphi,Intgral_simps,N_E);
        //SimpsonRule_overAlltimes(phi,Intgral_simps,N_E);
        
        dphi.clear(); //clear content
        dphi.resize(0); //resize it to 0
        dphi.shrink_to_fit(); 
    }
    else{
        // read a file
        double Int;
        for(int i = 0; i < N_E; i++){
            for(int j = 0; j < N_E; j++){
                Iin >> Int;
                Intgral_simps[i][j] = Int;
            }
        }
    }

    phi.clear(); //clear content
    phi.resize(0); //resize it to 0
    phi.shrink_to_fit(); 


    cout << "Step 3: plotting the given pulse ....\n\n";

    ofstream plsout("givenpulse_"+strcpp(Tc)+".dat");
    pulse_generating(pulse,Tc);
    
    for(int t = 0; t < N_time; t++){
        plsout << t_step(t) << "\t" << pulse[t].real()<< "\t" << atan2(pulse[t].imag(),pulse[t].real())<<endl;
    }    
    plsout.close();
    cout << "Step 4: starting main loop over all time ....\n\n";
    //ofstream fout2("psit5.dat");
    C.resize(N_E);
    occpy.resize(N_E);
    phaseC.resize(N_E);
    //initialization C(t)---- supose the particle in t = 0 is in ground state

    for(int e = 0; e < N_E; e++)C[e] = zero;
    C[0] = one;

    ofstream oout("groundstatoccpy_"+strcpp(Tc)+".dat");
    ofstream phout("phase_"+strcpp(Tc)+".dat");
   
    //time loop -- main loop
    for(int t = 0; t < N_time; t++){

        oout << t_step(t) << "\t" << pow(abs(C[0]),2)<< "\t" << phase_function(C,0)<< endl;
        double sum=0.;for(int j=0;j<N_E;j++)sum+=abs(C[j])*abs(C[j]);

        H.resize(N_E);for(int i=0;i<N_E;i++)H[i].resize(N_E);
        Hamiltonian_VG(H, Intgral_simps, pulse, E, Tc, t, N_E);
        //propagator
        propagator_method(H,C,N_E);
        
        H.clear();
        H.shrink_to_fit();
        if(t == N_time-5){
            for(int i = 0; i < N_E; i++){
                occpy[i] = abs(C[i])*abs(C[i]);
                phaseC[i] = phase_function(C,i);
            }
        }

        //phase of population
        phout<<t_step(t)<<"\t"<<phase_function(C,0)<<"\t"<<phase_function(C,1)<<"\t"<<phase_function(C,2)<<"\t"<<phase_function(C,3)<<"\t"<<phase_function(C,4)<<endl;
        
        if(t%1000 == 0)
            cout << "time = "<< t << "     "<< sum << endl;
    }
    oout.close();

    
    ofstream loute("lasttime_even_"+strcpp(Tc)+".dat");
    for(int i = 0; i < N_E; i+=2) loute<<E[i]<<"\t"<<occpy[i]<<"\t"<<2.0*phaseC[i]<<endl; loute.close();
    ofstream louto("lasttime_odd_"+strcpp(Tc)+".dat");
    for(int i = 1; i < N_E; i+=2) louto<<E[i]<<"\t"<<occpy[i]<<"\t"<<2.0*phaseC[i]<<endl; louto.close();
    
    
    ofstream lasttime("lasttime_"+strcpp(Tc)+".dat");
    for(int i = 1; i < N_E; i++) lasttime<<E[i]<<"\t"<<occpy[i]<<"\t"<<2.0*phaseC[i]<<endl; lasttime.close();
    
    
    ofstream pout("spectrum_"+strcpp(Tc)+".dat");
    auto K_Gaussian = [&](double E, double dE)->double{
        double norm_n = (1./(sqrt(pi)*abs(dE)));
        return norm_n*exp(-(E*E)/(dE*dE));
    };
    
    double P_E_sum,p_even_sum,p_odd_sum, dE;
    for(double E_spectrum = 0.0; E_spectrum < e_up; E_spectrum += 0.0001){
        dE = 0.005;
        P_E_sum = 0.0,p_even_sum=0.0,p_odd_sum=0.0;
        for(int j = 0; j < N_E; j++){
            if(E[j] > 0.0)
                P_E_sum += occpy[j] *K_Gaussian(E_spectrum - E[j], dE);
        }
        for(int j = 0; j < N_E; j+=2){
            if(E[j] > 0.0)
                p_even_sum += occpy[j] *K_Gaussian(E_spectrum - E[j], dE);
        }
        for(int j = 1; j < N_E; j+=2){
            if(E[j] > 0.0)
                p_odd_sum += occpy[j] *K_Gaussian(E_spectrum - E[j], dE);
        }
        pout<<E_spectrum<<"\t"<<P_E_sum<<"\t"<<p_even_sum<<"\t"<<p_odd_sum<<'\n';
    }
    pout.close();

    C.clear();
    C.shrink_to_fit();
    occpy.clear();
    occpy.shrink_to_fit();
    pout.close();
    Intgral_simps.clear();
    Intgral_simps.shrink_to_fit();
}

void TDSE::Creating_Boundpulse(Complex *pulse,double F_0,double T,double w0, double A){
    
    fftw::maxthreads=get_max_threads();
    
    fft1d Forward(N_time,-1);
    Complex *f=ComplexAlign(N_time);
    for(unsigned int i=0; i < N_time; i++){
        double w = w_0 + i*dw;
        Complex g = exp(-Complex(0.0,1.0)*phase_fourierFunction(w,T,A))*exp(-(w-w0)*(w-w0)*(T*T/(4.0)));
        f[i] = F_0*(T/sqrt(4.0*M_PI))*g;
    }
    Forward.fft(f);
    
    for(unsigned int i=0; i < N_time; i+=1){
        int fftshift = (i+N_time/2)%N_time;
        if(i%2==0){
            pulse[i] = f[fftshift]*dw;
        }
        else{
            pulse[i] = -f[fftshift]*dw;
        }
    }
    deleteAlign(f);
}

void TDSE::Creating_FElpulse(Complex *pulse,double F_0,double tau,double T,double w0, double A){
    
    fftw::maxthreads=get_max_threads();
    //pulse
    double Tp = tau*T*sqrt(2.0/(T*T+tau*tau));
    
    Complex *f=ComplexAlign(N_time);
    
    fft1d Forward(N_time,-1);
    fft1d Backward(N_time,1);
    for(unsigned int i=0; i < N_time; i++){
        double tp = t_0 + i*dt;
        f[i] = exp(-log(2.0)*tp*tp/(tau*tau))*cos(w0*tp);
    }
    Forward.fft(f);
    for(unsigned int i=0; i < N_time; i++){
        double w = w_0 + i*dw;
        f[i] = exp(Complex(0.0,1.0)*phase_fourierFunction(w-w_0,Tp,A))*f[i];
    }
    Backward.fftNormalized(f);
    for(unsigned int i=0; i < N_time; i+=1){
        double t = t_0 + i*dt;
        pulse[i] = F_0*exp(-log(2.0)*t*t/(T*T))*f[i];
    }
    deleteAlign(f);
}


double TDSE::phase_function(dcopxvec &C,int indx){
    double x = real(C[indx]);
    double y = imag(C[indx]);
    double phase_;
    if(x > 0){
        phase_ =  atan(y/x);
    }
    else if(x < 0){
        if(y >= 0){
            phase_ =  atan(y/x) + pi;
        }
        else{
            phase_ =  atan(y/x) - pi;
        }
    }
    
    return atan2(y,x);
    
}


//supervison suggested method
void TDSE::propagator_method(ddcopxvec &H, dcopxvec &C,int N_E){

    int N_L = 10;
    ddcopxvec r;
    r.resize(N_E);
    for (int i = 0; i < N_E; ++i)
        r[i].resize(N_L);

    for(int i = 0; i < N_E; i++)
        r[i][0] = C[i];

    for(int l = 1; l < N_L; l++){
        MV_MultiTS(H,r,N_E,l);
    }

    for(int i = 0; i < N_E; i++)
        C[i] = zero;

    for(int i = 0; i < N_E; i++){
        for(int l = 0; l < N_L; l++){
            if(l==0)
                C[i] = r[i][0];
            else
                C[i] += (pow(-I*dt,l)/double(factor(l)))*r[i][l];
        }
    }

    r.clear();
    r.shrink_to_fit();
}

// for velocity gauge ... A(t)*i*d/dx
void TDSE::Hamiltonian_VG(ddcopxvec &H, ddvec &Intgral_simps, Complex *pulse, dvec &E, double Tc, int t_index, int N_E){

    double At = pulse[t_index].real();
    omp_set_num_threads(NTHREADS);
    #pragma omp parallel for default(shared) schedule(static, CHUNK)
    for(int m = 0; m < N_E; m++){
        for(int n = 0; n < N_E; n++){
            H[m][n] = E[n]*Kdelta(m,n) + At*I*Intgral_simps[m][n];
        }
    }   
}


double TDSE::t_step(int t_index){
    return t_0 + (t_index)*dt;
}

// multiply matrix-vector
void TDSE::MV_MultiTS(ddcopxvec &matrix, ddcopxvec &vector, int size, int l){

    dcopxvec result;
    result.resize(size);

    int i,j;
    omp_set_num_threads(NTHREADS);
    #pragma omp parallel shared(matrix,result,vector) private(i,j) 
    {
        #pragma omp for  schedule(static)
        for (i=0; i<size; i=i+1){
            result[i] = 0.;
            for (j=0; j<size; j=j+1){
                result[i]=(result[i])+((matrix[i][j])*(vector[j][l-1]));
            }
        }
    }

    for (int k = 0; k < size; k++)
        vector[k][l] = result[k];

    result.clear();
    result.shrink_to_fit();
}


double TDSE::Normalization(dcopxvec &psi, int N_E){

    double sum = 0.;
    for(int j = 0; j < N_E; j++)
        sum += (psi[j].real()*psi[j].real() + psi[j].imag()*psi[j].imag());

    sum = sqrt(sum);
    return sum;
}

void TDSE::id_tise(){

    Numerov_Solver(phi,E);
    //cout << N_E << endl;

    E.resize(N_E);
    phi.resize(N_graid);
    for(int i = 0; i < N_graid; ++i)
        phi[i].resize(N_E);

    for(int i = 0; i < N_graid; i++)
        cout << x('l',i) << "\t" << phi[i][0] << endl;
}

//for length gauge
void TDSE::SimpsonRule_overAlltimes(ddvec &phi, ddvec &Intgral_simps, int N_E){
    
    omp_set_num_threads(NTHREADS);
    #pragma omp parallel for default(shared) schedule(static, CHUNK)
    for(int index_m = 0; index_m < N_E; index_m++){
        for(int index_n = 0; index_n < N_E; index_n++){
            
            double sum = 0.0;
            for(int i =0; i < N_graid; i++){
                if(i%2==0 && i!=0 && i!=(N_graid - 1)){
                    sum += 2.*phi[i][index_m]*x('l',i)*phi[i][index_n];
                }
                else if(i%2!=0 && i!=0 && i!=(N_graid - 1)){
                    sum += 4.*phi[i][index_m]*x('l',i)*phi[i][index_n];
                }
                else{
                    sum += phi[i][index_m]*x('l',i)*phi[i][index_n];
                }
            }
                Intgral_simps[index_m][index_n] =  (dx/3.)*sum;            
        }
    }

    ofstream Iout("Integraldipole_"+strcpp(xhbox)+"_"+strcpp(e_up)+ ".dat");
    for(int index_m = 0; index_m < N_E; index_m++){
        for(int index_n = 0; index_n < N_E; index_n++){
            Iout << Intgral_simps[index_m][index_n] << endl;
        }
    }
    Iout.close();
}


//for velocity gauge
void TDSE::SimpsonRule_overAlltimes_VG(ddvec &phi, ddvec &dphi, ddvec &Intgral_simps, int N_E){
    
    omp_set_num_threads(NTHREADS);
    #pragma omp parallel for default(shared) schedule(static, CHUNK)
    for(int index_m = 0; index_m < N_E; index_m++){
        for(int index_n = 0; index_n < N_E; index_n++){
            
            double sum = 0.0;
            for(int i =0; i < N_graid; i++){
                if(i%2==0 && i!=0 && i!=(N_graid - 1)){
                    sum += 2.*phi[i][index_m]*dphi[i][index_n];
                }
                else if(i%2!=0 && i!=0 && i!=(N_graid - 1)){
                    sum += 4.*phi[i][index_m]*dphi[i][index_n];
                }
                else{
                    sum += phi[i][index_m]*dphi[i][index_n];
                }
            }
            Intgral_simps[index_m][index_n] =  (dx/3.)*sum;            
        }
    }

    ofstream Iout("Integraldipole_"+strcpp(xhbox)+"_"+strcpp(e_up)+ ".dat");
    for(int index_m = 0; index_m < N_E; index_m++){
        for(int index_n = 0; index_n < N_E; index_n++){
            Iout << Intgral_simps[index_m][index_n] << endl;
        }
    }
    Iout.close();
}

//derivative of the wave function
void TDSE::Numerical5p_differentiation(ddvec &phi, ddvec &dphi, int N_E){

    for(int n = 0; n < N_E; n++){
        for(int i = 0; i < N_graid; i++){
            if(i == 0)
                dphi[i][n] = (-phi[i+2][n] + 8.*phi[i+1][n])/(12.*dx);
            else if(i == N_graid-1)
                dphi[i][n] = (-8.*phi[i-1][n] + phi[i-2][n])/(12.*dx);
            else if(i == 1)
                dphi[i][n] = (-phi[i+2][n] + 8.*phi[i+1][n] -8.*phi[i-1][n])/(12.*dx);
            else if(i == N_graid-2)
                dphi[i][n] = (8.*phi[i+1][n] -8.*phi[i-1][n] + phi[i-2][n])/(12.*dx);
            else
                dphi[i][n] = (-phi[i+2][n] + 8.*phi[i+1][n] -8.*phi[i-1][n] + phi[i-2][n])/(12.*dx);
            //cout << dphi[i][n] << endl;
        }
    }
}

// solving time-independent schrodinger equation by Numerov's method
void TDSE::Numerov_Solver(ddvec &wavfun, dvec &eignVal){

    int im = int(N_graid/2);
    int N_R = N_graid - im + 2;
    int N_L = im + 2;

    ofstream fout;
    ofstream dout;
    double *psi_l = NULL;
    double *psi_r = NULL;
    double *k2_l = NULL;
    double *k2_r = NULL;



    int i_save = 0;
    double xp_min = -0.0;

    int ii = 0, inode = 0, inodeUp = 1;
    while(e_tr < e_up){

        if(xhbox > 1500.0){
            if(e_tr < 0.0){
                        //contersection eigenvalues with potential
                xp_min = -150.0;
                i_save = int(abs((xp_min - x_min)/dx));// - int(N_graid/100);
            }
            else{
                //xp_min = x_min;
                i_save = 0;
            }
        }
        else{
            //xp_min = x_min;
            i_save = 0;
        }
        /*if(xp_min-80 > x_min+10){
            //contersection eigenvalues with potential
            xp_min = -sqrt(0.4*0.4/(e_tr*e_tr) - 12) - 80.;
            i_save = int(abs((xp_min - x_min)/dx)) - int(N_graid/100);
        }
        else{
            //xp_min = x_min;
            i_save = 0;
        }*/
        
        psi_l = new double[N_L];
        psi_r = new double[N_R];
        k2_l = new double[N_L];
        k2_r = new double[N_R];
        
        for(int i = 0 ;i < N_L; i++){
            psi_l[i] = 0.;
            psi_r[i] = 0.;
            k2_l[i] = 0.;
            k2_r[i] = 0.;
        }
        
        double ds1 = diff_slope(psi_r,psi_l,k2_r,k2_l,e_tr,N_L,N_R,i_save);
        e_tr += de;
        double ds2 = diff_slope(psi_r,psi_l,k2_r,k2_l,e_tr,N_L,N_R,i_save);

        if(ds1*ds2 < 0.0){
            
            eignVal.push_back(e_tr);
            cout << "E_" <<ii << " = " << e_tr << endl;
            //fout.open("psi_"+string_precision(ii,0)+ ".dat");

            Normalizer(psi_r,im);
            Normalizer(psi_l,im);

            if(inode%2 != 0.){
                for(int i = 0; i < im; i++){
                    //fout << x('l',i) << "\t" << -psi_l[i] << endl;
                    wavfun.push_back(dvec());
                    wavfun[i].push_back(-psi_l[i]);
                }
            }
            else{
                for(int i = 0; i < im; i++){
                    //fout << x('l',i) << "\t" << psi_l[i] << endl;
                    wavfun.push_back(dvec());
                    wavfun[i].push_back(psi_l[i]);
                }
            }
            int cc = im;
            for(int i = im-1; i > -1; i--){
                //fout << x('r',i) << "\t" << psi_r[i] << endl;
                wavfun.push_back(dvec());
                wavfun[cc++].push_back(psi_r[i]);
            }

            inode++;

            ii++;
            if(e_tr < 0.0){
                dout.open("groundwavefunction.dat");
                for(int i = 0; i < N_graid; i++)
                    dout << x('l',i) << "\t" << wavfun[i][0] <<  endl;
                dout.close();
            }
            //fout.close();
        }
        
        //cout << x_min << "\t\t\t" << x_max << endl;
        delete[] psi_l;
        delete[] psi_r;
        delete[] k2_l;
        delete[] k2_r;
    }

    N_E = ii;
}

//obtaining wave function whwn we have already the eigenvalues
void TDSE::time_independent_wavefunction(ddvec &wavfun, dvec &eignVal, int N_E){
    
    eignVal.resize(N_E);

    int im = int(N_graid/2);
    int N_R = N_graid - im + 2;
    int N_L = im + 2;

    ofstream fout;
    double *psi_l = NULL;
    double *psi_r = NULL;
    double *k2_l = NULL;
    double *k2_r = NULL;
    psi_l = new double[N_L];
    psi_r = new double[N_R];
    k2_l = new double[N_L];
    k2_r = new double[N_R];


    int i_save = 0;
    double xp_min = -0.0;

    for(int p = 0; p < N_E; p++){//(e_tr < e_up){
        e_tr = eignVal[p];
        if(xhbox > 1500.0){
            if(e_tr < 0.0){
                //contersection eigenvalues with potential
                xp_min = -150.0;
                i_save = int(abs((xp_min - x_min)/dx));// - int(N_graid/100);
            }
            else{
                //xp_min = x_min;
                i_save = 0;
            }
        }
        else{
            //xp_min = x_min;
            i_save = 0;
        }
        /*if(xp_min-80 > x_min+10){
            //contersection eigenvalues with potential
            xp_min = -sqrt(0.4*0.4/(e_tr*e_tr) - 12.0) - 80.;
            i_save = int(abs((xp_min - x_min)/dx)) - int(N_graid/100);
        }
        else{
            //xp_min = x_min;
            i_save = 0;
        }*/
        //cout << i_save << endl;
        for(int i = 0 ;i < N_L; i++){
            psi_l[i] = 0.;
            psi_r[i] = 0.;
            k2_l[i] = 0.;
            k2_r[i] = 0.;
        }
        
        psi_l[i_save] = 0.0;
        psi_l[i_save + 1] = c;
        //initialization right solution
        psi_r[i_save] = 0.0;
        psi_r[i_save + 1] = c;
        
        Setk2(k2_l, e_tr, N_L, 'l', i_save);
        Setk2(k2_r, e_tr, N_R, 'r', i_save);
        Numerov(psi_l, k2_l, N_L, i_save);
        Numerov(psi_r, k2_r, N_R, i_save);
        
        
        //fout.open("psi_"+string_precision(ii,0)+ ".dat");

        Normalizer(psi_r,im);
        Normalizer(psi_l,im);

        if(p%2 != 0.){
            for(int i = 0; i < im; i++){
                //cout << x('l',i) << "\t" << -psi_l[i] << endl;
                wavfun.push_back(dvec());
                wavfun[i].push_back(-psi_l[i]);
            }
        }
        else{
            for(int i = 0; i < im; i++){
                //cout << x('l',i) << "\t" << psi_l[i] << endl;
                wavfun.push_back(dvec());
                wavfun[i].push_back(psi_l[i]);
            }
        }
        int cc = im;
        for(int i = im-1; i > -1; i--){
            //cout << x('r',i) << "\t" << psi_r[i] << endl;
            wavfun.push_back(dvec());
            wavfun[cc++].push_back(psi_r[i]);
        }

        //fout.close();
        
        //cout << x_min << "\t\t\t" << x_max << endl;
    }
    delete[] psi_l;
    delete[] psi_r;
    delete[] k2_l;
    delete[] k2_r;
}



void TDSE::Normalizer(double *psi,int N){

    double sum = 0.;
    for(int i=0 ; i<N ; i++)
        sum += psi[i]*psi[i];

    sum=sqrt(2.*sum*dx); /* 2 must be here due to we have left an right sotuions(this mistake took my time about one month :(((( just 2 :( )*/
    for(int i=0 ; i<N ; i++){
        psi[i]=psi[i]/sum;
    }
}


void TDSE::Numerov(double *psi, double *k2 , int N, int i_save){

    double h12 = (dx*dx)/12.;
    for(int i = i_save + 1; i < N; i++){
        psi[i+1] = (2.*(1.-5.*h12*k2[i])*psi[i] - (1. + h12*k2[i-1])*psi[i-1])/(1. + h12*k2[i+1]);
    }
    //cout << k2[3] << endl;
}

void TDSE::Setk2(double *k2, double E, int N, char lr, int i_save){

    omp_set_num_threads(NTHREADS);
    #pragma omp parallel for default(shared) schedule(static, CHUNK)
    for(int i = i_save; i < N; i++){
        k2[i] = (2.*m/(h_bar*h_bar))*(E - Potential(lr,i));
    }
    //cout << "k2[3] = " <<k2[3]<<endl;
}

double TDSE::Potential(char lr, int index){
    //return 0.5*x(lr, index)*x(lr, index);
    //return -0.4/sqrt(x(lr, index)*x(lr, index)+12);
    double a1 = 24.856;
    double a2 = 0.16093;
    double a3 = 0.25225;
    return -exp(-a1*sqrt((x(lr, index)/a1)*(x(lr, index)/a1) + a2*a2))/(sqrt((x(lr, index)/a1)*(x(lr, index)/a1) + a3*a3));
}

double TDSE::x(char lr, int index){
    if(lr == 'l')
        return x_min + (index)*dx; // dx = h
    else
        return x_max - (index)*dx; // the right solution
}

double TDSE::diff_slope(double *psi_r, double *psi_l, double *k2_r, double *k2_l, double e_tr, int N_L, int N_R, int i_save){

    double dslope;
    //initialization left solution
    psi_l[i_save] = 0.0;
    psi_l[i_save + 1] = c;
    //initialization right solution
    psi_r[i_save] = 0.0;
    psi_r[i_save + 1] = c;
    Setk2(k2_l, e_tr, N_L, 'l', i_save);
    Setk2(k2_r, e_tr, N_R, 'r', i_save);
    Numerov(psi_l, k2_l, N_L, i_save);
    Numerov(psi_r, k2_r, N_R, i_save);
    double y_m = (psi_l[N_L-1]+psi_r[N_R-1])/2.;
    dslope = (2.*y_m - psi_l[N_L - 3] - psi_r[N_R - 3])/(dx*psi_r[N_R - 2]);
    //cout << dslope << endl;

	return dslope;
}


string TDSE::string_precision(int value, int n_digits){
    if(n_digits < 6) n_digits = 6;
    ostringstream out;
    out << fixed<<setprecision(n_digits) << value;
    return out.str();
}


template <class T>
string  TDSE::strcpp(T value){
    ostringstream out;
    out << value;
    return out.str();
}

int main(){

    TDSE tdse;
    tdse.tdse();
    //tdse.id_tise();
    return 0;
}

