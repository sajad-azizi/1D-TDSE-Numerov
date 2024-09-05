//********************************************************************
// File EbDawson.cpp                                   Dawson Integral
//********************************************************************
// Brought to you by courtesy of Extra Byte, www.ebyte.it
// For details, see www.dx.doi.org/10.3247/SL4Soft12.001
// Version 1.0 (2012)


//********************************************************************
// File EbDawson.h                                     Dawson Integral
//********************************************************************
#ifndef _EbDawson_hpp_
#define _EbDawson_hpp_

// Brought to you by courtesy of Extra Byte, www.ebyte.it
// For details, see www.dx.doi.org/10.3247/SL4Soft12.001
// Version 1.0 (2012)

typedef long double REAL;

//********************************************************************
// McCabe's continued fraction algorithm
//********************************************************************
// For details, see www.dx.doi.org/10.3247/SL4Soft12.001

REAL DawsonByMcCabeCF(REAL x);   // Slow, but precise to 17 dec.digits

//********************************************************************
// Stan Sykora's rational function approximations
//********************************************************************
// The approximations were optimized for best RELATIVE error
// The numeric postfix denotes the approximation order

double Dawson1(double x);       // max rel.error 45616 ppm
double Dawson2(double x);       // max rel.error  5185 ppm
double Dawson3(double x);       // max rel.error   693 ppm
double Dawson4(double x);       // max rel.error    90 ppm
double Dawson5(double x);       // max rel.error    10 ppm

//********************************************************************
#endif


#include <math.h>

//********************************************************************
// Cobe's continued fraction algorithm
//********************************************************************

REAL DawsonByMcCabeCF(REAL x)
//--------------------------------------------------------------------
// This uses the continued fraction expansion by J.H.McCabe
// and is precise to over 17 decimal digits (relative err).
// The argument may be any real number
{
  int  n;
  REAL f,a2,a,b,am1,bm1,an,bn,fn,newa,newb;

  a2   = 2*x*x;                 // Optimized initialization
  am1  = 1;
  bm1  = 1+a2;
  a    = -2*a2;
  b    = 3+a2;
  newa = b;
  newb = b*bm1 + a;
  fn   = newa/newb;
  an   = newa;
  bn   = newb;
  n    = 2;

  while (n < 100) {             // Worst case cycles: about 30
    n = n+1;                    // Iterative CF evaluation
    a = a - 2*a2;
    b = b + 2;
    newa = b*an + a*am1;
    newb = b*bn + a*bm1;
    f = newa/newb;
    if (f == fn) break;
    fn  = f;
    am1 = an;
    bm1 = bn;
    an  = newa;
    bn  = newb;
  }

  return x*f;                   // Final adjustment
}

//********************************************************************
// Stan Sykora's rational function approximations
//********************************************************************
// In all cases, optimized evaluation via the Horner scheme is used

double Dawson1(double x)        // max rel.error 45616 ppm
//--------------------------------------------------------------------
{
  double y,p,q;
  y = x*x;
  p = 1.0 + y*(0.4582332073);
  q = 1.0 + y*(0.8041350741 + 2*0.4582332073*y);
  return x*(p/q);
}

double Dawson2(double x)        // max rel.error  5185 ppm
//--------------------------------------------------------------------
{
  double y,p,q;
  y = x*x;
  p = 1.0 + y*(0.1329766230 + y*(0.0996005943));
  q = 1.0 + y*(0.8544964660 + y*(0.2259838671 + 2*0.0996005943*y));
  return x*(p/q);
}

double Dawson3(double x)        // max rel.error   693 ppm
//--------------------------------------------------------------------
{
  double y,p,q;
  y = x*x;
  p = 1.0 + y*(0.1349423927 + y*(0.0352304655
          + y*(0.0138159073)));
  q = 1.0 + y*(0.8001569104 + y*(0.3190611301
          + y*(0.0540828748 + 2*0.0138159073*y)));
  return x*(p/q);
}

double Dawson4(double x)        // max rel.error    90 ppm
//--------------------------------------------------------------------
{
  double y,p,q;
  y = x*x;
  p = 1.0 + y*(0.1107817784 + y*(0.0437734184
          + y*(0.0049750952 + y*(0.0015481656))));
  q = 1.0 + y*(0.7783701713 + y*(0.2924513912
          + y*(0.0756152146 + y*(0.0084730365 + 2*0.0015481656*y))));
  return x*(p/q);
}

double Dawson5(double x)        // max rel.error    10 ppm
//--------------------------------------------------------------------
{
  double y,p,q;
  y = x*x;
  p = 1.0 + y*(0.1049934947 + y*(0.0424060604
          + y*(0.0072644182 + y*(0.0005064034
          + y*(0.0001789971)))));
  q = 1.0 + y*(0.7715471019 + y*(0.2909738639
          + y*(0.0694555761 + y*(0.0140005442
          + y*(0.0008327945 + 2*0.0001789971*y)))));
  return x*(p/q);
}

//--------------------------------------------------------------------


