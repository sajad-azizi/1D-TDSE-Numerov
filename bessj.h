/* bessel.c
                      Copyright (c) 1998 
                  Kapteyn Institute Groningen
                     All Rights Reserved.
*/

/*
#>            bessel.dc2

Function:     BESSEL

Purpose:      Evaluate Bessel function J, Y, I, K of integer order.

Category:     MATH

File:         bessel.c

Author:       M.G.R. Vogelaar

Use:          See bessj.dc2, bessy.dc2, bessi.dc2 or bessk.dc2
                      
Description:  The differential equation 

                       2
                   2  d w       dw      2   2
                  x . --- + x . --- + (x - v ).w = 0
                        2       dx
                      dx
                      
              has two solutions called Bessel functions of the first kind
              Jv(x) and Bessel functions of the second kind Yv(x).
              The routines bessj and bessy return the J and Y for 
              integer v and therefore are called Bessel functions 
              of integer order.
              
              The differential equation 

                       2
                   2  d w       dw      2   2
                  x . --- + x . --- - (x + v ).w = 0
                        2       dx
                      dx
                      
              has two solutions called modified Bessel functions
              Iv(x) and Kv(x).
              The routines bessi and bessk return the I and K for 
              integer v and therefore are called Modified Bessel 
              functions of integer order.
              (Abramowitz & Stegun, Handbook of mathematical 
              functions, ch. 9, pages 358,- and 374,- )
                            
              The implementation is based on the ideas from 
              Numerical Recipes, Press et. al. 
              This routine is NOT callable in FORTRAN.

Updates:      Jun 29, 1998: VOG, Document created.
#<
*/

/*
#> bessel.h
#if !defined(_bessel_h_) 
#define _bessel_h_
extern double bessj( int, double );
extern double bessy( int, double );
extern double bessi( int, double );
extern double bessk( int, double );
#endif
#<
*/

                      

// #include<iostream>
#include     "math.h"
// #include     "setdblank.h"


#define ACC 40.0
#define BIGNO 1.0e10
#define BIGNI 1.0e-10



static double bessj0( double x )
/*------------------------------------------------------------*/
/* PURPOSE: Evaluate Bessel function of first kind and order  */
/*          0 at input x                                      */
/*------------------------------------------------------------*/
{
   double ax,z;
   double xx,y,ans,ans1,ans2;

   if ((ax=fabs(x)) < 8.0) {
      y=x*x;
      ans1=57568490574.0+y*(-13362590354.0+y*(651619640.7
         +y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))));
      ans2=57568490411.0+y*(1029532985.0+y*(9494680.718
         +y*(59272.64853+y*(267.8532712+y*1.0))));
      ans=ans1/ans2;
   } else {
      z=8.0/ax;
      y=z*z;
      xx=ax-0.785398164;
      ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
         +y*(-0.2073370639e-5+y*0.2093887211e-6)));
      ans2 = -0.1562499995e-1+y*(0.1430488765e-3
         +y*(-0.6911147651e-5+y*(0.7621095161e-6
         -y*0.934935152e-7)));
      ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
   }
   return ans;
}



static double bessj1( double x )
/*------------------------------------------------------------*/
/* PURPOSE: Evaluate Bessel function of first kind and order  */
/*          1 at input x                                      */
/*------------------------------------------------------------*/
{
   double ax,z;
   double xx,y,ans,ans1,ans2;

   if ((ax=fabs(x)) < 8.0) {
      y=x*x;
      ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
         +y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
      ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
         +y*(99447.43394+y*(376.9991397+y*1.0))));
      ans=ans1/ans2;
   } else {
      z=8.0/ax;
      y=z*z;
      xx=ax-2.356194491;
      ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
         +y*(0.2457520174e-5+y*(-0.240337019e-6))));
      ans2=0.04687499995+y*(-0.2002690873e-3
         +y*(0.8449199096e-5+y*(-0.88228987e-6
         +y*0.105787412e-6)));
      ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
      if (x < 0.0) ans = -ans;
   }
   return ans;
}



/*
#>            bessj.dc2

Function:     bessj

Purpose:      Evaluate Bessel function of first kind of integer order.

Category:     MATH

File:         bessel.c

Author:       M.G.R. Vogelaar

Use:          #include "bessel.h"
              double   result; 
              result = bessj( int n,
                              double x )


              bessj    Return the Bessel function of integer order
                       for input value x.
              n        Integer order of Bessel function.
              x        Double at which the function is evaluated.

                      
Description:  bessj evaluates at x the Bessel function of the first kind 
              and of integer order n. 
              This routine is NOT callable in FORTRAN.

Updates:      Jun 29, 1998: VOG, Document created.
#<
*/


double bessj( int n, double x )
/*------------------------------------------------------------*/
/* PURPOSE: Evaluate Bessel function of first kind and order  */
/*          n at input x                                      */
/* The function can also be called for n = 0 and n = 1.       */
/*------------------------------------------------------------*/
{
   int    j, jsum, m;
   double ax, bj, bjm, bjp, sum, tox, ans;


   /*if (n < 0)
   {
      double   dblank;
      setdblank_c( &dblank );
      return( dblank );
   }*/
   ax=fabs(x);
   if (n == 0)
      return( bessj0(ax) );
   if (n == 1)
      return( bessj1(ax) );
      

   if (ax == 0.0)
      return 0.0;
   else if (ax > (double) n) {
      tox=2.0/ax;
      bjm=bessj0(ax);
      bj=bessj1(ax);
      for (j=1;j<n;j++) {
         bjp=j*tox*bj-bjm;
         bjm=bj;
         bj=bjp;
      }
      ans=bj;
   } else {
      tox=2.0/ax;
      m=2*((n+(int) sqrt(ACC*n))/2);
      jsum=0;
      bjp=ans=sum=0.0;
      bj=1.0;
      for (j=m;j>0;j--) {
         bjm=j*tox*bj-bjp;
         bjp=bj;
         bj=bjm;
         if (fabs(bj) > BIGNO) {
            bj *= BIGNI;
            bjp *= BIGNI;
            ans *= BIGNI;
            sum *= BIGNI;
         }
         if (jsum) sum += bj;
         jsum=!jsum;
         if (j == n) ans=bjp;
      }
      sum=2.0*sum-bj;
      ans /= sum;
   }
   return  x < 0.0 && n%2 == 1 ? -ans : ans;
}




#undef ACC
#undef BIGNO
#undef BIGNI /*


int main(){
    std::cout << bessj(0, 8.0) << std::endl;
}*/
