// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

/**
 * xval: design variables
 * dfdx: gradient of objective
 * gx: constraint values
 * dgdx: gradient of constraints
 * xmin: lower bound on design variables
 * xmax: upper bound on design variables
 * norm2: norm of KKT residual
 * normInf: infinity norm of KKT residual
 * xo1: old design variables
 * xo2: old design variables
 * U: upper asymptote
 * L: lower asymptote
 * init: initial asymptote
 * decrease: decrease factor
 * increase: increase factor
 * n: number of design variables
 * m: number of constraints
*/

#include "mtop_MMA.hpp"
#include <fstream>
#include <iostream>

//#ifdef MFEM_USE_LAPACK
extern "C" void dgesv_(int* nLAP, int* nrhs, double* AA, int* lda, int* ipiv,
                       double* bb, int* ldb, int* info);
//#endif

namespace mma
{
MMA::MMA(int n, int m, double * x, int sx)
{
   std::cout << "Initialized" << std::endl;
}

//MMA::MMA(MPI_Comm Comm,int n, int m, double * x, int sx) {}

void MMA::mmasub(int nVar, int nCon, int iter, double* xval, double* xmin,
                 double* xmax, double* xold1, double* xold2, double* fval,
                 double* dfdx, double* gx, double* dgdx, double* low,
                 double* upp, double a0, double* a, double* c, double* d,
                 double* xmma, double* ymma, double* zmma, double* lam,
                 double* xsi, double* eta, double* mu, double& zet, double* s)
{
   double epsimin = 1e-7;
   double raa0 = 0.00001;
   double move = 0.5;
   double albefa = 0.1;
   double asyinit = 0.5;
   double asyincr = 1.2;
   double asydecr = 0.7;
   double xmamieps = 1e-5;

   double* factor = new double[nVar];
   double lowmin, lowmax, uppmin, uppmax, z = 0;
   double* xmami = new double[nVar];
   double* ux1 = new double[nVar];
   double* xl1 = new double[nVar];
   double* p0 = new double[nVar];
   double* q0 = new double[nVar];
   double* pq0 = new double[nVar];
   double* p = new double[nVar * nCon];
   double* q = new double[nVar * nCon];
   double* pq = new double[nVar * nCon];
   double* b = new double[nCon];
   double* alfa = new double[nVar];
   double* beta = new double[nVar];

   double* P = new double[nCon * nVar];
   double* Q = new double[nCon * nVar];
   double* PQ = new double[nCon * nVar];

   for (int i = 0; i < nVar; i++)
   {
      factor[i] = asyincr;
      p0[i] = 0.0;
      q0[i] = 0.0;
      pq0[i] = 0.0;
      xmami[i] = 0.0;
      ux1[i] = 0.0;
      xl1[i] = 0.0;
      alfa[i] = 0.0;
      beta[i] = 0.0;
   }
   for (int i = 0; i < (nCon*nVar); i++)
   {
      p[i] = 0.0;
      q[i] = 0.0;
      pq[i] = 0.0;
      P[i] = 0.0;
      Q[i] = 0.0;
      PQ[i] = 0.0;
   }

   for (int i = 0; i < nCon; i++)
   {
      b[i] = 0.0;
   }

   // Calculation of the asymptotes low and upp
   if (iter < 2.5)
   {
      for (int i = 0; i < nVar; i++)
      {
         low[i] = xval[i] - asyinit * (xmax[i] - xmin[i]);
         upp[i] = xval[i] + asyinit * (xmax[i] - xmin[i]);
      }
   }
   else
   {

      for (int i = 0; i < nVar; i++)
      {
         //Determine sign
         z = (xval[i] - xold1[i]) * (xold1[i] - xold2[i]);
         if (z > 0.0)
         {
            factor[i] = asyincr;
         }
         else if (z < 0.0)
         {
            factor[i] = asydecr;
         }

         //Find new asymptote
         low[i] = xval[i] - factor[i] * (xold1[i] - low[i]);
         upp[i] = xval[i] + factor[i] * (upp[i] - xold1[i]);

         lowmin = xval[i] - 10.0 * (xmax[i] - xmin[i]);
         lowmax = xval[i] - 0.01 * (xmax[i] - xmin[i]);
         uppmin = xval[i] + 0.01 * (xmax[i] - xmin[i]);
         uppmax = xval[i] + 10.0 * (xmax[i] - xmin[i]);

         low[i] = std::max(low[i], lowmin);
         low[i] = std::min(low[i], lowmax);
         upp[i] = std::min(upp[i], uppmax);
         upp[i] = std::max(upp[i], uppmin);
      }
   }

   for (int i = 0; i < nVar; i++)
   {
      // Calculation of bounds alfa and beta according to:
      // alfa = max{xmin, low + 0.1(xval-low), xval-0.5(xmax-xmin)}
      // beta = min{xmax, upp - 0.1(upp-xval), xval+0.5(xmax-xmin)}
      alfa[i] = std::max(std::max(low[i] + albefa * (xval[i] - low[i]),
                                  xval[i] - move * (xmax[i] - xmin[i])), xmin[i]);
      beta[i] = std::min(std::min(upp[i] - albefa * (upp[i] - xval[i]),
                                  xval[i] + move * (xmax[i] - xmin[i])), xmax[i]);
      xmami[i] = std::max(xmax[i] - xmin[i], xmamieps);

      // Calculations of p0, q0, P, Q, and b
      ux1[i] = upp[i] - xval[i];
      xl1[i] = xval[i] - low[i];

      p0[i] = std::max(dfdx[i], 0.0);
      q0[i] = std::max(-dfdx[i], 0.0);
      pq0[i] = 0.001 * (p0[i] + q0[i]) + raa0 / xmami[i];
      p0[i] += pq0[i];
      q0[i] += pq0[i];
      p0[i] *= ux1[i] * ux1[i];
      q0[i] *= xl1[i] * xl1[i];
   }
   // P = max(dgdx,0)
   // Q = max(-dgdx,0)
   // PQ = 0.001(P+Q) + raa0/xmami
   // P = P + PQ
   // Q = Q + PQ
   for (int i = 0; i < nCon; i++)
   {
      for (int j = 0; j < nVar; j++)
      {
         p[i * nVar + j] = std::max(dgdx[i * nVar + j], 0.0);
         q[i * nVar + j] = std::max(-1*dgdx[i * nVar + j], 0.0);
         pq[i * nVar + j] = 0.001 * (p[i * nVar + j] + q[i * nVar + j]) + raa0 /
                            xmami[j];
         p[i * nVar + j] += pq[i * nVar + j];
         q[i * nVar + j] += pq[i * nVar + j];
         // P = P * spdiags(ux2,0,n,n)
         // Q = Q * spdiags(xl2,0,n,n)
         P[i * nVar + j] = p[i * nVar + j] * ux1[j] * ux1[j];
         Q[i * nVar + j] = q[i * nVar + j] * xl1[j] * xl1[j];
         // b = P/ux1 + Q/xl1 - gx
         b[i] += P[i * nVar + j] / ux1[j] + Q[i * nVar + j] / xl1[j];
      }
      b[i] -= gx[i];
   }

   subsolv(nVar, nCon, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d,
           xmma, ymma, zmma, lam, xsi, eta, mu, &zet, s);

   delete[] factor;
   delete[] xmami;
   delete[] ux1;
   delete[] xl1;
   delete[] p0;
   delete[] q0;
   delete[] p;
   delete[] q;
   delete[] pq;
   delete[] b;
   delete[] alfa;
   delete[] beta;
   delete[] pq0;
   delete[] P;
   delete[] Q;
   delete[] PQ;
}

/**
 * This function solves the MMA subproblem:
 * minimize   SUM[ p0j/(uppj-xj) + q0j/(xj-lowj) ] + a0*z +
 *         + SUM[ ci*yi + 0.5*di*(yi)^2 ],
 *
 * subject to SUM[ pij/(uppj-xj) + qij/(xj-lowj) ] - ai*z - yi <= bi,
 *           alfaj <=  xj <=  betaj,  yi >= 0,  z >= 0.
 *
 * Input: m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b.
 * Output: xmma, ymma, zmma, slack variables and Lagrange multiplers.
*/
void MMA::subsolv(int nVar, int nCon, double epsimin, double* low, double* upp,
                  double* alfa, double* beta, double* p0,
                  double* q0, double* P, double* Q,
                  double a0, double* a, double* b, double* c,
                  double* d, double* xmma, double* ymma,
                  double* zmma, double* lamma, double* xsimma,
                  double* etamma, double* mumma, double* zetmma, double* smma)
{
   double epsi = 1.0;
   double machineEpsilon = 1e-10;
   double* epsvecn = new double[nVar];
   double* epsvecm = new double[nCon];
   double* x = new double[nVar];
   double* y = new double[nCon];
   double z = 1.0;
   double* lam = new double[nCon];
   double* xsi = new double[nVar];
   double* eta = new double[nVar];
   double* mu = new double[nCon];
   double zet = 1.0;
   double* s = new double[nCon];
   int ittt, itto, itera = 0;

   double* ux1 = new double[nVar];
   double* ux2 = new double[nVar];
   double* ux3 = new double[nVar];
   double* xl1 = new double[nVar];
   double* xl2 = new double[nVar];
   double* xl3 = new double[nVar];
   double* uxinv1 = new double[nVar];
   double* xlinv1 = new double[nVar];
   double* plam = new double[nVar];
   double* qlam = new double[nVar];
   double* gvec = new double[nCon];
   double* dpsidx = new double[nVar];
   double* rex = new double[nVar];
   double* rey = new double[nCon];
   double rez;
   double* relam = new double[nCon];
   double* rexsi = new double[nVar];
   double* reeta = new double[nVar];
   double* remu = new double[nCon];
   double rezet;
   double* res = new double[nCon];
   double* residu1 = new double[nVar + nCon + 1];
   double* residu2 = new double[3 * nCon + 2 * nVar + 1];
   double* residu = new double[3 * nVar + 4 * nCon + 2];
   double residunorm, residumax, resinew;
   double* GG = new double[nVar * nCon];
   double* Puxinv = new double[nVar * nCon];
   double* Qxlinv = new double[nVar * nCon];
   double* delx = new double[nVar];
   double* dely = new double[nCon];
   double delz;
   double* dellam = new double[nCon];
   double* dellamyi = new double[nCon];
   double* diagx = new double[nVar];
   double* diagxinv = new double[nVar];
   double* diagy = new double[nCon];
   double* diagyinv = new double[nCon];
   double* diaglam = new double[nCon];
   double* diaglamyi = new double[nCon];
   long double* diaglamyiinv = new long double[nCon];
   double* blam = new double[nCon];
   double* bb = new double[nVar + 1];
   double* Alam = new double[nCon * nCon];
   double* AA = new double[(nVar + 1) * (nVar + 1)];
   double* solut = new double[nVar + nCon + 1];
   double* dlam = new double[nCon];
   double dz;
   double* dx = new double[nVar];
   double* dy = new double[nCon];
   double* dxsi = new double[nVar];
   double* deta = new double[nVar];
   double* dmu = new double[nCon];
   double dzet;
   double* Axx = new double[nVar * nCon];
   double azz;
   double* axz = new double[nVar];
   double* ds = new double[nCon];
   double* xx = new double[4 * nCon + 2 * nVar + 2];
   double* dxx = new double[4 * nCon + 2 * nVar + 2];
   double* stepxx = new double[4 * nCon + 2 * nVar + 2];
   double sum, sum1 = 0;
   double stmxx;
   double* stepalfa = new double[nVar];
   double stmalfa;
   double* stepbeta = new double[nVar];
   double stmbeta;
   double stmalbe;
   double stmalbexx;
   double stminv;
   double steg;
   double* xold = new double[nVar];
   double* yold = new double[nCon];
   double zold;
   double* lamold = new double[nCon];
   double* xsiold = new double[nVar];
   double* etaold = new double[nVar];
   double* muold = new double[nCon];
   double zetold;
   double* sold = new double[nCon];

   std::ofstream results;
   results.open("sub.dat", std::ios::app);

   for (int i = 0; i < nVar; i++)
   {
      epsvecn[i] = epsi;
      x[i] = 0.5 * (alfa[i] + beta[i]);
      xsi[i] = 1/(x[i] - alfa[i]);
      xsi[i] = std::max(xsi[i], 1.0);
      eta[i] = 1/(beta[i] - x[i]);
      eta[i] = std::max(eta[i], 1.0);
   }
   for (int i = 0; i < nCon; i++)
   {
      epsvecm[i] = epsi;
      y[i] = 1.0;
      lam[i] = 1.0;
      mu[i] = std::max(1.0, 0.5 * c[i]);
      s[i] = 1;
   }


   while (epsi > epsimin)
   {
      rez = a0 - zet;
      for (int i = 0; i < nVar; i++)
      {
         epsvecn[i] = epsi;
         ux1[i] = upp[i] - x[i];
         xl1[i] = x[i] - low[i];
         ux2[i] = ux1[i]*ux1[i];
         xl2[i] = xl1[i]*xl1[i];
         uxinv1[i] = 1.0 / ux1[i];
         xlinv1[i] = 1.0 / xl1[i];

         // plam = P' * lam, qlam = Q' * lam
         plam[i] = p0[i];
         qlam[i] = q0[i];
         for (int j = 0; j < nCon; j++)
         {
            plam[i] += P[j * nVar + i] * lam[j];
            qlam[i] += Q[j * nVar + i] * lam[j];
         }
         dpsidx[i] = plam[i] * uxinv1[i] * uxinv1[i] - qlam[i] * xlinv1[i] * xlinv1[i];
         rex[i] = dpsidx[i] - xsi[i] + eta[i];
         rez -= a[i] * lam[i];
         rexsi[i] = xsi[i] * (x[i] - alfa[i]) - epsvecn[i];
         if (rexsi[i] <= machineEpsilon)
         {
            rexsi[i] = machineEpsilon;
         }
         reeta[i] = eta[i] * (beta[i] - x[i]) - epsvecn[i];
         if (reeta[i] <= machineEpsilon)
         {
            reeta[i] = machineEpsilon;
         }
         residu1[i] = rex[i];
         residu2[nCon + i] = rexsi[i];
         residu2[nCon + nVar + i] = reeta[i];
      }
      for (int i = 0; i < nCon; i++)
      {
         epsvecm[i] = epsi;
         // gvec = P/ux + Q/xl
         for (int j = 0; j < nVar; j++)
         {
            gvec[i] += P[i * nVar + j] * uxinv1[j] + Q[i * nVar + j] * xlinv1[j];
         }
         rey[i] = c[i] + d[i] * y[i] - mu[i] - lam[i];
         relam[i] = gvec[i] - a[i] * z - y[i] + s[i] - b[i];
         remu[i] = mu[i] * y[i] - epsvecm[i];
         res[i] = lam[i] * s[i] - epsvecm[i];

         residu2[i] = relam[i];
         residu1[nVar + i] = rey[i];
         residu2[nCon + 2 * nVar + i] = remu[i];
         residu2[2 * nVar + 2 * nCon + 1 + i] = res[i];
      }
      rezet = zet * z - epsi;
      residu1[nVar + nCon] = rez;
      residu2[2 * nVar + 2 * nCon] = rezet;

      // Concatenate the residuals
      for (int i = 0; i < (nVar + nCon + 1); i++)
      {
         residu[i] = residu1[i];
      }
      for (int i = 0; i < (2 * nVar + 3 * nCon + 1); i++)
      {
         residu[nVar + nCon + 1 + i] = residu2[i];
      }

      //Get vector product and maximum absolute value
      residunorm = 0.0;
      residumax = 0.0;
      for (int i = 0; i < (3 * nVar + 4 * nCon + 2); i++)
      {
         residunorm += residu[i] * residu[i];
         residumax = std::max(residumax, std::abs(residu[i]));
      }
      // Norm of the residual
      residunorm = std::sqrt(residunorm);

      ittt = 0;

      while (residumax > 0.9 * epsi && ittt < 200)
      {
         ittt++;
         for (int i = 0; i < nVar; i++)
         {
            ux1[i] = upp[i] - x[i];
            xl1[i] = x[i] - low[i];
            ux2[i] = ux1[i]*ux1[i];
            xl2[i] = xl1[i]*xl1[i];
            ux3[i] = ux2[i]*ux1[i];
            xl3[i] = xl2[i]*xl1[i];
            uxinv1[i] = 1.0 / ux1[i];
            xlinv1[i] = 1.0 / xl1[i];

            // plam = P' * lam, qlam = Q' * lam
            plam[i] = p0[i];
            qlam[i] = q0[i];
            for (int j = 0; j < nCon; j++)
            {
               plam[i] += P[j * nVar + i] * lam[j];
               qlam[i] += Q[j * nVar + i] * lam[j];
            }
            dpsidx[i] = plam[i] * uxinv1[i] * uxinv1[i] - qlam[i] * xlinv1[i] * xlinv1[i];
            delx[i] = dpsidx[i] - epsvecn[i] / (x[i] - alfa[i]) + epsvecn[nVar + i] /
                      (beta[i] - x[i]);
            if (delx[i] <= machineEpsilon)
            {
               delx[i] = machineEpsilon;
            }
            diagx[i] = 2 * (plam[i] / ux3[i] + qlam[i] / xl3[i]) + xsi[i] /
                       (x[i] - alfa[i]) + eta[i] / (beta[i] - x[i]);
            if (diagx[i] <= machineEpsilon)
            {
               diagx[i] = machineEpsilon;
            }
            diagxinv[i] = 1.0 / diagx[i];
         }
         delz = a0 - epsi/z;

         for (int i = 0; i < nCon; i++)
         {
            gvec[i] = 0.0;
            // gvec = P/ux + Q/xl
            for (int j = 0; j < nVar; j++)
            {
               gvec[i] += P[i * nVar + j] * uxinv1[j] + Q[i * nVar + j] * xlinv1[j];
               Puxinv[i * nVar + j] = P[i * nVar + j] * uxinv1[j] * uxinv1[j];
               Qxlinv[i * nVar + j] = Q[i * nVar + j] * xlinv1[j] * xlinv1[j];

               GG[i * nVar + j] = Puxinv[i * nVar + j] - Qxlinv[i * nVar + j];
            }

            dely[i] = c[i] + d[i] * y[i] - lam[i] - epsvecm[i] / y[i];
            delz -= a[i] * lam[i];
            dellam[i] = gvec[i] - a[i] * z - y[i] - b[i] + epsvecm[i] / lam[i];
            diagy[i] = d[i] + mu[i] / y[i];
            diagyinv[i] = 1.0 / diagy[i];
            diaglam[i] = s[i] / lam[i];
            diaglamyi[i] = diaglam[i] + diagyinv[i];
         }

         if (nCon < nVar)
         {
            //To-Do: Double check this in debug <---------------------------------------- !!!!!!!!!!!!
            // blam = dellam + dely./diagy - GG*(delx./diagx);
            // bb = [blam; delz];
            sum = 0.0;
            for (int j = 0; j < nCon; j++)
            {
               sum = 0.0;
               for (int i = 0; i < nVar; i++)
               {
                  sum -= GG[j * nVar + i] * (delx[i] * diagxinv[i]);
               }
               blam[j] = dellam[j] + dely[j] * diagyinv[j] - sum;
               bb[j] = blam[j];
            }
            bb[nCon] = delz;


            // Alam = spdiags(diaglamyi,0,m,m) + GG*spdiags(diagxinv,0,n,n)*GG';
            // AA = [Alam     a
            //       a'    -zet/z];
            for (int i = 0; i < nCon; i++)
            {
               for (int j = 0; j < nVar; j++)
               {
                  AA[i * nCon + j] = diaglamyi[i] + GG[i * nCon + j] * diagxinv[j] * GG[i * nCon +
                                                                                        j];
               }
               AA[nCon * nCon + i] = a[i];
               AA[3 * nCon + i] = a[i];
            }
            AA[4 * nCon] = -zet / z;
            // ----------------------------------------------------------------------------
            //solut = AA\bb --> solve linear system of equations using Gaussian elimination
            for (int k = 0; k < (nVar + 1) -1; k++)
            {
               for (int i = k + 1; i < (nVar + 1); i++)
               {
                  double fac = AA[i * (nVar + 1) + k] / AA[k * (nVar + 1) + k];
                  for (int j = k + 1; j < (nVar + 1); j++)
                  {
                     AA[i * (nVar + 1) + j] -= fac * AA[k * (nVar + 1) + j];
                  }
                  bb[i] -= fac * bb[k];
               }
            }
            // Back substitution
            solut[nVar] = bb[nVar] / AA[nVar * (nVar + 1) + nVar];
            for (int i = (nVar + 1) - 2; i >= 0; i--)
            {
               sum = bb[i];
               for (int j = i + 1; j < (nVar + 1); j++)
               {
                  sum -= AA[i * (nVar + 1) + j] * solut[j];
               }
               solut[i] = sum / AA[i * (nVar + 1) + i];
            }
            // ----------------------------------------------------------------------------

            //dlam = solut(1:nCon);
            for (int i = 0; i < nCon; i++)
            {
               dlam[i] = solut[i];
            }
            dz = solut[nCon];
            //dx = -(GG'*dlam)./diagx - delx./diagx;
            for (int i = 0; i < nVar; i++)
            {
               double sum = 0.0;
               for (int j = 0; j < nCon; j++)
               {
                  sum -= GG[j * nVar + i] * dlam[j];
               }
               dx[i] = -sum * diagxinv[i] - delx[i] * diagxinv[i];
            }
         }
         else
         {
            sum = 0.0;
            for (int i = 0; i < nCon; i++)
            {
               diaglamyiinv[i] = 1.0 / diaglamyi[i];
               dellamyi[i] = dellam[i] + dely[i] * diagyinv[i];
               // azz = zet/z + a'*(a./diaglamyi)
               sum += a[i] * (a[i] * diaglamyiinv[i]);
            }
            azz = zet / z + sum;
            // Axx = spdiags(diagx,0,nVar,nVar) + GG'*spdiags(diaglamyiinv,0,nCon,nCon)*GG;
            // AA = [Axx      axz
            //       axz'     azz];
            for (int i = 0; i < nVar; i++)
            {
               // Axx =  GG'*spdiags(diaglamyiinv,0,nCon,nCon);
               for (int k = 0; k < nCon; k++)
               {
                  Axx[i * nCon + k] = GG[k * nVar + i] * diaglamyiinv[k];
               }
               sum = 0.0;
               // axz = -GG'*(a./diaglamyi)
               for (int j = 0; j < nCon; j++)
               {
                  sum -= GG[j * nVar + i] * (a[j] * diaglamyiinv[j]);
               }
               axz[i] = sum;
            }
            //Assemble matrix AA
            for (int i = 0; i < (nVar + 1); i++)
            {
               for (int j = 0; j < (nVar + 1); j++)
               {
                  // AA = [Axx  .
                  //       .    .]
                  AA[i * (nVar + 1) + j] = 0.0;
                  if (i < nVar && j < nVar)
                  {
                     // Axx = Axx*GG
                     for (int k = 0; k < nCon; k++)
                     {
                        AA[i * (nVar + 1) + j] += Axx[i * nCon + k] * GG[k * nVar + j];
                     }
                     // Axx = Axx + spdiags(diagx,0,nVar,nVar)
                     if (i == j)
                     {
                        AA[i * (nVar + 1) + j] += diagx[j];
                     }
                  }
                  // AA = [Axx  axz
                  //       axz' azz]
                  else if (i < nVar && j == nVar)
                  {
                     AA[i * (nVar + 1) + j] = axz[i];
                  }
                  else if (i == nVar && j < nVar)
                  {
                     AA[i * (nVar + 1) + j] = axz[j];
                  }
                  else
                  {
                     AA[i * (nVar + 1) + j] = azz;
                  }
               }
            }
            // bb = [-bx'; -bz]
            // bx = delx - GG'*(dellamyi./diaglamyi)
            // bz = delz - a'*(dellamyi./diaglamyi)
            for (int i = 0; i < nVar; i++)
            {
               sum = 0.0;
               for (int j = 0; j < nCon; j++)
               {
                  sum += GG[j * nVar + i] * (dellamyi[j] * diaglamyiinv[j]);
               }
               bb[i] = -(delx[i] + sum);
            }
            sum = 0.0;
            for (int i = 0; i < nCon; i++)
            {
               sum += a[i] * (dellamyi[i] * diaglamyiinv[i]);
            }
            bb[nVar] = -(delz - sum);
            results << "Entering matrix solver" << std::endl;
            // ----------------------------------------------------------------------------
            //#ifdef MFEM_USE_LAPACK
            //solut = AA\bb --> solve linear system of equations using LAPACK
            int info;
            int nLAP = nVar + 1;
            int nrhs = 1;
            int lda = nLAP;
            int ldb = nLAP;
            int* ipiv = new int[nLAP];
            dgesv_(&nLAP, &nrhs, AA, &lda, ipiv, bb, &ldb, &info);
            delete[] ipiv;
            for (int i = 0; i < (nVar + 1); i++)
            {
               solut[i] = bb[i];
            }
            //#else
            //solut = AA\bb --> solve linear system of equations using Gaussian elimination
            // may cause problems with sparse matrices
            /*
            for (int k = 0; k < (nVar + 1) - 1; k++)
            {
               for (int i = k + 1; i < (nVar + 1); i++)
               {
                  double fac = AA[i * (nVar + 1) + k] / AA[k * (nVar + 1) + k];
                  for (int j = k + 1; j < (nVar + 1); j++)
                  {
                     AA[i * (nVar + 1) + j] -= fac * AA[k * (nVar + 1) + j];
                  }
                  bb[i] -= fac * bb[k];
               }
            }
            // Back substitution
            solut[nVar] = bb[nVar] / AA[nVar * (nVar + 1) + nVar];
            for (int i = (nVar + 1) - 2; i >= 0; i--)
            {
               sum = bb[i];
               for (int j = i + 1; j < (nVar + 1); j++)
               {
                  sum -= AA[i * (nVar + 1) + j] * solut[j];
               }
               solut[i] = sum / AA[i * (nVar + 1) + i];
            }
            #endif
            */
            // ----------------------------------------------------------------------------
            //dx = solut(1:nVar);
            for (int i = 0; i < nVar; i++)
            {
               dx[i] = solut[i];
            }
            dz = solut[nVar];
            //dlam = (GG*dx)./diaglamyi - dz*(a./diaglamyi) + dellamyi./diaglamyi;
            for (int i = 0; i < nCon; i++)
            {
               sum = 0.0;
               for (int j = 0; j < nVar; j++)
               {
                  sum += GG[i * nVar + j] * dx[j];
               }
               dlam[i] = sum * diaglamyiinv[i] - dz * (a[i] * diaglamyiinv[i]) + dellamyi[i] *
                         diaglamyiinv[i];
            }
         }

         for (int i = 0; i < nCon; i++)
         {
            dy[i] = -dely[i] * diagyinv[i] + dlam[i] * diagyinv[i];
            dmu[i] = -mu[i] + epsvecm[i] / y[i] - (mu[i] * dy[i]) / y[i];
            ds[i] = -s[i] + epsvecm[i] / lam[i] - (s[i] * dlam[i]) / lam[i];
            // xx = [y z lam xsi eta mu zet s]
            // dxx = [dy dz dlam dxsi deta dmu dzet ds]
            xx[i] = y[i];
            xx[nCon + 1 + i] = lam[i];
            xx[2 * nCon + 1 + 2 * nVar + i] = mu[i];
            xx[3 * nCon + 2 * nVar + 2 + i] = s[i];
            dxx[i] = dy[i];
            dxx[nCon + 1 + i] = dlam[i];
            dxx[2 * nCon + 1 + 2 * nVar + i] = dmu[i];
            dxx[3 * nCon + 2 * nVar + 2 + i] = ds[i];
         }
         xx[nCon] = z;
         xx[3 * nCon + 2 * nVar + 1] = zet;
         dxx[nCon] = dz;
         dxx[3 * nCon + 2 * nVar + 1] = dzet;
         for (int i = 0; i < nVar; i++)
         {
            dxsi[i] = -xsi[i] + epsvecn[i] / (x[i] - alfa[i]) - (xsi[i] * dx[i]) /
                      (x[i] - alfa[i]);
            if (dxsi[i] <= machineEpsilon)
            {
               dxsi[i] = machineEpsilon;
            }
            deta[i] = -eta[i] + epsvecn[i] / (beta[i] - x[i]) + (eta[i] * dx[i]) /
                      (beta[i] - x[i]);
            if (deta[i] <= machineEpsilon)
            {
               deta[i] = machineEpsilon;
            }
            xx[nCon + 1 + nCon + i] = xsi[i];
            xx[nCon + 1 + nCon + nVar + i] = eta[i];
            dxx[nCon + 1 + nCon + i] = dxsi[i];
            dxx[nCon + 1 + nCon + nVar + i] = deta[i];
         }
         dzet = -zet + epsi / z - zet * dz / z;

         stmxx = 0.0;
         for (int i = 0; i < (4 * nCon + 2 * nVar + 2); i++)
         {
            stepxx[i] = -1.01*dxx[i] / xx[i];
            if (stepxx[i] > stmxx)
            {
               stmxx = stepxx[i];
            }
         }
         stmalfa = 0.0;
         stmbeta = 0.0;
         for (int i = 0; i < nVar; i++)
         {
            stepalfa[i] = -1.01*dx[i] / (x[i] - alfa[i]);
            if (stepalfa[i] > stmalfa)
            {
               stmalfa = stepalfa[i];
            }
            stepbeta[i] = 1.01*dx[i] / (beta[i] - x[i]);
            if (stepbeta[i] > stmbeta)
            {
               stmbeta = stepbeta[i];
            }
         }
         stmalbe = std::max(stmalfa, stmbeta);
         stmalbexx = std::max(stmalbe, stmxx);
         stminv = std::max(stmalbexx, 1.0);
         steg = 1.0 / stminv;

         for (int i = 0; i < nVar; i++)
         {
            xold[i] = x[i];
            xsiold[i] = xsi[i];
            etaold[i] = eta[i];
         }
         for (int i = 0; i < nCon; i++)
         {
            yold[i] = y[i];
            lamold[i] = lam[i];
            muold[i] = mu[i];
            sold[i] = s[i];
         }
         zold = z;
         zetold = zet;

         itto = 0;
         resinew = 2.0 * residunorm;

         results << "Entering Loop 50" << std::endl;
         while (resinew > residunorm && itto < 50)
         {
            itto++;

            for (int i = 0; i < nCon; ++i)
            {
               y[i] = yold[i] + steg * dy[i];
               lam[i] = lamold[i] + steg * dlam[i];
               mu[i] = muold[i] + steg * dmu[i];
               s[i] = sold[i] + steg * ds[i];
            }

            for (int i = 0; i < nVar; ++i)
            {
               x[i] = xold[i] + steg * dx[i];
               xsi[i] = xsiold[i] + steg * dxsi[i];
               eta[i] = etaold[i] + steg * deta[i];

               ux1[i] = upp[i] - x[i];
               xl1[i] = x[i] - low[i];
               ux2[i] = ux1[i] * ux1[i];
               xl2[i] = xl1[i] * xl1[i];
               uxinv1[i] = 1.0 / ux1[i];
               xlinv1[i] = 1.0 / xl1[i];
               // plam & qlam
               plam[i] = p0[i];
               qlam[i] = q0[i];
               for (int j = 0; j < nCon; j++)
               {
                  plam[i] += P[j * nVar + i] * lam[j];
                  qlam[i] += Q[j * nVar + i] * lam[j];
               }
               dpsidx[i] = plam[i] * uxinv1[i] * uxinv1[i] - qlam[i] * xlinv1[i] * xlinv1[i];
               rex[i] = dpsidx[i] - xsi[i] + eta[i];
               rez -= a[i] * lam[i];
               rexsi[i] = xsi[i] * (x[i] - alfa[i]) - epsvecn[i];
               if (rexsi[i] <= machineEpsilon)
               {
                  rexsi[i] = machineEpsilon;
               }
               reeta[i] = eta[i] * (beta[i] - x[i]) - epsvecn[i];
               if (reeta[i] <= machineEpsilon)
               {
                  reeta[i] = machineEpsilon;
               }

               residu1[i] = rex[i];
               residu2[nCon + i] = rexsi[i];
               residu2[nCon + nVar + i] = reeta[i];
            }
            z = zold + steg * dz;
            zet = zetold + steg * dzet;

            // gvec = P/ux + Q/xl
            for (int i = 0; i < nCon; i++)
            {
               gvec[i] = 0.0;
               for (int j = 0; j < nVar; j++)
               {
                  gvec[i] += P[i * nVar + j] * uxinv1[j] + Q[i * nVar + j] * xlinv1[j];
               }
               rey[i] = c[i] + d[i] * y[i] - mu[i] - lam[i];
               relam[i] = gvec[i] - a[i] * z - y[i] + s[i] - b[i];
               remu[i] = mu[i] * y[i] - epsvecm[i];
               res[i] = lam[i] * s[i] - epsvecm[i];

               residu2[i] = relam[i];
               residu1[nVar + i] = rey[i];
               residu2[nCon + 2 * nVar + i] = remu[i];
               residu2[2 * nVar + 2 * nCon + 1 + i] = res[i];
            }
            rez = a0 - zet;
            rezet = zet * z - epsi;
            residu1[nVar + nCon] = rez;
            residu2[2 * nVar + 2 * nCon] = rezet;

            // Concatenate the residuals
            for (int i = 0; i < (nVar + nCon + 1); i++)
            {
               residu[i] = residu1[i];
            }
            for (int i = 0; i < (2 * nVar + 3 * nCon + 1); i++)
            {
               residu[nVar + nCon + 1 + i] = residu2[i];
            }
            // New residual
            sum = 0.0;
            for (int i = 0; i < (3 * nVar + 4 * nCon + 2); i++)
            {
               sum += residu[i] * residu[i];
            }
            // Norm of the residual
            resinew = std::sqrt(sum);
            steg = steg / 2.0;
         }
         residunorm = resinew;
         residumax = 0.0;
         for (int i = 0; i < (3 * nVar + 4 * nCon + 2); i++)
         {
            residumax = std::max(residumax, std::abs(residu[i]));
         }
         steg = steg * 2.0;
      }


      if (ittt > 198)
      {
         //printf("Warning: Maximum number of iterations reached in subsolv.\n");
      }
      epsi = 0.1 * epsi;
   }

   // Update new values
   for (int i = 0; i < nVar; i++)
   {
      xmma[i] = x[i];
      xsimma[i] = xsi[i];
      etamma[i] = eta[i];
   }
   for (int i = 0; i < nCon; i++)
   {
      ymma[i] = y[i];
      lamma[i] = lam[i];
      mumma[i] = mu[i];
      smma[i] = s[i];
   }
   *zmma = z;
   *zetmma = zet;

   delete[] epsvecn;
   delete[] epsvecm;
   delete[] x;
   delete[] y;
   delete[] lam;
   delete[] xsi;
   delete[] eta;
   delete[] mu;
   delete[] s;
   delete[] ux1;
   delete[] ux2;
   delete[] ux3;
   delete[] xl1;
   delete[] xl2;
   delete[] xl3;
   delete[] uxinv1;
   delete[] xlinv1;
   delete[] plam;
   delete[] qlam;
   delete[] gvec;
   delete[] dpsidx;
   delete[] rex;
   delete[] rey;
   delete[] relam;
   delete[] rexsi;
   delete[] reeta;
   delete[] remu;
   delete[] res;
   delete[] residu1;
   delete[] residu2;
   delete[] residu;
   delete[] GG;
   delete[] Puxinv;
   delete[] Qxlinv;
   delete[] delx;
   delete[] dely;
   delete[] dellam;
   delete[] dellamyi;
   delete[] diagx;
   delete[] diagxinv;
   delete[] diagy;
   delete[] diagyinv;
   delete[] diaglam;
   delete[] diaglamyi;
   delete[] diaglamyiinv;
   delete[] blam;
   delete[] bb;
   delete[] Alam;
   delete[] AA;
   delete[] solut;
   delete[] dlam;
   delete[] dx;
   delete[] dy;
   delete[] dxsi;
   delete[] deta;
   delete[] dmu;
   delete[] Axx;
   delete[] axz;
   delete[] ds;
   delete[] xx;
   delete[] dxx;
   delete[] stepxx;
   delete[] stepalfa;
   delete[] stepbeta;
   delete[] xold;
   delete[] yold;
   delete[] lamold;
   delete[] xsiold;
   delete[] etaold;
   delete[] muold;
   delete[] sold;
}

void MMA::kktcheck(int nCon, int nVar, double* x, double* y, double z,
                   double* lam, double* xsi, double* eta,
                   double* mu, double zet, double* s,
                   double* xmin, double* xmax,
                   double* dfdx, double* gx, double* dgdx,
                   double a0, double* a, const double* c, double* d,
                   double* kktnorm)
{
   double* rex = new double[nVar];
   double* rey = new double[nCon];
   double rez, rezet;
   double* relam = new double[nCon];
   double* rexsi = new double[nVar];
   double* reeta = new double[nVar];
   double* remu = new double[nCon];
   double* res = new double[nCon];
   double* residu1 = new double[nVar + nCon + 1];
   double* residu2 = new double[2 * nVar + 3 * nCon + 2];
   double* residu = new double[3 * nVar + 4 * nCon + 2];
   double residunorm, residumax;
   double sum = 0.0;
   double* sum1 = new double[nVar];

   for (int i = 0; i < nVar; i++)
   {
      sum1[i] = 0.0;
      for (int j = 0; j < nCon; j++)
      {
         sum1[i] += dgdx[j * nVar + i] * lam[j];
      }
   }
   for (int i = 0; i < nVar; i++)
   {
      rex[i] = dfdx[i] + sum1[i] - xsi[i] + eta[i];
      rexsi[i] = xsi[i] * (x[i] - xmin[i]);
      reeta[i] = eta[i] * (xmax[i] - x[i]);
   }
   for (int i = 0; i < nCon; i++)
   {
      rey[i] = c[i] + d[i] * y[i] - mu[i] - lam[i];
      relam[i] = gx[i] - a[i] * z - y[i] + s[i];
      remu[i] = mu[i] * y[i];
      res[i] = lam[i] * s[i];
   }
   sum = 0.0;
   for (int i = 0; i < nCon; i++)
   {
      sum += a[i] * lam[i];
   }
   rez = a0 - zet - sum;
   rezet = zet * z;
   //---------------------------------------------------------------------

   for (int i = 0; i < nVar; i++)
   {
      residu1[i] = rex[i];
      residu2[nCon + i] = rexsi[i];
      residu2[nCon + nVar + i] = reeta[i];
   }

   for (int i = 0; i < nCon; i++)
   {
      residu2[i] = relam[i];
      residu1[nVar + i] = rey[i];
      residu2[nCon + 2 * nVar + i] = remu[i];
      residu2[2 * nVar + 2 * nCon + 1 + i] = res[i];
   }
   residu1[nVar + nCon] = rez;
   residu2[2 * nVar + 2 * nCon] = rezet;

   // Concatenate the residuals
   for (int i = 0; i < (nVar + nCon + 1); i++)
   {
      residu[i] = residu1[i];
   }
   for (int i = 0; i < (2 * nVar + 3 * nCon + 1); i++)
   {
      residu[nVar + nCon + 1 + i] = residu2[i];
   }
   //Get vector product and maximum absolute value
   residunorm = 0.0;
   residumax = 0.0;
   for (int i = 0; i < (3 * nVar + 4 * nCon + 2); i++)
   {
      residunorm += residu[i] * residu[i];
      residumax = std::max(residumax, std::abs(residu[i]));
   }
   // Norm of the residual
   residunorm = std::sqrt(residunorm);
   *kktnorm = residunorm;

   delete[] rex;
   delete[] rey;
   delete[] relam ;
   delete[] rexsi;
   delete[] reeta;
   delete[] remu;
   delete[] res;
   delete[] residu1;
   delete[] residu2;
   delete[] residu;
}



void MMA::Restart(double* xo1, double* xo2, double* xo3, double* upp,
                  double* low, int length, int iter)
{
   std::ofstream mma;
   mma.open("Restart.dat");
   //print results
   mma << iter << "\n";
   for (int i = 0; i < length; i++)
   {
      mma << xo1[i] << "\n";
   }
   for (int i = 0; i < length; i++)
   {
      mma << xo2[i] << "\n";
   }
   for (int i = 0; i < length; i++)
   {
      mma << xo3[i] << "\n";
   }
   for (int i = 0; i < length; i++)
   {
      mma << upp[i] << "\n";
   }
   for (int i = 0; i < length; i++)
   {
      mma << low[i] << "\n";
   }
   mma.close();
}
} // end mma namespace
