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

#include "tmop_pa.hpp"

namespace mfem
{

struct MetricTMOP_302 : MetricTMOPKer3D
{
   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int qz, const int e,
                  const double weight, double *Jrt, double *Jpr,
                  const double (&Jpt)[9], const double *w,
                  const DeviceTensor<8> &H) const override
   {
      double B[9];
      double         dI1b[9],          ddI1b[9];
      double dI2[9], dI2b[9], ddI2[9], ddI2b[9];
      double         dI3b[9];// = Jrt;
      // (dI2b*dI1b + dI1b*dI2b)/9 + (I1b/9)*ddI2b + (I2b/9)*ddI1b
      kernels::InvariantsEvaluator3D ie
      (Args().J(Jpt).B(B).dI1b(dI1b).ddI1b(ddI1b)
       .dI2(dI2).dI2b(dI2b).ddI2(ddI2).ddI2b(ddI2b).dI3b(dI3b));

      const double c1 = weight/9.;
      const double I1b = ie.Get_I1b();
      const double I2b = ie.Get_I2b();
      ConstDeviceMatrix di1b(ie.Get_dI1b(),DIM,DIM);
      ConstDeviceMatrix di2b(ie.Get_dI2b(),DIM,DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i,j),DIM,DIM);
            ConstDeviceMatrix ddi2b(ie.Get_ddI2b(i,j),DIM,DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const double dp =
                     (di2b(r,c)*di1b(i,j) + di1b(r,c)*di2b(i,j))
                     + ddi2b(r,c)*I1b
                     + ddi1b(r,c)*I2b;
                  H(r,c,i,j,qx,qy,qz,e) = c1 * dp;
               }
            }
         }
      }
   }
};

struct MetricTMOP_303 : MetricTMOPKer3D
{
   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int qz, const int e,
                  const double weight, double *Jrt, double *Jpr,
                  const double (&Jpt)[9], const double *w,
                  const DeviceTensor<8> &H) const override
   {
      double B[9];
      double         dI1b[9], ddI1[9], ddI1b[9];
      double dI2[9], dI2b[9], ddI2[9], ddI2b[9];
      double        *dI3b = Jrt,      *ddI3b = Jpr;

      // ddI1b/3
      kernels::InvariantsEvaluator3D ie
      (Args().J(Jpt).B(B)
       .dI1b(dI1b).ddI1(ddI1).ddI1b(ddI1b)
       .dI2(dI2).dI2b(dI2b).ddI2(ddI2).ddI2b(ddI2b)
       .dI3b(dI3b).ddI3b(ddI3b));

      const double c1 = weight/3.;
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i,j),DIM,DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const double dp = ddi1b(r,c);
                  H(r,c,i,j,qx,qy,qz,e) = c1 * dp;
               }
            }
         }
      }
   }
};

struct MetricTMOP_315 : MetricTMOPKer3D
{
   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int qz, const int e,
                  const double weight, double *Jrt, double *Jpr,
                  const double (&Jpt)[9], const double *w,
                  const DeviceTensor<8> &H) const override
   {
      double *dI3b = Jrt, *ddI3b = Jpr;
      // 2*(dI3b x dI3b) + 2*(I3b - 1)*ddI3b
      kernels::InvariantsEvaluator3D ie(Args().J(Jpt).dI3b(dI3b).ddI3b(ddI3b));
      double sign_detJ;
      const double I3b = ie.Get_I3b(sign_detJ);
      ConstDeviceMatrix di3b(ie.Get_dI3b(sign_detJ),DIM,DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi3b(ie.Get_ddI3b(i,j),DIM,DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const double dp =
                     2.0 * weight * (I3b - 1.0) * ddi3b(r,c) +
                     2.0 * weight * di3b(r,c) * di3b(i,j);
                  H(r,c,i,j,qx,qy,qz,e) = dp;
               }
            }
         }
      }
   }
};

struct MetricTMOP_318 : MetricTMOPKer3D
{
   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int qz, const int e,
                  const double weight, double *Jrt, double *Jpr,
                  const double (&Jpt)[9], const double *w,
                  const DeviceTensor<8> &H) const override
   {
      double *dI3b = Jrt, *ddI3b = Jpr;
      // dP_318 = (I3b - 1/I3b^3)*ddI3b + (1 + 3/I3b^4)*(dI3b x dI3b)
      // Uses the I3b form, as dI3 and ddI3 were not implemented at the time
      kernels::InvariantsEvaluator3D ie(Args().J(Jpt).dI3b(dI3b).ddI3b(ddI3b));
      double sign_detJ;
      const double I3b = ie.Get_I3b(sign_detJ);
      ConstDeviceMatrix di3b(ie.Get_dI3b(sign_detJ),DIM,DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi3b(ie.Get_ddI3b(i,j),DIM,DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const double dp =
                     weight * (I3b - 1.0/(I3b*I3b*I3b)) * ddi3b(r,c) +
                     weight * (1.0 + 3.0/(I3b*I3b*I3b*I3b)) * di3b(r,c)*di3b(i,j);
                  H(r,c,i,j,qx,qy,qz,e) = dp;
               }
            }
         }
      }
   }
};

struct MetricTMOP_321 : MetricTMOPKer3D
{
   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int qz, const int e,
                  const double weight, double *Jrt, double *Jpr,
                  const double (&Jpt)[9], const double *w,
                  const DeviceTensor<8> &H) const override
   {
      double B[9];
      double         dI1b[9], ddI1[9], ddI1b[9];
      double dI2[9], dI2b[9], ddI2[9], ddI2b[9];
      double        *dI3b = Jrt,      *ddI3b = Jpr;

      // ddI1 + (-2/I3b^3)*(dI2 x dI3b + dI3b x dI2)
      //      + (1/I3)*ddI2
      //      + (6*I2/I3b^4)*(dI3b x dI3b)
      //      + (-2*I2/I3b^3)*ddI3b
      kernels::InvariantsEvaluator3D ie
      (Args()
       .J(Jpt).B(B)
       .dI1b(dI1b).ddI1(ddI1).ddI1b(ddI1b)
       .dI2(dI2).dI2b(dI2b).ddI2(ddI2).ddI2b(ddI2b)
       .dI3b(dI3b).ddI3b(ddI3b));
      double sign_detJ;
      const double I2 = ie.Get_I2();
      const double I3b = ie.Get_I3b(sign_detJ);
      ConstDeviceMatrix di2(ie.Get_dI2(),DIM,DIM);
      ConstDeviceMatrix di3b(ie.Get_dI3b(sign_detJ),DIM,DIM);
      const double c0 = 1.0/I3b;
      const double c1 = weight*c0*c0;
      const double c2 = -2*c0*c1;
      const double c3 = c2*I2;
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1(ie.Get_ddI1(i,j),DIM,DIM);
            ConstDeviceMatrix ddi2(ie.Get_ddI2(i,j),DIM,DIM);
            ConstDeviceMatrix ddi3b(ie.Get_ddI3b(i,j),DIM,DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const double dp =
                     weight * ddi1(r,c)
                     + c1 * ddi2(r,c)
                     + c3 * ddi3b(r,c)
                     + c2 * ((di2(r,c)*di3b(i,j) + di3b(r,c)*di2(i,j)))
                     -3*c0*c3 * di3b(r,c)*di3b(i,j);
                  H(r,c,i,j,qx,qy,qz,e) = dp;
               }
            }
         }
      }
   }
};

struct MetricTMOP_332 : MetricTMOPKer3D
{
   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int qz, const int e,
                  const double weight, double *Jrt, double *Jpr,
                  const double (&Jpt)[9], const double *w,
                  const DeviceTensor<8> &H) const override
   {
      double B[9];
      double         dI1b[9], /*ddI1[9],*/ ddI1b[9];
      double dI2[9], dI2b[9], ddI2[9], ddI2b[9];
      double        *dI3b=Jrt,        *ddI3b=Jpr;
      // w0 H_302 + w1 H_315
      kernels::InvariantsEvaluator3D ie
      (Args().J(Jpt).B(B)
       .dI1b(dI1b).ddI1b(ddI1b)
       .dI2(dI2).dI2b(dI2b).ddI2(ddI2).ddI2b(ddI2b)
       .dI3b(dI3b).ddI3b(ddI3b));
      double sign_detJ;
      const double c1 = weight / 9.0;
      const double I1b = ie.Get_I1b();
      const double I2b = ie.Get_I2b();
      const double I3b = ie.Get_I3b(sign_detJ);
      ConstDeviceMatrix di1b(ie.Get_dI1b(),DIM,DIM);
      ConstDeviceMatrix di2b(ie.Get_dI2b(),DIM,DIM);
      ConstDeviceMatrix di3b(ie.Get_dI3b(sign_detJ),DIM,DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i,j),DIM,DIM);
            ConstDeviceMatrix ddi2b(ie.Get_ddI2b(i,j),DIM,DIM);
            ConstDeviceMatrix ddi3b(ie.Get_ddI3b(i,j),DIM,DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const double dp_302 =
                     (di2b(r,c)*di1b(i,j) + di1b(r,c)*di2b(i,j))
                     + ddi2b(r,c)*I1b
                     + ddi1b(r,c)*I2b;
                  const double dp_315 =
                     2.0 * weight * (I3b - 1.0) * ddi3b(r,c) +
                     2.0 * weight * di3b(r,c) * di3b(i,j);
                  H(r,c,i,j,qx,qy,qz,e) =
                     w[0] * c1 * dp_302 + w[1] * dp_315;
               }
            }
         }
      }
   }
};

struct MetricTMOP_338 : MetricTMOPKer3D
{
   MFEM_HOST_DEVICE
   void AssembleH(const int qx, const int qy, const int qz, const int e,
                  const double weight, double *Jrt, double *Jpr,
                  const double (&Jpt)[9], const double *w,
                  const DeviceTensor<8> &H) const override
   {
      double B[9];
      double         dI1b[9],          ddI1b[9];
      double dI2[9], dI2b[9], ddI2[9], ddI2b[9];
      double        *dI3b=Jrt,        *ddI3b=Jpr;
      // w0 H_302 + w1 H_318
      kernels::InvariantsEvaluator3D ie
      (Args().J(Jpt).B(B)
       .dI1b(dI1b).ddI1b(ddI1b)
       .dI2(dI2).dI2b(dI2b).ddI2(ddI2).ddI2b(ddI2b)
       .dI3b(dI3b).ddI3b(ddI3b));
      double sign_detJ;
      const double c1 = weight/9.;
      const double I1b = ie.Get_I1b();
      const double I2b = ie.Get_I2b();
      const double I3b = ie.Get_I3b(sign_detJ);
      ConstDeviceMatrix di1b(ie.Get_dI1b(),DIM,DIM);
      ConstDeviceMatrix di2b(ie.Get_dI2b(),DIM,DIM);
      ConstDeviceMatrix di3b(ie.Get_dI3b(sign_detJ),DIM,DIM);
      for (int i = 0; i < DIM; i++)
      {
         for (int j = 0; j < DIM; j++)
         {
            ConstDeviceMatrix ddi1b(ie.Get_ddI1b(i,j),DIM,DIM);
            ConstDeviceMatrix ddi2b(ie.Get_ddI2b(i,j),DIM,DIM);
            ConstDeviceMatrix ddi3b(ie.Get_ddI3b(i,j),DIM,DIM);
            for (int r = 0; r < DIM; r++)
            {
               for (int c = 0; c < DIM; c++)
               {
                  const double dp_302 =
                     (di2b(r,c)*di1b(i,j) + di1b(r,c)*di2b(i,j))
                     + ddi2b(r,c)*I1b
                     + ddi1b(r,c)*I2b;
                  const double dp_318 =
                     weight * (I3b - 1.0/(I3b*I3b*I3b)) * ddi3b(r,c) +
                     weight * (1.0 + 3.0/(I3b*I3b*I3b*I3b)) * di3b(r,c)*di3b(i,j);
                  H(r,c,i,j,qx,qy,qz,e) =
                     w[0] * c1 * dp_302 + w[1] * dp_318;
               }
            }
         }
      }
   }
};

struct TMOP_SetupGradPA_3D
{
   const mfem::TMOP_Integrator *ti; // not owned
   const Vector &x;

   TMOP_SetupGradPA_3D(const mfem::TMOP_Integrator *ti,
                       const Vector &x): ti(ti), x(x) { }

   int d() const { return ti->PA.maps->ndof; }

   int q() const { return ti->PA.maps->nqpt; }

   template<typename METRIC, int T_D1D, int T_Q1D, int T_MAX = 4>
   void operator()()
   {
      constexpr int DIM = 3;
      const double metric_normal = ti->metric_normal;

      Array<double> mp;
      if (auto m = dynamic_cast<TMOP_Combo_QualityMetric *>(ti->metric))
      {
         m->GetWeights(mp);
      }
      const double *w = mp.Read();

      const int d = ti->PA.maps->ndof, q = ti->PA.maps->nqpt, d1d = d, q1d = q;

      const int NE = ti->PA.ne;

      const ConstDeviceCube &W = Reshape(ti->PA.ir->GetWeights().Read(), q,q,q);
      const ConstDeviceMatrix &B = Reshape(ti->PA.maps->B.Read(), q,d);
      const ConstDeviceMatrix &G = Reshape(ti->PA.maps->G.Read(), q,d);
      const DeviceTensor<6, const double> &J =
         Reshape(ti->PA.Jtr.Read(), DIM,DIM, q,q,q, NE);
      const DeviceTensor<5, const double> &X = Reshape(x.Read(), d,d,d, DIM, NE);
      const DeviceTensor<8> &H =
         Reshape(ti->PA.H.Write(), DIM,DIM, DIM,DIM, q,q,q, NE);

      const int Q1D = T_Q1D ? T_Q1D : q1d;

      mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
      {
         const int D1D = T_D1D ? T_D1D : d1d;
         const int Q1D = T_Q1D ? T_Q1D : q1d;
         constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
         constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

         MFEM_SHARED double s_BG[2][MQ1*MD1];
         MFEM_SHARED double s_DDD[3][MD1*MD1*MD1];
         MFEM_SHARED double s_DDQ[9][MD1*MD1*MQ1];
         MFEM_SHARED double s_DQQ[9][MD1*MQ1*MQ1];
         MFEM_SHARED double s_QQQ[9][MQ1*MQ1*MQ1];

         kernels::internal::LoadX<MD1>(e,D1D,X,s_DDD);
         kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,B,G,s_BG);

         kernels::internal::GradX<MD1,MQ1>(D1D,Q1D,s_BG,s_DDD,s_DDQ);
         kernels::internal::GradY<MD1,MQ1>(D1D,Q1D,s_BG,s_DDQ,s_DQQ);
         kernels::internal::GradZ<MD1,MQ1>(D1D,Q1D,s_BG,s_DQQ,s_QQQ);

         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  const double *Jtr = &J(0,0,qx,qy,qz,e);
                  const double detJtr = kernels::Det<3>(Jtr);
                  const double weight = metric_normal * W(qx,qy,qz) * detJtr;

                  // Jrt = Jtr^{-1}
                  double Jrt[9];
                  kernels::CalcInverse<3>(Jtr, Jrt);

                  // Jpr = X^T.DSh
                  double Jpr[9];
                  kernels::internal::PullGrad<MQ1>(Q1D,qx,qy,qz, s_QQQ, Jpr);

                  // Jpt = X^T . DS = (X^T.DSh) . Jrt = Jpr . Jrt
                  double Jpt[9];
                  kernels::Mult(3,3,3, Jpr, Jrt, Jpt);

                  METRIC{}.AssembleH(qx,qy,qz,e, weight,Jrt,Jpr,Jpt, w,H);
               } // qx
            } // qy
         } // qz
      });
   }
};

template<typename M /* metric */, typename K /* kernel */>
static void Launch(K &ker, const int d, const int q)
{
   if (d==2 && q==2) { return ker.template operator()<M,2,2>(); }
   if (d==2 && q==3) { return ker.template operator()<M,2,3>(); }
   if (d==2 && q==4) { return ker.template operator()<M,2,4>(); }
   if (d==2 && q==5) { return ker.template operator()<M,2,5>(); }
   if (d==2 && q==6) { return ker.template operator()<M,2,6>(); }

   if (d==3 && q==3) { return ker.template operator()<M,3,3>(); }
   if (d==3 && q==4) { return ker.template operator()<M,3,4>(); }
   if (d==3 && q==5) { return ker.template operator()<M,3,5>(); }
   if (d==3 && q==6) { return ker.template operator()<M,3,6>(); }

   if (d==4 && q==4) { return ker.template operator()<M,4,4>(); }
   if (d==4 && q==5) { return ker.template operator()<M,4,5>(); }
   if (d==4 && q==6) { return ker.template operator()<M,4,6>(); }

   if (d==5 && q==5) { return ker.template operator()<M,5,5>(); }
   if (d==5 && q==6) { return ker.template operator()<M,5,6>(); }

   ker.template operator()<M,0,0>();
}

} // namespace mfem
