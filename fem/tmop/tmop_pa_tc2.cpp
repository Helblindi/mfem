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

using namespace mfem;

namespace mfem
{

template<int T_Q1D = 0>
void TMOP_TcIdealShapeUnitSize_2D(const int NE,
                                  const ConstDeviceMatrix &W,
                                  DeviceTensor<5> &J,
                                  const int q1d)
{
   constexpr int NBZ = 1;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int DIM = 2;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            kernels::Set(DIM,DIM, 1.0, &W(0,0), &J(0,0,qx,qy,e));
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0, int T_MAX = 4>
void TMOP_TcIdealShapeGivenSize_2D(const int NE,
                                   const double detW,
                                   const ConstDeviceMatrix &B,
                                   const ConstDeviceMatrix &G,
                                   const ConstDeviceMatrix &W,
                                   const DeviceTensor<4, const double> &X,
                                   DeviceTensor<5> &J,
                                   const int d1d,
                                   const int q1d,
                                   const int max)
{
   constexpr int NBZ = 1;
   const int Q1D = T_Q1D ? T_Q1D : q1d;

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int DIM = 2;
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;

      constexpr int NBZ = 1;
      constexpr int MQ1 = T_Q1D ? T_Q1D : T_MAX;
      constexpr int MD1 = T_D1D ? T_D1D : T_MAX;

      MFEM_SHARED double BG[2][MQ1*MD1];
      MFEM_SHARED double XY[2][NBZ][MD1*MD1];
      MFEM_SHARED double DQ[4][NBZ][MD1*MQ1];
      MFEM_SHARED double QQ[4][NBZ][MQ1*MQ1];

      kernels::internal::LoadX<MD1,NBZ>(e,D1D,X,XY);
      kernels::internal::LoadBG<MD1,MQ1>(D1D,Q1D,B,G,BG);

      kernels::internal::GradX<MD1,MQ1,NBZ>(D1D,Q1D,BG,XY,DQ);
      kernels::internal::GradY<MD1,MQ1,NBZ>(D1D,Q1D,BG,DQ,QQ);

      MFEM_FOREACH_THREAD(qy,y,Q1D)
      {
         MFEM_FOREACH_THREAD(qx,x,Q1D)
         {
            double Jtr[4];
            const double *Wid = &W(0,0);
            kernels::internal::PullGrad<MQ1,NBZ>(Q1D,qx,qy,QQ,Jtr);
            const double detJ = kernels::Det<2>(Jtr);
            const double alpha = std::pow(detJ/detW,1./2);
            kernels::Set(DIM,DIM,alpha,Wid,&J(0,0,qx,qy,e));
         }
      }
   });
}

template<> bool
TargetConstructor::ComputeAllElementTargets<2>(const FiniteElementSpace &fes,
                                               const IntegrationRule &ir,
                                               const Vector &,
                                               DenseTensor &Jtr) const
{
   MFEM_ASSERT(target_type == IDEAL_SHAPE_UNIT_SIZE || nodes != nullptr, "");
   const Mesh *mesh = fes.GetMesh();
   const int NE = mesh->GetNE();
   // Quick return for empty processors:
   if (NE == 0) { return true; }
   const int dim = mesh->Dimension();
   MFEM_VERIFY(mesh->GetNumGeometries(dim) <= 1,
               "mixed meshes are not supported");
   MFEM_VERIFY(!fes.IsVariableOrder(), "variable orders are not supported");
   const FiniteElement &fe = *fes.GetFE(0);
   MFEM_VERIFY(fe.GetGeomType() == Geometry::SQUARE, "");
   const DenseMatrix &w = Geometries.GetGeomToPerfGeomJac(Geometry::SQUARE);
   const double detW = w.Det();
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   const DofToQuad &maps = fe.GetDofToQuad(ir, mode);
   const int d = maps.ndof, q = maps.nqpt;

   constexpr int DIM = 2;
   const auto W = Reshape(w.Read(), DIM,DIM);
   const auto B = Reshape(maps.B.Read(), q, d);
   const auto G = Reshape(maps.G.Read(), q, d);
   auto J = Reshape(Jtr.Write(), DIM,DIM, q,q, NE);

   switch (target_type)
   {
      case IDEAL_SHAPE_UNIT_SIZE: // Jtr(i) = Wideal;
      {
         decltype(&TMOP_TcIdealShapeUnitSize_2D<>) ker =
            TMOP_TcIdealShapeUnitSize_2D;

         if (q==2) { ker = TMOP_TcIdealShapeUnitSize_2D<2>; }
         if (q==3) { ker = TMOP_TcIdealShapeUnitSize_2D<3>; }
         if (q==4) { ker = TMOP_TcIdealShapeUnitSize_2D<4>; }
         if (q==5) { ker = TMOP_TcIdealShapeUnitSize_2D<5>; }
         if (q==6) { ker = TMOP_TcIdealShapeUnitSize_2D<6>; }

         ker(NE,W,J,q);
         return true;
      }
      case IDEAL_SHAPE_EQUAL_SIZE: return false;
      case IDEAL_SHAPE_GIVEN_SIZE:
      {
         MFEM_VERIFY(nodes, "");
         const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
         const Operator *R = fes.GetElementRestriction(ordering);
         Vector x(R->Height(), Device::GetDeviceMemoryType());
         x.UseDevice(true);
         R->Mult(*nodes, x);
         MFEM_ASSERT(nodes->FESpace()->GetVDim() == 2, "");
         const auto X = Reshape(x.Read(), d, d, DIM, NE);

         decltype(&TMOP_TcIdealShapeGivenSize_2D<>) ker =
            TMOP_TcIdealShapeGivenSize_2D;

         if (d==2 && q==2) { ker = TMOP_TcIdealShapeGivenSize_2D<2,2>; }
         if (d==2 && q==3) { ker = TMOP_TcIdealShapeGivenSize_2D<2,3>; }
         if (d==2 && q==4) { ker = TMOP_TcIdealShapeGivenSize_2D<2,4>; }
         if (d==2 && q==5) { ker = TMOP_TcIdealShapeGivenSize_2D<2,5>; }
         if (d==2 && q==6) { ker = TMOP_TcIdealShapeGivenSize_2D<2,6>; }

         if (d==3 && q==3) { ker = TMOP_TcIdealShapeGivenSize_2D<3,3>; }
         if (d==3 && q==4) { ker = TMOP_TcIdealShapeGivenSize_2D<3,4>; }
         if (d==3 && q==5) { ker = TMOP_TcIdealShapeGivenSize_2D<3,5>; }
         if (d==3 && q==6) { ker = TMOP_TcIdealShapeGivenSize_2D<3,6>; }

         if (d==4 && q==4) { ker = TMOP_TcIdealShapeGivenSize_2D<4,4>; }
         if (d==4 && q==5) { ker = TMOP_TcIdealShapeGivenSize_2D<4,5>; }
         if (d==4 && q==6) { ker = TMOP_TcIdealShapeGivenSize_2D<4,6>; }

         if (d==5 && q==5) { ker = TMOP_TcIdealShapeGivenSize_2D<5,5>; }
         if (d==5 && q==6) { ker = TMOP_TcIdealShapeGivenSize_2D<5,6>; }

         ker(NE,detW,B,G,W,X,J,d,q,4);
         return true;
      }
      case GIVEN_SHAPE_AND_SIZE: return false;
      default: return false;
   }
   return false;
}

} // namespace mfem
