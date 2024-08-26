// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "lpq-common.hpp"

using namespace std;
using namespace mfem;

namespace lpq_common
{

int NDIGITS = 20;
int MG_MAX_ITER = 10;
real_t MG_REL_TOL = 1e-10;

int dim = 0;
int space_dim = 0;
real_t freq = 1.0;
real_t kappa = 1.0;

// Custom monitor that prints a csv-formatted file
DataMonitor::DataMonitor(string file_name, int ndigits)
   : os(file_name),
     precision(ndigits)
{
   if (Mpi::Root())
   {
      mfem::out << "Saving iterations into: " << file_name << endl;
   }
   os << "it,res,sol" << endl;
   os << fixed << setprecision(precision);
}

void DataMonitor::MonitorResidual(int it, real_t norm, const Vector &x,
                                  bool final)
{
   os << it << "," << norm << ",";
}

void DataMonitor::MonitorSolution(int it, real_t norm, const Vector &x,
                                  bool final)
{
   os << norm << endl;
}

// L(p,q) general geometric multigrid method, derived from GeometricMultigrid
LpqGeometricMultigrid::LpqGeometricMultigrid(
   ParFiniteElementSpaceHierarchy& fes_hierarchy,
   Array<int>& ess_bdr,
   IntegratorType it,
   SolverType st,
   real_t p_order,
   real_t q_order)
   : GeometricMultigrid(fes_hierarchy, ess_bdr),
     integrator_type(it),
     solver_type(st),
     p_order(p_order),
     q_order(q_order),
     coarse_pc(nullptr),
     one(1.0)
{
   ConstructCoarseOperatorAndSolver(fes_hierarchy.GetFESpaceAtLevel(0));
   for (int l = 1; l < fes_hierarchy.GetNumLevels(); ++l)
   {
      ConstructOperatorAndSmoother(fes_hierarchy.GetFESpaceAtLevel(l), l);
   }
}

void LpqGeometricMultigrid::ConstructCoarseOperatorAndSolver(
   ParFiniteElementSpace& coarse_fespace)
{
   ConstructBilinearForm(coarse_fespace);

   HypreParMatrix* coarse_mat = new HypreParMatrix();
   bfs[0]->FormSystemMatrix(*essentialTrueDofs[0], *coarse_mat);

   Solver* coarse_solver = nullptr;
   switch (solver_type)
   {
      case sli:
         coarse_solver = new SLISolver(MPI_COMM_WORLD);
         break;
      case cg:
         coarse_solver = new CGSolver(MPI_COMM_WORLD);
         break;
      default:
         mfem_error("Invalid solver type!");
   }

   coarse_pc = new OperatorLpqJacobiSmoother(*coarse_mat, *essentialTrueDofs[0],
                                             p_order, q_order);

   IterativeSolver *it_solver = dynamic_cast<IterativeSolver*>(coarse_solver);
   if (it_solver)
   {
      it_solver->SetRelTol(MG_REL_TOL);
      it_solver->SetMaxIter(MG_MAX_ITER);
      it_solver->SetPrintLevel(-1);
      it_solver->SetPreconditioner(*coarse_pc);
   }
   coarse_solver->SetOperator(*coarse_mat);
   AddLevel(coarse_mat, coarse_solver, true, true);
}

void LpqGeometricMultigrid::ConstructOperatorAndSmoother(
   ParFiniteElementSpace& fespace, int level)
{
   const Array<int> &ess_tdof_list = *essentialTrueDofs[level];
   ConstructBilinearForm(fespace);

   auto level_mat = new HypreParMatrix();
   bfs.Last()->FormSystemMatrix(ess_tdof_list, *level_mat);

   Solver* smoother = new OperatorLpqJacobiSmoother(*level_mat,
                                                    ess_tdof_list,
                                                    p_order,
                                                    q_order);

   AddLevel(level_mat, smoother, true, true);
}


void LpqGeometricMultigrid::ConstructBilinearForm(ParFiniteElementSpace&
                                                  fespace)
{
   ParBilinearForm* form = new ParBilinearForm(&fespace);
   switch (integrator_type)
   {
      case mass:
         form->AddDomainIntegrator(new MassIntegrator);
         break;
      case diffusion:
         form->AddDomainIntegrator(new DiffusionIntegrator);
         break;
      case elasticity:
         form->AddDomainIntegrator(new ElasticityIntegrator(one, one));
         break;
      case maxwell:
         form->AddDomainIntegrator(new CurlCurlIntegrator(one));
         form->AddDomainIntegrator(new VectorFEMassIntegrator(one));
         break;
      default:
         mfem_error("Invalid integrator type! Check ParBilinearForm");
   }
   form->Assemble();
   bfs.Append(form);
}

// Abs-L(1) general geometric multigrid method, derived from GeometricMultigrid
AbsL1GeometricMultigrid::AbsL1GeometricMultigrid(
   ParFiniteElementSpaceHierarchy& fes_hierarchy,
   Array<int>& ess_bdr,
   IntegratorType it,
   SolverType st,
   AssemblyLevel al)
   : GeometricMultigrid(fes_hierarchy, ess_bdr),
     integrator_type(it),
     solver_type(st),
     assembly_level(al),
     coarse_pc(nullptr),
     one(1.0)
{
   ConstructCoarseOperatorAndSolver(fes_hierarchy.GetFESpaceAtLevel(0));
   for (int l = 1; l < fes_hierarchy.GetNumLevels(); ++l)
   {
      ConstructOperatorAndSmoother(fes_hierarchy.GetFESpaceAtLevel(l), l);
   }
}

void AbsL1GeometricMultigrid::ConstructCoarseOperatorAndSolver(
   ParFiniteElementSpace& coarse_fespace)
{
   ConstructBilinearForm(coarse_fespace);

   OperatorPtr coarse_mat;
   bfs[0]->FormSystemMatrix(*essentialTrueDofs[0], coarse_mat);

   Solver* coarse_solver = nullptr;
   switch (solver_type)
   {
      case sli:
         coarse_solver = new SLISolver(MPI_COMM_WORLD);
         break;
      case cg:
         coarse_solver = new CGSolver(MPI_COMM_WORLD);
         break;
      default:
         mfem_error("Invalid solver type!");
   }

   {
      Vector local_ones(coarse_mat->Height());
      Vector result(coarse_mat->Height());

      local_ones = 1.0;
      coarse_mat->AbsMult(local_ones, result);

      coarse_pc = new OperatorJacobiSmoother(result, *essentialTrueDofs[0]);
   }

   IterativeSolver *it_solver = dynamic_cast<IterativeSolver*>(coarse_solver);
   if (it_solver)
   {
      it_solver->SetRelTol(MG_REL_TOL);
      it_solver->SetMaxIter(MG_MAX_ITER);
      it_solver->SetPrintLevel(-1);
      it_solver->SetPreconditioner(*coarse_pc);
   }
   coarse_solver->SetOperator(*coarse_mat);
   AddLevel(coarse_mat.Ptr(), coarse_solver, true, true);
}

void AbsL1GeometricMultigrid::ConstructOperatorAndSmoother(
   ParFiniteElementSpace& fespace, int level)
{
   const Array<int> &ess_tdof_list = *essentialTrueDofs[level];
   ConstructBilinearForm(fespace);

   OperatorPtr level_mat;
   bfs.Last()->FormSystemMatrix(ess_tdof_list, level_mat);

   Vector local_ones(level_mat->Height());
   Vector result(level_mat->Height());

   local_ones = 1.0;
   level_mat->AbsMult(local_ones, result);

   Solver* smoother = new OperatorJacobiSmoother(result, *essentialTrueDofs[0]);

   AddLevel(level_mat.Ptr(), smoother, true, true);
}


void AbsL1GeometricMultigrid::ConstructBilinearForm(ParFiniteElementSpace&
                                                    fespace)
{
   ParBilinearForm* form = new ParBilinearForm(&fespace);
   form->SetAssemblyLevel(assembly_level);
   switch (integrator_type)
   {
      case mass:
         form->AddDomainIntegrator(new MassIntegrator);
         break;
      case diffusion:
         form->AddDomainIntegrator(new DiffusionIntegrator);
         break;
      case elasticity:
         form->AddDomainIntegrator(new ElasticityIntegrator(one, one));
         break;
      case maxwell:
         form->AddDomainIntegrator(new CurlCurlIntegrator(one));
         form->AddDomainIntegrator(new VectorFEMassIntegrator(one));
         break;
      default:
         mfem_error("Invalid integrator type! Check ParBilinearForm");
   }
   form->Assemble();
   bfs.Append(form);
}

real_t diffusion_solution(const Vector &x)
{
   if (dim == 3)
   {
      return sin(kappa * x(0)) * sin(kappa * x(1)) * sin(kappa * x(2)) + 1.0;
   }
   else
   {
      return sin(kappa * x(0)) * sin(kappa * x(1)) + 1.0;
   }
}

real_t diffusion_source(const Vector &x)
{
   if (dim == 3)
   {
      return dim * kappa * kappa * sin(kappa * x(0)) * sin(kappa * x(1)) * sin(
                kappa * x(2));
   }
   else
   {
      return dim * kappa * kappa * sin(kappa * x(0)) * sin(kappa * x(1));
   }
}

void elasticity_solution(const Vector &x, Vector &u)
{
   if (dim == 3)
   {
      u(0) = sin(kappa * x(0));
      u(1) = sin(kappa * x(1));
      u(2) = sin(kappa * x(2));
   }
   else
   {
      u(0) = sin(kappa * x(0));
      u(1) = sin(kappa * x(1));
      if (x.Size() == 3) { u(2) = 0.0; }
   }
}

void elasticity_source(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = - 3.0 * kappa * kappa * sin(kappa * x(0));
      f(1) = - 3.0 * kappa * kappa * sin(kappa * x(1));
      f(2) = - 3.0 * kappa * kappa * sin(kappa * x(2));
   }
   else
   {
      f(0) = - 3.0 * kappa * kappa * sin(kappa * x(0));
      f(1) = - 3.0 * kappa * kappa * sin(kappa * x(1));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}

void maxwell_solution(const Vector &x, Vector &u)
{
   if (dim == 3)
   {
      u(0) = sin(kappa * x(1));
      u(1) = sin(kappa * x(2));
      u(2) = sin(kappa * x(0));
   }
   else
   {
      u(0) = sin(kappa * x(1));
      u(1) = sin(kappa * x(0));
      if (x.Size() == 3) { u(2) = 0.0; }
   }
}

void maxwell_source(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
      f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
   }
   else
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}

} // end namespace lpq_jacobi
