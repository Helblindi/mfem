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

#include "../config/config.hpp"
#include "hiop-sparse.hpp"

#ifdef MFEM_USE_HIOP
#include <iostream>

#include "hiopAlgFilterIPM.hpp"

using namespace hiop;

namespace mfem
{

bool HiopSparseOptimizationProblem::get_prob_sizes(size_type &n, size_type &m)
{
   n = ntdofs_glob;
   m = problem.GetNumConstraints();

   return true;
}

bool HiopSparseOptimizationProblem::get_starting_point(const size_type &n, double *x0)
{
   MFEM_ASSERT(x_start != NULL && ntdofs_loc == x_start->Size(),
               "Starting point is not set properly.");

   memcpy(x0, x_start->GetData(), ntdofs_loc * sizeof(double));

   return true;
}

bool HiopSparseOptimizationProblem::get_vars_info(const size_type &n,
                                            double *xlow, double *xupp,
                                            NonlinearityType *type)
{
   MFEM_ASSERT(n == ntdofs_glob, "Global input mismatch.");
   MFEM_ASSERT(problem.GetBoundsVec_Lo() && problem.GetBoundsVec_Hi(),
               "Solution bounds are not set!");

   const int s = ntdofs_loc * sizeof(double);
   std::memcpy(xlow, problem.GetBoundsVec_Lo()->GetData(), s);
   std::memcpy(xupp, problem.GetBoundsVec_Hi()->GetData(), s);

   return true;
}

bool HiopSparseOptimizationProblem::get_cons_info(const size_type &m,
                                            double *clow, double *cupp,
                                            NonlinearityType *type)
{
   MFEM_ASSERT(m == m_total, "Global constraint size mismatch.");

   int csize = 0;
   if (problem.GetC())
   {
      csize = problem.GetEqualityVec()->Size();
      const int s = csize * sizeof(double);
      std::memcpy(clow, problem.GetEqualityVec()->GetData(), s);
      std::memcpy(cupp, problem.GetEqualityVec()->GetData(), s);
   }
   if (problem.GetD())
   {
      const int s = problem.GetInequalityVec_Lo()->Size() * sizeof(double);
      std::memcpy(clow + csize, problem.GetInequalityVec_Lo()->GetData(), s);
      std::memcpy(cupp + csize, problem.GetInequalityVec_Hi()->GetData(), s);
   }

   return true;
}

bool HiopSparseOptimizationProblem::eval_f(const size_type &n, const double *x,
                                     bool new_x, double &obj_value)
{
   MFEM_ASSERT(n == ntdofs_glob, "Global input mismatch.");

   if (new_x) { constr_info_is_current = false; }

   Vector x_vec(ntdofs_loc);
   x_vec = x;
   problem.new_x = new_x;
   obj_value = problem.CalcObjective(x_vec);

   return true;
}

bool HiopSparseOptimizationProblem::eval_grad_f(const size_type &n, const double *x,
                                          bool new_x, double *gradf)
{
   MFEM_ASSERT(n == ntdofs_glob, "Global input mismatch.");

   if (new_x) { constr_info_is_current = false; }

   Vector x_vec(ntdofs_loc), gradf_vec(ntdofs_loc);
   x_vec = x;
   problem.new_x = new_x;
   problem.CalcObjectiveGrad(x_vec, gradf_vec);
   std::memcpy(gradf, gradf_vec.GetData(), ntdofs_loc * sizeof(double));

   return true;
}

bool HiopSparseOptimizationProblem::eval_cons(const size_type &n, const size_type &m,
                                        const size_type &num_cons,
                                        const index_type *idx_cons,
                                        const double *x, bool new_x,
                                        double *cons)
{
   return false;
}

bool HiopSparseOptimizationProblem::eval_cons(const size_type &n, 
                                              const size_type &m,
                                              const double *x, 
                                              bool new_x,
                                              double *cons)
{
   MFEM_ASSERT(n == ntdofs_glob, "Global input mismatch.");
   MFEM_ASSERT(m == m_total, "Constraint size mismatch.");

   if (new_x) { constr_info_is_current = false; }
   Vector x_vec(ntdofs_loc);
   x_vec = x;
   problem.new_x = new_x;
   UpdateConstrValsGrads(x_vec);

   for (int c = 0; c < m; c++)
   {
      cons[c] = constr_vals(c);
   }

   return true;
}



bool HiopSparseOptimizationProblem::eval_Jac_cons(const size_type &n,
                                            const size_type &m,
                                            const double *x, bool new_x,
                                            const size_type &nnzJacS,
                                            index_type *iJacS,
                                            index_type *jJacS,
                                            double * MJacS)
{
   MFEM_ASSERT(n == ntdofs_glob, "Global input mismatch.");
   MFEM_ASSERT(m == m_total, "Constraint size mismatch.");

   if (m == 0) { return true; }

   if (new_x) { constr_info_is_current = false; }
   Vector x_vec(ntdofs_loc);
   x_vec = x;
   problem.new_x = new_x;
   UpdateConstrValsGrads(x_vec);

   if (iJacS != NULL && jJacS != NULL)
   {
      int count = 0;
      for (int i = 0; i < m/*num_rows*/; i++)
      {
         for (int k = cgIArr[i]; k < cgIArr[i+1]; k++, count++)
         {
            iJacS[count] = i;
            jJacS[count] = cgJArr[count];
         }
      }
      assert(count == nnzHSS);
   }

   if (MJacS != NULL)
   {
      int last;

      for (int i = 0; i < nnzJacS; i++)
      {
         MJacS[i] = cgDataArr[i];
      }

   }
   return true;
}

bool HiopSparseOptimizationProblem::get_vecdistrib_info(size_type global_n,
                                                  index_type *cols)
{
#ifdef MFEM_USE_MPI
   int nranks;
   MPI_Comm_size(comm, &nranks);

   size_type *sizes = new size_type[nranks];
   MPI_Allgather(&ntdofs_loc, 1, MPI_HIOP_SIZE_TYPE, sizes, 1,
                 MPI_HIOP_SIZE_TYPE, comm);
   cols[0] = 0;
   for (int r = 1; r <= nranks; r++)
   {
      cols[r] = sizes[r-1] + cols[r-1];
   }

   delete [] sizes;
   sizes = nullptr;

   return true;
#else
   // Returning false means that Hiop runs in non-distributed mode.
   return false;
#endif
}

void HiopSparseOptimizationProblem::solution_callback(hiop::hiopSolveStatus status,
                                                hiop::size_type n,
                                                const double *x,
                                                const double *z_L,
                                                const double *z_U,
                                                hiop::size_type m,
                                                const double *g,
                                                const double *lambda,
                                                double obj_value)
{
   auto hp = dynamic_cast<const HiOpSparseProblem *>(&problem);
   if (!hp) { return; }
   hp->SolutionCallback(status, n, x, z_L, z_U, m, g, lambda, obj_value);
}

bool HiopSparseOptimizationProblem::iterate_callback(int iter,
                                               double obj_value,
                                               double logbar_obj_value,
                                               int n,
                                               const double *x,
                                               const double *z_L,
                                               const double *z_U,
                                               int m_ineq,
                                               const double *s,
                                               int m,
                                               const double *g,
                                               const double *lambda,
                                               double inf_pr,
                                               double inf_du,
                                               double onenorm_pr_,
                                               double mu,
                                               double alpha_du,
                                               double alpha_pr,
                                               int ls_trials)
{
   auto hp = dynamic_cast<const HiOpSparseProblem *>(&problem);
   if (!hp) { return true; }
   return hp->IterateCallback(iter, obj_value, logbar_obj_value, n, x, z_L, z_U,
                              m_ineq, s, m, g, lambda, inf_pr, inf_du,
                              onenorm_pr_, mu, alpha_du, alpha_pr, ls_trials);
}

void HiopSparseOptimizationProblem::UpdateConstrValsGrads(const Vector x)
{
   // std::cout << "UpdateConstrValsGrads\n";
   if (constr_info_is_current) { return; }

   cgIArr.SetSize(0, 0), cgJArr.SetSize(0,0), cgDataArr.SetSize(0,0.);

   // If needed (e.g. for CG spaces), communication should be handled by the
   // operators' Mult() and GetGradient() methods.
   Array<int> cols;
   Vector row(nnz_constr);
   int cheight = 0, lastc_row = 0;
   if (problem.GetC())
   {
      cheight = problem.GetC()->Height();

      // Values of C.
      Vector vals_C(constr_vals.GetData(), cheight);
      problem.GetC()->Mult(x, vals_C);

      // Gradients C.
      const Operator &oper_C = problem.GetC()->GetGradient(x);
      const SparseMatrix *grad_C = dynamic_cast<const SparseMatrix *>(&oper_C);
      MFEM_VERIFY(grad_C, "HiopSparse expects SparseMatrices as operator gradients.");
      MFEM_ASSERT(grad_C->Height() == cheight && grad_C->Width() == ntdofs_loc,
                  "Incorrect dimensions of the C constraint gradient.");
      const int *In = grad_C->GetI(), *Jn = grad_C->GetJ();
      const double *Datan = grad_C->GetData();

      for (int i = 0, k=0; i < cheight; i++)
      {
         cgIArr.Append(In[i]);
         for (int end = In[i+1]; k < end; k++)
         {
            int j = Jn[k];
            cgJArr.Append(j);
            cgDataArr.Append(Datan[k]);
         }
      }

      lastc_row = In[cheight];
      cgIArr.Append(lastc_row);
   }

   int dheight = 0;
   if (problem.GetD())
   {
      dheight = problem.GetD()->Height();

      // Values of D.
      Vector vals_D(constr_vals.GetData() + cheight, dheight);
      problem.GetD()->Mult(x, vals_D);

      // Gradients of D.
      const Operator &oper_D = problem.GetD()->GetGradient(x);
      const SparseMatrix *grad_D = dynamic_cast<const SparseMatrix *>(&oper_D);
      MFEM_VERIFY(grad_D, "HiopSparse expects SparseMatrices as operator gradients.");
      MFEM_ASSERT(grad_D->Height() == dheight && grad_D->Width() == ntdofs_loc,
                  "Incorrect dimensions of the D constraint gradient.");
      
      const int *DI = grad_D->GetI(), *DJ = grad_D->GetJ();
      const double *DData = grad_D->GetData();
      
      for (int i = 0, k=0; i < dheight; i++)
      {
         cgIArr.Append(DI[i+1]+lastc_row);
         for (int end = DI[i+1]; k < end; k++)
         {
            int j = DJ[k];
            cgJArr.Append(j);
            cgDataArr.Append(DData[k]);
         }
      }
   }
   constr_info_is_current = true;
}


/* Additional pure virtual functions for sparse implementation. */
/**
 * nx - number of optimized parameters
 * nnz_sparse_Jaceq - number of nonzero elements in the sparse 
 *    blocks in the Jacobian of the equality constraints
 * nnz_sparse_Jacineq - number of nonzero elements in the sparse 
 *    blocks in the Jacobian of the inequality constraints
 * nnz_sparse_Hess_Lagr - number of nonzero elements in the sparse 
 *    blocks in the Hessian of the Lagrangian
 */
bool HiopSparseOptimizationProblem::get_sparse_blocks_info(
   size_type& nx, size_type& nnz_sparse_Jaceq,
   size_type& nnz_sparse_Jacineq, 
   size_type& nnz_sparse_Hess_Lagr)
{
   nx = ntdofs_loc;
   nnz_sparse_Jaceq = problem.get_nnz_sparse_Jaceq();
   nnz_sparse_Jacineq = problem.get_nnz_sparse_Jacineq();
   nnz_sparse_Hess_Lagr = problem.get_nnz_sparse_Hess_Lagr();
   return true;
}

bool HiopSparseOptimizationProblem::eval_Jac_cons(const size_type& n, const size_type& m,
                        const size_type& num_cons, const index_type* idx_cons,
                        const double* x, bool new_x,
                        const size_type& nnzJacS, index_type* iJacS, index_type* jJacS, double* MJacS)
{
   return false;
}

void HiopSparseOptimizationProblem::UpdateHessLagr(const Vector &x, const Vector &lambda)
{
   // std::cout << "UpdateHessLagr\n";
   hessIArr.SetSize(0,0), hessJArr.SetSize(0,0), hessData.SetSize(0,0.);

   problem.SetLagrangeMultipliers(lambda);
   Operator &hess = problem.CalcObjectiveHess(x);

   const SparseMatrix * hl_ptr = dynamic_cast<SparseMatrix *>(&hess);

   const int *In = hl_ptr->GetI(), *Jn = hl_ptr->GetJ();
   const double *Data = hl_ptr->GetData();

   const int lagr_height = hl_ptr->Height();

   for (int i = 0, k=0; i < lagr_height; i++)
   {
      hessIArr.Append(In[i]);
      for (int end = In[i+1]; k < end; k++)
      {
         int j = Jn[k];
         hessJArr.Append(j);
         hessData.Append(Data[k]);
      }
   }
   int lastc_row = In[lagr_height];
   hessIArr.Append(lastc_row);
}

bool HiopSparseOptimizationProblem::eval_Hess_Lagr(const size_type& n,
   const size_type& m, const double* x, bool new_x,
   const double& obj_factor, const double* lambda,
   bool new_lambda, const size_type& nnzHSS,
   index_type* iHSS, index_type* jHSS, double* MHSS)
{
   // std::cout << "eval_Hess_Lagr\n";
   MFEM_ASSERT(n == ntdofs_glob, "Global input mismatch.");

   if (new_x || new_lambda) { constr_info_is_current = false; }


   Vector x_vec(ntdofs_loc), lambda_vec(m);
   x_vec = x;
   lambda_vec = lambda;
   UpdateHessLagr(x_vec, lambda_vec); // Note: hess_lagr is CSR format

   /* Must convert from csr to triplet format */
   if (iHSS != NULL && jHSS != NULL)
   {
      // const int *I = hess_lagr.GetI();
      // const int *J = hess_lagr.GetJ();

      int count = 0;
      for (int i = 0; i < n/*num rows*/; i++)
      {
         for (int k = hessIArr[i]; k < hessIArr[i+1]; k++, count++)
         {
            iHSS[count] = i;
            jHSS[count] = hessJArr[count];
         }
      }
      assert(count == nnzHSS);
   }
   
   if (MHSS != NULL)
   {
      int last;

      // const auto data_ = hess_lagr.GetData();
      for (int i = 0; i < nnzHSS; i++)
      {
         MHSS[i] = hessData[i];
      }
   }

   return true;
}


HiopNlpSparseOptimizer::HiopNlpSparseOptimizer() : OptimizationSolver(), hiop_problem(NULL)
{
#ifdef MFEM_USE_MPI
   // Set in case a serial driver uses a parallel MFEM build.
   comm = MPI_COMM_WORLD;
   int initialized, nret = MPI_Initialized(&initialized);
   MFEM_ASSERT(MPI_SUCCESS == nret, "Failure in calling MPI_Initialized!");
   if (!initialized)
   {
      nret = MPI_Init(NULL, NULL);
      MFEM_ASSERT(MPI_SUCCESS == nret, "Failure in calling MPI_Init!");
   }
#endif
}

#ifdef MFEM_USE_MPI
HiopNlpSparseOptimizer::HiopNlpSparseOptimizer(MPI_Comm comm_)
   : OptimizationSolver(comm_), hiop_problem(NULL), comm(comm_) { }
#endif

HiopNlpSparseOptimizer::~HiopNlpSparseOptimizer()
{
   // std::cout << "HiopNlpSparseOptimizer virtual destructor\n";
   delete hiop_problem;
}

void HiopNlpSparseOptimizer::SetOptimizationProblem(const OptimizationProblem &prob)
{
   // std::cout << "HiopNlpSparseOptimizer::SetOptimizationProblem\n";
   // Ensure that SetNNZSparse has been called first
   if (this->nnz_constr <= 0)
   {
      MFEM_ABORT("Must call HiopNlpSparseOptimizer::SetNNZSparse first.\n");
   }

   problem = &prob;
   height = width = problem->input_size;

   if (hiop_problem) { delete hiop_problem; }

#ifdef MFEM_USE_MPI
   hiop_problem = new HiopSparseOptimizationProblem(comm, *problem, nnz_constr);
#else
   hiop_problem = new HiopSparseOptimizationProblem(*problem, nnz_constr);
#endif
}

void HiopNlpSparseOptimizer::Mult(const Vector &xt, Vector &x) const
{
   MFEM_ASSERT(hiop_problem != NULL,
               "Unspecified OptimizationProblem that must be solved.");

   hiop_problem->setStartingPoint(xt);

   hiop::hiopNlpSparse hiopInstance(*hiop_problem);

   /* my added options */
   hiopInstance.options->SetStringValue("force_resto", "yes");
   hiopInstance.options->SetStringValue("KKTLinsys", "xdycyd");
   hiopInstance.options->SetStringValue("duals_init", "zero");

   hiopInstance.options->SetNumericValue("rel_tolerance", rel_tol);
   hiopInstance.options->SetNumericValue("tolerance", abs_tol);
   hiopInstance.options->SetIntegerValue("max_iter", max_iter);
   hiopInstance.options->SetNumericValue("mu0", 1e-1);

   // hiopInstance.options->SetStringValue("fixed_var", "relax");
   // hiopInstance.options->SetNumericValue("fixed_var_tolerance", 1e-20);
   // hiopInstance.options->SetNumericValue("fixed_var_perturb", 1e-9);
   hiopInstance.options->SetIntegerValue("verbosity_level", print_level);
   // Disable safe mode to avoid crash
   hiopInstance.options->SetStringValue("linsol_mode", "stable");

   // Use the IPM solver.
   hiop::hiopAlgFilterIPMNewton solver(&hiopInstance);
   const hiop::hiopSolveStatus status = solver.run();

   final_norm = solver.getObjective();
   final_iter = solver.getNumIterations();

   if (status != hiop::Solve_Success && status != hiop::Solve_Success_RelTol)
   {
      // converged = false;
      converged = true;
      MFEM_WARNING("HIOP returned with a non-success status: " << status);
      std::cout << "final_norm: " << final_norm << std::endl;
      std::cout << "final_iter: " << final_iter << std::endl;
   }
   else { converged = true; }

   // Copy the final solution in x.
   solver.getSolution(x.GetData());
}

} // mfem namespace
#endif // MFEM_USE_HIOP
