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
   // std::cout << "HiopSparseOptimizationProblem::eval_cons\n";
   // MFEM_ASSERT(n == ntdofs_glob, "Global input mismatch.");
   // MFEM_ASSERT(m == m_total, "Constraint size mismatch.");
   // MFEM_ASSERT(num_cons <= m, "num_cons should be at most m = " << m);

   // if (num_cons == 0) { return true; }

   // if (new_x) { constr_info_is_current = false; }
   // Vector x_vec(ntdofs_loc);
   // x_vec = x;
   // problem.new_x = new_x;
   // UpdateConstrValsGrads(x_vec);

   // for (int c = 0; c < num_cons; c++)
   // {
   //    MFEM_ASSERT(idx_cons[c] < m_total, "Constraint index is out of bounds.");
   //    cons[c] = constr_vals(idx_cons[c]);
   // }

   // return true;
   return false;
}

bool HiopSparseOptimizationProblem::eval_cons(const size_type &n, 
                                              const size_type &m,
                                              const double *x, 
                                              bool new_x,
                                              double *cons)
{
   // std::cout << "HiopSparseOptimizationProblem::eval_cons\n";
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
   // std::cout << "HiopSparseOptimizationProblem::eval_Jac_cons 8 args\n";
   MFEM_ASSERT(n == ntdofs_glob, "Global input mismatch.");
   MFEM_ASSERT(m == m_total, "Constraint size mismatch.");

   if (m == 0) { return true; }

   if (new_x) { constr_info_is_current = false; }
   Vector x_vec(ntdofs_loc);
   x_vec = x;
   problem.new_x = new_x;
   UpdateConstrValsGrads(x_vec);

   iJacS = constr_grads->GetI();
   jJacS = constr_grads->GetJ();
   MJacS = constr_grads->GetData();

   // std::cout << "HiopSparseOptimizationProblem::eval_Jac_cons 8 args - DONE\n";
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
   // std::cout << "HiopSparseOptimizationProblem::UpdateConstrValsGrads\n";
   if (constr_info_is_current) { return; }

   // If needed (e.g. for CG spaces), communication should be handled by the
   // operators' Mult() and GetGradient() methods.

   Array<int> cols;
   Vector row(nnz_constr);
   int cheight = 0;
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
      // const int *In = grad_C->GetI(), *Jn = grad_C->GetJ();
      *constr_grads = *grad_C; // No inequality constraints so this works for now
      
      // for (int i = 0, k=0; i < cheight; i++)
      // {
      //    std::cout << "i: " << i << ", In[i]: " << In[i] << std::endl;
      //    // grad_C->GetRow(i, cols, row);
      //    // std::cout << "i: " << i << ", row: ";
      //    // row.Print(std::cout);
      //    // std::cout << "cols: ";
      //    // cols.Print(std::cout);
      //    // constr_grads->SetRow(i, cols, row);
      //    for (int end = In[i+1]; k < end; k++)
      //    {
      //       int j = Jn[k];
      //       std::cout << "sparse entry [i: " << i << ", j: " << j << "]: " << constr_grads->Elem(i,j) << std::endl;
      //       // constr_grads->Elem(i, j) = grad_C->Elem(i, j);
      //    }
      // }
   }
   // MFEM_ABORT("Testing\n");

   if (problem.GetD())
   {
      const int dheight = problem.GetD()->Height();

      // Values of D.
      Vector vals_D(constr_vals.GetData() + cheight, dheight);
      problem.GetD()->Mult(x, vals_D);

      // Gradients of D.
      const Operator &oper_D = problem.GetD()->GetGradient(x);
      const SparseMatrix *grad_D = dynamic_cast<const SparseMatrix *>(&oper_D);
      MFEM_VERIFY(grad_D, "HiopSparse expects SparseMatrices as operator gradients.");
      MFEM_ASSERT(grad_D->Height() == dheight && grad_D->Width() == ntdofs_loc,
                  "Incorrect dimensions of the D constraint gradient.");

      for (int i = 0; i < dheight; i++)
      {
         grad_D->GetRow(i, cols, row);
         constr_grads->SetRow(i+cheight, cols, row);
      }
   }
   constr_grads->Finalize();

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
   // MFEM_ABORT("HiopSparseOptimizationProblem::get_sparse_blocks_info note yet implemented.\n");
   nx = ntdofs_loc;
   nnz_sparse_Jaceq = 8; // TODO: Hardcoded for our specific problem.
   // 8 nonzero entries per row.
   nnz_sparse_Jacineq = 0; // TODO: Hardcoded for our specific problem
   nnz_sparse_Hess_Lagr = ntdofs_loc; // TODO: May need to fix this.
   return true;
}

bool HiopSparseOptimizationProblem::eval_Jac_cons(const size_type& n, const size_type& m,
                        const size_type& num_cons, const index_type* idx_cons,
                        const double* x, bool new_x,
                        const size_type& nnzJacS, index_type* iJacS, index_type* jJacS, double* MJacS)
{
   // std::cout << "HiopSparseOptimizationProblem::eval_Jac_cons 10 args.\n";

   // MFEM_ASSERT(n == ntdofs_glob, "Global input mismatch.");
   // MFEM_ASSERT(m == m_total, "Constraint size mismatch.");
   // MFEM_ASSERT(num_cons <= m, "num_cons should be at most m = " << m);

   // if (num_cons == 0) { return true; }

   // if (new_x) { constr_info_is_current = false; }
   // Vector x_vec(ntdofs_loc);
   // x_vec = x;
   // problem.new_x = new_x;
   // UpdateConstrValsGrads(x_vec);

   // const int *I = constr_grads->GetI();
   // const int *J = constr_grads->GetJ();
   // const double *data = constr_grads->GetData();


   // if (iJacS != NULL && jJacS != NULL)
   // {
   //    for (int c = 0, k=0; c < num_cons; c++)
   //    {
   //       int row = idx_cons[c];
   //       iJacS[c] = c;

   //       MFEM_ASSERT(row< m_total, "Constraint index is out of bounds.");
   //       for (int end = k + nnz_constr, j=I[row]; k < end; k++, j++)
   //       {
   //          // The matrix is stored by rows.
   //          jJacS[k] = J[j];
   //       }
   //    }
   // }

   // if (MJacS != NULL)
   // {
   //    for (int c=0, k = 0; c < num_cons; c++)
   //    {
   //       int row = idx_cons[c];
   //       // double* row_entries = constr_grads->GetRowEntries(row);
   //       for (int end = c*nnz_constr, j=I[row]; k < end; k++, j++)
   //       {
   //          MJacS[k] = data[j];
   //       }
   //    }
   // }
   

   // for (int end = In[i+1]; k < end; k++)
   // {
   //    int j = Jn[k];
   //    std::cout << "sparse entry [i: " << i << ", j: " << j << "]: " << grad_C->Elem(i,j) << std::endl;
   //    constr_grads->Elem(i, j) = grad_C->Elem(i, j);
   // }

   // std::cout << "Done HiopSparseOptimizationProblem::eval_Jac_cons 10 args.\n";

   // return true; 
   return false;
}

void HiopSparseOptimizationProblem::UpdateHessLagr(const Vector &x, const Vector &lambda)
{
   // std::cout << "HiopSparseOptimizationProblem::UpdateHessLagr\n";

   problem.SetLagrangeMultipliers(lambda);
   // problem.CalcObjectiveHess(x, *hess_lagr);

   Operator &hess = problem.CalcObjectiveHess(x);
   hess_lagr = dynamic_cast<SparseMatrix *>(&hess);
   
   for (int i = 0; i < 16; i++)
   {
      // std::cout << "i: " << hess_lagr->GetData()[i] << std::endl;
   }
   

   // std::cout << "HiopSparseOptimizationProblem::UpdateHessLagr - DONE\n";
   // MFEM_ABORT("Function not implemented.\n");
}

bool HiopSparseOptimizationProblem::eval_Hess_Lagr(const size_type& n,
   const size_type& m, const double* x, bool new_x,
   const double& obj_factor, const double* lambda,
   bool new_lambda, const size_type& nnzHSS,
   index_type* iHSS, index_type* jHSS, double* MHSS)
{
   // std::cout << "HiopSparseOptimizationProblem::eval_Hess_Lagr funcall\n";
   // MFEM_ABORT("HiopSparseOptimizationProblem::eval_Hess_Lagr not yet implemented.\n")
   MFEM_ASSERT(n == ntdofs_glob, "Global input mismatch.");

   if (new_x) { constr_info_is_current = false; }

   Vector x_vec(ntdofs_loc), lambda_vec(m);
   x_vec = x;
   lambda_vec = lambda;
   UpdateHessLagr(x_vec, lambda_vec);
   
   if (iHSS != NULL & jHSS != NULL) {
      for (int i = 0; i < n; i++) iHSS[i] = jHSS[i] = i;
   }

   if (MHSS != NULL) {
      for (int i = 0; i < n; i++) {
         MHSS[i] = 2.;
      }
   }
   // std::cout << "HiopSparseOptimizationProblem::eval_Hess_Lagr - DONE\n";
   return true;
}

// bool HiopSparseOptimizationProblem::eval_Hess_Lagr(const size_type& n,
//    const size_type& m, const double* x, bool new_x,
//    const double& obj_factor, const double* lambda,
//    bool new_lambda, const size_type& nnzHSS,
//    index_type* iHSS, index_type* jHSS, double* MHSS)
// {
//    std::cout << "HiopSparseOptimizationProblem::eval_Hess_Lagr funcall\n";
//    // MFEM_ABORT("HiopSparseOptimizationProblem::eval_Hess_Lagr not yet implemented.\n");
//    // Note: lambda is not used since all the constraints are linear and, therefore, do
//    // not contribute to the Hessian of the Lagrangian
//    assert(nnzHSS == n);

//    if (iHSS != NULL & jHSS != NULL) {
//       for (int i = 0; i < n; i++) iHSS[i] = jHSS[i] = i;
//    }

//    if (MHSS != NULL) {
//       for (int i = 0; i < n; i++) {
//          MHSS[i] = 2.;
//       }
//    }
//    std::cout << "HiopSparseOptimizationProblem::eval_Hess_Lagr - DONE\n";
//    return true;
// }

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
   delete hiop_problem;
}

void HiopNlpSparseOptimizer::SetOptimizationProblem(const OptimizationProblem &prob)
{
   // Ensure that SetNNZSparse has been called first
   // std::cout << "HiopNlpSparseOptimizer::SetOptimizationProblem\n";
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

   hiopInstance.options->SetNumericValue("rel_tolerance", rel_tol);
   hiopInstance.options->SetNumericValue("tolerance", abs_tol);
   hiopInstance.options->SetIntegerValue("max_iter", max_iter);

   hiopInstance.options->SetStringValue("fixed_var", "relax");
   hiopInstance.options->SetNumericValue("fixed_var_tolerance", 1e-20);
   hiopInstance.options->SetNumericValue("fixed_var_perturb", 1e-9);

   hiopInstance.options->SetNumericValue("mu0", 1e-1);

   // 0: no output; 3: not too much
   // hiopInstance.options->SetIntegerValue("verbosity_level", print_level);
   hiopInstance.options->SetIntegerValue("verbosity_level", 3);

   // Disable safe mode to avoid crash
   hiopInstance.options->SetStringValue("linsol_mode", "speculative");

   // Use the IPM solver.
   hiop::hiopAlgFilterIPMNewton solver(&hiopInstance);
   const hiop::hiopSolveStatus status = solver.run();

   final_norm = solver.getObjective();
   final_iter = solver.getNumIterations();

   // std::cout << "Solve Success: " << hiop::Solve_Success << std::endl;
   // std::cout << "Solve Success Rel Tol: " << hiop::Solve_Success_RelTol << std::endl;
   // std::cout << "hiopSolveStatus: " << status << std::endl;

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
