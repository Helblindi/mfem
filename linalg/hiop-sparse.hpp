// Written by: Madison Sheridan

#ifndef MFEM_HIOP_SPARSE
#define MFEM_HIOP_SPARSE

#include "../config/config.hpp"

#ifdef MFEM_USE_HIOP
#include "../general/globals.hpp"
#include "solvers.hpp"
#include "sparsemat.hpp"

#ifdef MFEM_USE_MPI
#include "operator.hpp"
#endif

#include "hiopInterface.hpp"
#include "hiopNlpFormulation.hpp"

namespace mfem
{
using size_type = hiop::size_type;
using index_type = hiop::index_type;

/// Internal class - adapts the OptimizationProblem class to HiOp's sparse interface.
class HiopSparseOptimizationProblem : public hiop::hiopInterfaceSparse
{
private:

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif

   // Problem info.
   const OptimizationProblem &problem;

   // Local and global number of variables and constraints.
   const size_type ntdofs_loc, m_total, nnz_constr;
   size_type ntdofs_glob;

   // Initial guess.
   const Vector *x_start;

   Vector constr_vals;
   SparseMatrix *constr_grads;
   SparseMatrix *hess_lagr;
   bool constr_info_is_current;
   void UpdateConstrValsGrads(const Vector x);
   void UpdateHessLagr(const Vector &x, const Vector &lambda);

public:
   HiopSparseOptimizationProblem(const OptimizationProblem &prob, const size_type &_nnz_constr)
      : problem(prob),
        ntdofs_loc(prob.input_size), m_total(prob.GetNumConstraints()),
        nnz_constr(_nnz_constr), ntdofs_glob(ntdofs_loc),
        x_start(NULL),
        constr_vals(m_total),
        constr_info_is_current(false)
   {
#ifdef MFEM_USE_MPI
      // Used when HiOp with MPI support is called by a serial driver.
      comm = MPI_COMM_WORLD;
#endif
      constr_grads = new SparseMatrix(m_total, ntdofs_loc, nnz_constr);
      hess_lagr = new SparseMatrix(ntdofs_loc, ntdofs_loc, 9);
   }

#ifdef MFEM_USE_MPI
   HiopSparseOptimizationProblem(const MPI_Comm& comm_,
                           const OptimizationProblem &prob,
                           const size_type &_nnz_constr)
      : comm(comm_),
        problem(prob),
        ntdofs_loc(prob.input_size), m_total(prob.GetNumConstraints()),
        nnz_constr(_nnz_constr), ntdofs_glob(0),
        x_start(NULL),
        constr_vals(m_total),
        constr_info_is_current(false)
   {
      MPI_Allreduce(&ntdofs_loc, &ntdofs_glob, 1, MPI_HIOP_SIZE_TYPE, MPI_SUM, comm);
      constr_grads = new SparseMatrix(m_total, ntdofs_loc, nnz_constr);
   }
#endif

   ~HiopSparseOptimizationProblem()
   {
      delete constr_grads;
      constr_grads = nullptr;

      delete hess_lagr;
      hess_lagr = nullptr;
   }

   void setStartingPoint(const Vector &x0) { x_start = &x0; }

   /** Extraction of problem dimensions:
    *  n is the number of variables, m is the number of constraints. */
   virtual bool get_prob_sizes(size_type& n, size_type& m);

   /** Provide an primal starting point. This point is subject to adjustments
    *  internally in HiOp. */
   virtual bool get_starting_point(const size_type &n, double *x0);

   using hiop::hiopInterfaceBase::get_starting_point;

   virtual bool get_vars_info(const size_type &n, double *xlow, double* xupp,
                              NonlinearityType* type);

   /** bounds on the constraints
    *  (clow<=-1e20 means no lower bound, cupp>=1e20 means no upper bound) */
   virtual bool get_cons_info(const size_type &m, double *clow, double *cupp,
                              NonlinearityType* type);

   /** Objective function evaluation.
    *  Each rank returns the global objective value. */
   virtual bool eval_f(const size_type &n, const double *x, bool new_x,
                       double& obj_value);

   /** Gradient of the objective function (local chunk). */
   virtual bool eval_grad_f(const size_type &n, const double *x, bool new_x,
                            double *gradf);

   /** Evaluates a subset of the constraints cons(x). The subset is of size
    *  num_cons and is described by indexes in the idx_cons array,
    *  i.e. cons[c] = C(x)[idx_cons[c]] where c = 0 .. num_cons-1.
    *  The methods may be called multiple times, each time for a subset of the
    *  constraints, for example, for the subset containing the equalities and
    *  for the subset containing the inequalities. However, each constraint will
    *  be inquired EXACTLY once. This is done for performance considerations,
    *  to avoid temporary holders and memory copying.
    *
    *  Parameters:
    *   - n, m: the global number of variables and constraints
    *   - num_cons, idx_cons (array of size num_cons): the number and indexes of
    *     constraints to be evaluated
    *   - x: the point where the constraints are to be evaluated
    *   - new_x: whether x has been changed from the previous call to f, grad_f,
    *     or Jac
    *   - cons: array of size num_cons containing the value of the  constraints
    *     indicated by idx_cons
    *
    *  When MPI enabled, every rank populates cons, since the constraints are
    *  not distributed.
    */
   virtual bool eval_cons(const size_type &n, const size_type &m,
                          const size_type &num_cons,
                          const index_type *idx_cons,
                          const double *x, bool new_x, double *cons);

   virtual bool eval_cons(const size_type& n,
                          const size_type& m,
                          const double* x,
                          bool new_x,
                          double* cons);

   /** Evaluates the Jacobian of the subset of constraints indicated by
    *  idx_cons. The idx_cons is assumed to be of size num_cons.
    *  Example: if cons[c] = C(x)[idx_cons[c]] where c = 0 .. num_cons-1, then
    *  one needs to do Jac[c][j] = d cons[c] / dx_j, j = 1 .. n_loc.
    *  Jac is computed and stored in a contiguous vector (offset by rows).
    *
    *  Parameters: see eval_cons().
    *
    *  When MPI enabled, each rank computes only the local columns of the
    *  Jacobian, that is the partials with respect to local variables.
    */
   virtual bool eval_Jac_cons(const size_type &n, const size_type &m,
                              const double *x, bool new_x,
                              const size_type &nnzJacS, index_type *iJacS,
                              index_type *jJacS, double * MJacS);

   /** Specifies column partitioning for distributed memory vectors.
    *  Process p owns vector entries with indices cols[p] to cols[p+1]-1,
    *  where p = 0 .. nranks-1. The cols array is of size nranks + 1.
    *  Example: for a vector x of 6 entries (globally) on 3 ranks, the uniform
    *  column partitioning is cols=[0,2,4,6].
    */
   virtual bool get_vecdistrib_info(size_type global_n,
                                    index_type *cols);

   virtual void solution_callback(hiop::hiopSolveStatus status,
                                  size_type n,
                                  const double *x,
                                  const double *z_L,
                                  const double *z_U,
                                  size_type m,
                                  const double *g,
                                  const double *lambda,
                                  double obj_value);

   virtual bool iterate_callback(int iter,
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
                                 int ls_trials);

#ifdef MFEM_USE_MPI
   virtual bool get_MPI_comm(MPI_Comm &comm_out)
   {
      comm_out = comm;
      return true;
   }
#endif

/* Additional pure virtual functions for sparse implementation */
   virtual bool get_sparse_blocks_info(size_type& nx,
                                       size_type& nnz_sparse_Jaceq,
                                       size_type& nnz_sparse_Jacineq,
                                       size_type& nnz_sparse_Hess_Lagr);

   virtual bool eval_Jac_cons(const size_type& n,
                              const size_type& m,
                              const size_type& num_cons, 
                              const index_type* idx_cons,
                              const double* x,
                              bool new_x,
                              const size_type& nnzJacS,
                              index_type* iJacS,
                              index_type* jJacS,
                              double* MJacS);

   virtual bool eval_Hess_Lagr(const size_type& n,
                               const size_type& m,
                               const double* x,
                               bool new_x,
                               const double& obj_factor,
                               const double* lambda,
                               bool new_lambda,
                               const size_type& nnzHSS,
                               index_type* iHSS,
                               index_type* jHSS,
                               double* MHSS);

   virtual void GetHessForLagr(const Vector &x, SparseMatrix &hess) const 
   { MFEM_ABORT("The objective hessian is not implemented."); }
};

/// Users can inherit this class to access to HiOp-specific functionality.
class HiOpSparseProblem : public OptimizationProblem
{
public:
   HiOpSparseProblem(int insize, const Operator *C_, const Operator *D_)
      : OptimizationProblem(insize, C_, D_) { }

   /// See hiopInterfaceBase::solution_callback(...).
   virtual void SolutionCallback(hiop::hiopSolveStatus status,
                                 size_type n,
                                 const double *x,
                                 const double *z_L,
                                 const double *z_U,
                                 size_type m,
                                 const double *g,
                                 const double *lambda,
                                 double obj_value) const
   { }

   /// See hiopInterfaceBase::iterate_callback(...).
   virtual bool IterateCallback(int iter,
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
                                int ls_trials) const
   { return true; }
};

/// Adapts the HiOp functionality to the MFEM OptimizationSolver interface.
class HiopNlpSparseOptimizer : public OptimizationSolver
{
protected:
   HiopSparseOptimizationProblem *hiop_problem;
   size_type nnz_constr = 0;

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif

public:
   HiopNlpSparseOptimizer();
#ifdef MFEM_USE_MPI
   HiopNlpSparseOptimizer(MPI_Comm comm_);
#endif
   virtual ~HiopNlpSparseOptimizer();

   virtual void SetOptimizationProblem(const OptimizationProblem &prob);
   void SetNNZSparse(const size_type &_nnz_constr) { this->nnz_constr = _nnz_constr; }

   /// Solves the optimization problem with xt as initial guess.
   virtual void Mult(const Vector &xt, Vector &x) const;
};

} // mfem namespace

#endif //MFEM_USE_HIOP
#endif //MFEM_HIOP_SPARSE guard
