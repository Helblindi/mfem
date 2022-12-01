//                                MFEM Obstacle Problem
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double spherical_obstacle(const Vector &pt);
void spherical_obstacle_gradient(const Vector &pt, Vector &grad);
double exact_solution_obstacle(const Vector &pt);
double exact_solution_biactivity(const Vector &pt);
double load_biactivity(const Vector &pt);

class LogarithmGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   Coefficient *obstacle;
   double min_val;

public:
   LogarithmGridFunctionCoefficient(GridFunction &u_, Coefficient &obst_, double min_val_=-1e6)
      : u(&u_), obstacle(&obst_), min_val(min_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class ExponentialGridFunctionCoefficient : public Coefficient
{
protected:
   GridFunction *u; // grid function
   Coefficient *obstacle;
   double min_val;
   double max_val;

public:
   ExponentialGridFunctionCoefficient(GridFunction &u_, Coefficient &obst_, double min_val_=1e-8, double max_val_=1e8)
      : u(&u_), obstacle(&obst_), min_val(min_val_), max_val(max_val_) { }

   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

int main(int argc, char *argv[])
{
   MPI_Session mpi;
   int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();

   // 1. Parse command-line options.
   const char *mesh_file = "./disk.mesh";
   int order = 1;
   bool visualization = true;
   bool adaptive = false;
   int max_it = 1;
   int ref_levels = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  "isoparametric space.");
   args.AddOption(&ref_levels, "-r", "--refs",
                  "Number of h-refinements.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of iterations");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&adaptive, "-amr", "--amr", "-n-amr",
                  "--no-amr",
                  "Enable or disable AMR.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   for (int l = 0; l < ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   mesh.SetCurvature(2);
   mesh.EnsureNCMesh();

   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   H1_FECollection H1fec(order, dim);
   ParFiniteElementSpace H1fes(&pmesh, &H1fec);

   L2_FECollection L2fec(order-1, dim);
   ParFiniteElementSpace L2fes(&pmesh, &L2fec);

   cout << "Number of finite element unknowns: "
        << H1fes.GetTrueVSize() << endl;

   Array<int> ess_tdof_list;
   Array<int> ess_bdr(pmesh.bdr_attributes.Max());
   if (pmesh.bdr_attributes.Size())
   {
      ess_bdr = 1;
      H1fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   auto sol_func = [](const Vector &x)
   {
      double val = 1.0;
      for (int i=0; i<x.Size(); i++)
      {
         val *= sin(M_PI*x(i));
      }
      return val + 2.0;
   };

   auto rhs_func = [](const Vector &x)
   {
      double val = 1.0;
      for (int i=0; i<x.Size(); i++)
      {
         val *= sin(M_PI*x(i));
      }
      return (x.Size()*pow(M_PI,2)) * val  + log(val + 2.0);
   };

   ParGridFunction u_gf(&H1fes);
   ParGridFunction delta_psi_gf(&L2fes);
   delta_psi_gf = 0.0;

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.

   auto IC_func = [](const Vector &x)
   {
      double r0 = 1.0;
      double rr = 0.0;
      for (int i=0; i<x.Size(); i++)
      {
         rr += x(i)*x(i);
      }
      return r0*r0 - rr;
   };
   ConstantCoefficient one(1.0);
   ConstantCoefficient minus_one(-1.0);
   ConstantCoefficient zero(0.0);

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   ParGridFunction u_old_gf(&H1fes);
   ParGridFunction psi_old_gf(&L2fes);
   ParGridFunction psi_gf(&L2fes);
   u_old_gf = 0.0;
   psi_old_gf = 0.0;

   /////////// Example 1   
   // FunctionCoefficient f(rhs_func);
   // FunctionCoefficient IC_coef(IC_func);
   // ConstantCoefficient bdry_coef(0.1);
   // ConstantCoefficient obstacle(0.0);
   // SumCoefficient bdry_funcoef(bdry_coef, IC_coef);
   // u_gf.ProjectCoefficient(bdry_funcoef);
   // double alpha0 = 0.1;

   /////////// Example 2
   FunctionCoefficient exact_coef(exact_solution_obstacle);
   FunctionCoefficient IC_coef(IC_func);
   ConstantCoefficient f(0.0);
   FunctionCoefficient obstacle(spherical_obstacle);
   u_gf.ProjectCoefficient(IC_coef);
   u_old_gf = u_gf;
   double alpha0 = 0.1;
   // double alpha0 = 1.0;

   /////////// Example 3
   // u_gf = 0.5;
   // FunctionCoefficient exact_coef(exact_solution_biactivity);
   // FunctionCoefficient f(load_biactivity);
   // FunctionCoefficient bdry_coef(exact_solution_biactivity);
   // ConstantCoefficient obstacle(0.0);
   // u_gf.ProjectBdrCoefficient(bdry_coef, ess_bdr);
   // double alpha0 = 1.0;

   /////////// Newton TEST
   // FunctionCoefficient f(rhs_func);
   // ConstantCoefficient obstacle(0.0);
   // FunctionCoefficient sol(sol_func);
   // u_gf.ProjectCoefficient(sol);
   // u_old_gf = 0.0;
   // double alpha0 = 1.0;

   LogarithmGridFunctionCoefficient ln_u(u_gf, obstacle);
   psi_gf.ProjectCoefficient(ln_u);
   psi_old_gf = psi_gf;
   // psi_old_gf = 0.0;

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock.precision(8);

   socketstream psi_sock(vishost, visport);
   psi_sock.precision(8);

   LpErrorEstimator estimator(2, exact_coef, u_gf);
   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.9);

   // 12. Iterate
   int k;
   int total_iterations = 0;
   double tol = 1e-8;
   double increment_u = 0.1;
   double comp;
   double entropy;

   for (k = 0; k < max_it; k++)
   {
      // double alpha = alpha0 / sqrt(k+1);
      // double alpha = alpha0 * sqrt(k+1);
      double alpha = alpha0;
      // double alpha = alpha0 * (k+1);
      // alpha *= 2;

      ParGridFunction u_tmp(&H1fes);
      u_tmp = u_old_gf;

      mfem::out << "\nOUTER ITERATION " << k+1 << endl;

      int j;
      for ( j = 0; j < 10; j++)
      {
         total_iterations++;
         // A. Assembly
         
         // // MD
         // double c1 = 1.0;
         // double c2 = 1.0 - alpha;

         // // IMD
         // double c1 = 1.0 + alpha;
         // double c2 = 1.0;

         // // Other
         double c1 = alpha;
         double c2 = 0.0;

         ConstantCoefficient c1_cf(c1);

         ParLinearForm b(&H1fes);

         ExponentialGridFunctionCoefficient exp_psi(psi_gf, zero);
         ParGridFunction minus_psi_gf(&L2fes);
         minus_psi_gf = psi_gf;
         minus_psi_gf *= -1.0;
         ExponentialGridFunctionCoefficient exp_minus_psi(minus_psi_gf, zero);
         ProductCoefficient alpha_f(alpha, f);
         GridFunctionCoefficient psi_cf(&psi_gf);
         GridFunctionCoefficient psi_old_cf(&psi_old_gf);
         SumCoefficient psi_old_minus_psi(psi_old_cf, psi_cf, 1.0, -1.0);
         ProductCoefficient phi_exp_minus_psi(obstacle, exp_minus_psi);

         b.AddDomainIntegrator(new DomainLFIntegrator(alpha_f));
         b.AddDomainIntegrator(new DomainLFIntegrator(psi_old_minus_psi));
         b.AddDomainIntegrator(new DomainLFIntegrator(phi_exp_minus_psi));
         b.AddDomainIntegrator(new DomainLFIntegrator(one));
         b.Assemble();

         ParBilinearForm a(&H1fes);
         a.AddDomainIntegrator(new DiffusionIntegrator(c1_cf));
         a.AddDomainIntegrator(new MassIntegrator(exp_minus_psi));
         a.Assemble();

         OperatorPtr A;
         Vector B, X;
         a.FormLinearSystem(ess_tdof_list, u_gf, b, A, X, B);

         // CGSolver cg(MPI_COMM_WORLD);
         // cg.SetRelTol(1e-12);
         // cg.SetMaxIter(2000);
         // cg.SetPrintLevel(-1);
         // HypreBoomerAMG prec;
         // prec.SetPrintLevel(-1);
         // cg.SetPreconditioner(prec);
         // cg.SetOperator(*A);
         // cg.Mult(B, X);

         MUMPSSolver mumps;
         mumps.SetPrintLevel(0);
         mumps.SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_POSITIVE_DEFINITE);
         mumps.SetOperator(*A);
         mumps.Mult(B, X);

         a.RecoverFEMSolution(X, b, u_gf);

         u_tmp -= u_gf;
         double Newton_update_size = u_tmp.ComputeL2Error(zero);
         u_tmp = u_gf;


         GridFunctionCoefficient u_cf(&u_gf);
         SumCoefficient u_cf_minus_phi(u_cf, obstacle, 1.0, -1.0);
         ProductCoefficient u_cf_minus_phi_exp_minus_psi(u_cf_minus_phi,exp_minus_psi);

         // ParBilinearForm c(&H1fes);
         // c.AddDomainIntegrator(new MassIntegrator(one));
         // c.Assemble();

         // ParLinearForm d(&H1fes);
         // d.AddDomainIntegrator(new DomainLFIntegrator(u_cf_minus_phi_phi_exp_minus_psine));
         // d.AddDomainIntegrator(new DomainLFIntegrator(minus_one));
         // d.Assemble();
         
         delta_psi_gf.ProjectCoefficient(u_cf_minus_phi_exp_minus_psi);
         delta_psi_gf -= 1.0;

         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock << "solution\n" << pmesh << u_gf << "window_title 'Discrete solution'" << flush;
         mfem::out << endl;
         // psi_sock << "solution\n" << pmesh << delta_psi_gf << "window_title 'delta psi'" << flush;

         double gamma = 0.1;
         // double gamma = 1.0;
         delta_psi_gf *= gamma;
         psi_gf += delta_psi_gf;

         // for (int i = 0; i < delta_psi_gf.Size(); i++)
         // {
         //    if (psi_gf[i] < -7.0) { psi_gf[i] = -7.0; }
         // }

         psi_sock << "solution\n" << pmesh << psi_gf << "window_title 'psi'" << flush;
         
         mfem::out << "Newton_update_size = " << Newton_update_size/gamma << endl;

         // double update_tol = 1e-10;
         if (Newton_update_size/gamma < increment_u)
         // if (Newton_update_size < increment_u/10.0)
         {
            mfem::out << "break" << endl;
            break;
         }
      }
      // if (j > 5)
      // {
      //    alpha0 /= 2.0;
      // }
      mfem::out << "Number of Newton iterations = " << j+1 << endl;
      
      u_tmp = u_gf;
      u_tmp -= u_old_gf;
      increment_u = u_tmp.ComputeL2Error(zero);

      mfem::out << "|| u_h - u_h_prvs || = " << increment_u << endl;

      // delta_psi_gf = psi_gf;
      // delta_psi_gf -= psi_old_gf;
      // delta_psi_gf = 0.0;

      u_old_gf = u_gf;
      psi_old_gf = psi_gf;

      {
         // CHECK COMPLIMENTARITY | a(u_h, \phi_h - u_h) - (f, \phi_h - u_h) | < tol.
         // TODO: Need to check this with Socratis

         ParLinearForm b(&H1fes);
         b.AddDomainIntegrator(new DomainLFIntegrator(f));
         b.Assemble();

         ParBilinearForm a(&H1fes);
         a.AddDomainIntegrator(new DiffusionIntegrator());
         a.Assemble();
         a.EliminateEssentialBC(ess_bdr, u_gf, b, mfem::Operator::DIAG_ONE);
         a.Finalize();

         ParGridFunction obstacle_gf(&H1fes);
         obstacle_gf.ProjectCoefficient(obstacle);
         obstacle_gf -= u_gf;
         
         comp = a.InnerProduct(u_gf, obstacle_gf);
         comp -= b(obstacle_gf);
         comp = abs(comp);
         mfem::out << "|< phi - u_h, A u_h - f >| = " << comp << endl;


         ParLinearForm e(&H1fes);

         LogarithmGridFunctionCoefficient ln_u(u_gf, obstacle);
         ConstantCoefficient neg_one(-1.0);

         e.AddDomainIntegrator(new DomainLFIntegrator(ln_u));
         e.AddDomainIntegrator(new DomainLFIntegrator(neg_one));
         e.Assemble();

         // entropy = -( a.InnerProduct(u_gf, u_gf) );
         entropy = e(obstacle_gf);
         mfem::out << "entropy = " << entropy << endl;

      }

      if (increment_u < tol || k == max_it-1)
      // if (comp < tol)
      {
         break;
      }

      // estimator.GetLocalErrors();
      // double total_error = estimator.GetTotalError();
      double total_error = u_gf.ComputeL2Error(exact_coef);

      // mfem::out << "total_error = " << total_error << endl;
      // mfem::out << "increment_u = " << increment_u << endl;

      // 14. Send the solution by socket to a GLVis server.
      // u_old = u;
      // ExponentialGridFunctionCoefficient exp_psi(psi, obstacle);
      // u.ProjectCoefficient(exp_psi);
      // sol_sock << "solution\n" << mesh << exp_psi << "window_title 'Discrete solution'" << flush;
      // sol_sock << "solution\n" << mesh << u_gf << "window_title 'Discrete solution'" << flush;
      
   }

   mfem::out << "\n Outer iterations: " << k+1
             << "\n Total iterations: " << total_iterations
             << "\n dofs:             " << H1fes.GetTrueVSize()
             << endl;

   // 14. Exact solution.
   if (visualization)
   {
      socketstream err_sock(vishost, visport);
      err_sock.precision(8);

      ParGridFunction error(&H1fes);
      error = 0.0;
      error.ProjectCoefficient(exact_coef);
      error -= u_gf;

      mfem::out << "\n Final L2-error (|| u - u_h||) = " << u_gf.ComputeL2Error(exact_coef) << endl;

      err_sock << "parallel " << num_procs << " " << myid << "\n";
      err_sock << "solution\n" << pmesh << error << "window_title 'Error'"  << flush;
   }
   
   return 0;
}

double LogarithmGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip) - obstacle->Eval(T, ip);
   return max(min_val, log(val));
}

double ExponentialGridFunctionCoefficient::Eval(ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   MFEM_ASSERT(u != NULL, "grid function is not set");

   double val = u->GetValue(T, ip);
   return min(max_val, max(min_val, exp(val) + obstacle->Eval(T, ip)));
}

double spherical_obstacle(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double r = sqrt(x*x + y*y);
   double r0 = 0.5;
   double beta = 0.9;

   double b = r0*beta;
   double tmp = sqrt(r0*r0 - b*b);
   double B = tmp + b*b/tmp;
   double C = -b/tmp;

   if (r > b)
   {
      return B + r * C;
   }
   else
   {
      return sqrt(r0*r0 - r*r);
   }
}

void spherical_obstacle_gradient(const Vector &pt, Vector &grad)
{
   double x = pt(0), y = pt(1);
   double r = sqrt(x*x + y*y);
   double r0 = 0.5;
   double beta = 0.9;

   double b = r0*beta;
   double tmp = sqrt(r0*r0-b*b);
   double C = -b/tmp;

   if (r > b)
   {
      grad(0) = C * x / r;
      grad(1) = C * y / r;
   }
   else
   {
      grad(0) = - x / sqrt( r0*r0 - r*r );
      grad(1) = - y / sqrt( r0*r0 - r*r );
   }
}

double exact_solution_obstacle(const Vector &pt)
{
   double x = pt(0), y = pt(1);
   double r = sqrt(x*x + y*y);
   double r0 = 0.5;
   double a =  0.348982574111686;
   double A = -0.340129705945858;

   if (r > a)
   {
      return A * log(r);
   }
   else
   {
      return sqrt(r0*r0-r*r);
   }
}

double exact_solution_biactivity(const Vector &pt)
{
   double x = pt(0);

   if (x > 0.0)
   {
      return x*x;
   }
   else
   {
      return 0.0;
   }
}

double load_biactivity(const Vector &pt)
{
   double x = pt(0);

   if (x > 0.0)
   {
      return -2.0;
   }
   else
   {
      return 0.0;
   }
}

// double IC_biactivity(const Vector &pt)
// {
//    double x = pt(0);
//    return x*x;
// }