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

#include "tmop_tools.hpp"
#include "nonlinearform.hpp"
#include "pnonlinearform.hpp"
#include "../general/osockstream.hpp"

namespace mfem
{

using namespace mfem;

void AdvectorCG::SetInitialField(const Vector &init_nodes,
                                 const Vector &init_field)
{
   nodes0 = init_nodes;
   field0 = init_field;
}

void AdvectorCG::ComputeAtNewPosition(const Vector &new_nodes,
                                      Vector &new_field,
                                      int new_nodes_ordering)
{
   FiniteElementSpace *space = fes;
#ifdef MFEM_USE_MPI
   if (pfes) { space = pfes; }
#endif
   int fes_ordering = space->GetOrdering(),
       ncomp = space->GetVDim();

   // TODO: Implement for AMR meshes.
   const int pnt_cnt = field0.Size() / ncomp;

   new_field = field0;
   Vector new_field_temp;
   for (int i = 0; i < ncomp; i++)
   {
      if (fes_ordering == Ordering::byNODES)
      {
         new_field_temp.MakeRef(new_field, i*pnt_cnt, pnt_cnt);
      }
      else
      {
         new_field_temp.SetSize(pnt_cnt);
         for (int j = 0; j < pnt_cnt; j++)
         {
            new_field_temp(j) = new_field(i + j*ncomp);
         }
      }
      ComputeAtNewPositionScalar(new_nodes, new_field_temp);
      if (fes_ordering == Ordering::byVDIM)
      {
         for (int j = 0; j < pnt_cnt; j++)
         {
            new_field(i + j*ncomp) = new_field_temp(j);
         }
      }
   }

   field0 = new_field;
   nodes0 = new_nodes;
}

void AdvectorCG::ComputeAtNewPositionScalar(const Vector &new_nodes,
                                            Vector &new_field)
{
   Mesh *m = mesh;
#ifdef MFEM_USE_MPI
   if (pmesh) { m = pmesh; }
#endif

   MFEM_VERIFY(m != NULL, "No mesh has been given to the AdaptivityEvaluator.");

   // This will be used to move the positions.
   GridFunction *mesh_nodes = m->GetNodes();
   *mesh_nodes = nodes0;
   double minv = new_field.Min(), maxv = new_field.Max();

   // Velocity of the positions.
   GridFunction u(mesh_nodes->FESpace());
   subtract(new_nodes, nodes0, u);

   // Define a scalar FE space for the solution, and the advection operator.
   TimeDependentOperator *oper = NULL;
   FiniteElementSpace *fess = NULL;
#ifdef MFEM_USE_MPI
   ParFiniteElementSpace *pfess = NULL;
#endif
   if (fes)
   {
      fess = new FiniteElementSpace(fes->GetMesh(), fes->FEColl(), 1);
      oper = new SerialAdvectorCGOper(nodes0, u, *fess, al);
   }
#ifdef MFEM_USE_MPI
   else if (pfes)
   {
      pfess = new ParFiniteElementSpace(pfes->GetParMesh(), pfes->FEColl(), 1);
      oper  = new ParAdvectorCGOper(nodes0, u, *pfess, al, opt_mt);
   }
#endif
   MFEM_VERIFY(oper != NULL,
               "No FE space has been given to the AdaptivityEvaluator.");
   ode_solver.Init(*oper);

   // Compute some time step [mesh_size / speed].
   double h_min = std::numeric_limits<double>::infinity();
   for (int i = 0; i < m->GetNE(); i++)
   {
      h_min = std::min(h_min, m->GetElementSize(i));
   }
   double v_max = 0.0;
   const int s = new_field.Size();

   u.HostReadWrite();
   for (int i = 0; i < s; i++)
   {
      double vel = 0.;
      for (int j = 0; j < m->Dimension(); j++)
      {
         vel += u(i+j*s)*u(i+j*s);
      }
      v_max = std::max(v_max, vel);
   }

#ifdef MFEM_USE_MPI
   if (pfes)
   {
      double v_loc = v_max, h_loc = h_min;
      MPI_Allreduce(&v_loc, &v_max, 1, MPI_DOUBLE, MPI_MAX, pfes->GetComm());
      MPI_Allreduce(&h_loc, &h_min, 1, MPI_DOUBLE, MPI_MIN, pfes->GetComm());
   }
#endif

   if (v_max == 0.0) // No need to change the field.
   {
      delete oper;
      delete fess;
#ifdef MFEM_USE_MPI
      delete pfess;
#endif
      return;
   }

   v_max = std::sqrt(v_max);
   double dt = dt_scale * h_min / v_max;

   double t = 0.0;
   bool last_step = false;
   while (!last_step)
   {
      if (t + dt >= 1.0)
      {
         dt = 1.0 - t;
         last_step = true;
      }
      ode_solver.Step(new_field, t, dt);
   }

   double glob_minv = minv,
          glob_maxv = maxv;
#ifdef MFEM_USE_MPI
   if (pfes)
   {
      MPI_Allreduce(&minv, &glob_minv, 1, MPI_DOUBLE, MPI_MIN, pfes->GetComm());
      MPI_Allreduce(&maxv, &glob_maxv, 1, MPI_DOUBLE, MPI_MAX, pfes->GetComm());
   }
#endif

   // Trim the overshoots and undershoots.
   new_field.HostReadWrite();
   for (int i = 0; i < s; i++)
   {
      if (new_field(i) < glob_minv) { new_field(i) = glob_minv; }
      if (new_field(i) > glob_maxv) { new_field(i) = glob_maxv; }
   }

   delete oper;
   delete fess;
#ifdef MFEM_USE_MPI
   delete pfess;
#endif
}

SerialAdvectorCGOper::SerialAdvectorCGOper(const Vector &x_start,
                                           GridFunction &vel,
                                           FiniteElementSpace &fes,
                                           AssemblyLevel al)
   : TimeDependentOperator(fes.GetVSize()),
     x0(x_start), x_now(*fes.GetMesh()->GetNodes()),
     u(vel), u_coeff(&u), M(&fes), K(&fes), al(al)
{
   ConvectionIntegrator *Kinteg = new ConvectionIntegrator(u_coeff);
   K.AddDomainIntegrator(Kinteg);
   K.SetAssemblyLevel(al);
   K.Assemble(0);
   K.Finalize(0);

   MassIntegrator *Minteg = new MassIntegrator;
   M.AddDomainIntegrator(Minteg);
   M.SetAssemblyLevel(al);
   M.Assemble(0);
   M.Finalize(0);
}

void SerialAdvectorCGOper::Mult(const Vector &ind, Vector &di_dt) const
{
   // Move the mesh.
   const double t = GetTime();
   add(x0, t, u, x_now);
   K.FESpace()->GetMesh()->NodesUpdated();

   // Assemble on the new mesh.
   K.BilinearForm::operator=(0.0);
   K.Assemble();
   Vector rhs(K.Size());
   K.Mult(ind, rhs);
   M.BilinearForm::operator=(0.0);
   M.Assemble();

   di_dt = 0.0;
   CGSolver lin_solver;
   Solver *prec = nullptr;
   Array<int> ess_tdof_list;
   if (al == AssemblyLevel::PARTIAL)
   {
      prec = new OperatorJacobiSmoother(M, ess_tdof_list);
      lin_solver.SetOperator(M);
   }
   else
   {
      prec = new DSmoother(M.SpMat());
      lin_solver.SetOperator(M.SpMat());
   }
   lin_solver.SetPreconditioner(*prec);
   lin_solver.SetRelTol(1e-12); lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(100);
   lin_solver.SetPrintLevel(0);
   lin_solver.Mult(rhs, di_dt);

   delete prec;
}

#ifdef MFEM_USE_MPI
ParAdvectorCGOper::ParAdvectorCGOper(const Vector &x_start,
                                     GridFunction &vel,
                                     ParFiniteElementSpace &pfes,
                                     AssemblyLevel al,
                                     MemoryType mt)
   : TimeDependentOperator(pfes.GetVSize()),
     x0(x_start), x_now(*pfes.GetMesh()->GetNodes()),
     u(vel), u_coeff(&u), M(&pfes), K(&pfes), al(al)
{
   ConvectionIntegrator *Kinteg = new ConvectionIntegrator(u_coeff);
   if (al == AssemblyLevel::PARTIAL)
   {
      Kinteg->SetPAMemoryType(mt);
   }
   K.AddDomainIntegrator(Kinteg);
   K.SetAssemblyLevel(al);
   K.Assemble(0);
   K.Finalize(0);

   MassIntegrator *Minteg = new MassIntegrator;
   if (al == AssemblyLevel::PARTIAL)
   {
      Minteg->SetPAMemoryType(mt);
   }
   M.AddDomainIntegrator(Minteg);
   M.SetAssemblyLevel(al);
   M.Assemble(0);
   M.Finalize(0);
}

void ParAdvectorCGOper::Mult(const Vector &ind, Vector &di_dt) const
{
   // Move the mesh.
   const double t = GetTime();
   add(x0, t, u, x_now);
   K.ParFESpace()->GetParMesh()->NodesUpdated();

   // Assemble on the new mesh.
   K.BilinearForm::operator=(0.0);
   K.Assemble();
   ParGridFunction rhs(K.ParFESpace());
   K.Mult(ind, rhs);
   M.BilinearForm::operator=(0.0);
   M.Assemble();

   HypreParVector *RHS = rhs.ParallelAssemble();
   HypreParVector X(K.ParFESpace());
   X = 0.0;

   OperatorHandle Mop;
   Solver *prec = nullptr;
   Array<int> ess_tdof_list;
   if (al == AssemblyLevel::PARTIAL)
   {
      M.FormSystemMatrix(ess_tdof_list, Mop);
      prec = new OperatorJacobiSmoother(M, ess_tdof_list);
   }
   else
   {
      Mop.Reset(M.ParallelAssemble());
      prec = new HypreSmoother;
      static_cast<HypreSmoother*>(prec)->SetType(HypreSmoother::Jacobi, 1);
   }

   CGSolver lin_solver(M.ParFESpace()->GetParMesh()->GetComm());
   lin_solver.SetPreconditioner(*prec);
   lin_solver.SetOperator(*Mop);
   lin_solver.SetRelTol(1e-8);
   lin_solver.SetAbsTol(0.0);
   lin_solver.SetMaxIter(100);
   lin_solver.SetPrintLevel(0);
   lin_solver.Mult(*RHS, X);
   K.ParFESpace()->GetProlongationMatrix()->Mult(X, di_dt);

   delete RHS;
   delete prec;
}
#endif

#ifdef MFEM_USE_GSLIB
void InterpolatorFP::SetInitialField(const Vector &init_nodes,
                                     const Vector &init_field)
{
   nodes0 = init_nodes;
   Mesh *m = mesh;
#ifdef MFEM_USE_MPI
   if (pmesh) { m = pmesh; }
#endif
   m->SetNodes(nodes0);

   const double rel_bbox_el = 0.1;
   const double newton_tol  = 1.0e-12;
   const int npts_at_once   = 256;

   if (finder)
   {
      finder->FreeData();
      delete finder;
   }

   FiniteElementSpace *f = fes;
#ifdef MFEM_USE_MPI
   if (pfes)
   {
      f = pfes;
      finder = new FindPointsGSLIB(pfes->GetComm());
   }
   else { finder = new FindPointsGSLIB(); }
#else
   finder = new FindPointsGSLIB();
#endif
   finder->Setup(*m, rel_bbox_el, newton_tol, npts_at_once);

   field0_gf.SetSpace(f);
   field0_gf = init_field;
}

void InterpolatorFP::ComputeAtNewPosition(const Vector &new_nodes,
                                          Vector &new_field,
                                          int new_nodes_ordering)
{
   finder->Interpolate(new_nodes, field0_gf, new_field, new_nodes_ordering);
}

#endif

double TMOPNewtonSolver::ComputeScalingFactor(const Vector &x,
                                              const Vector &b) const
{
   const FiniteElementSpace *fes = NULL;
   double energy_in = 0.0;
#ifdef MFEM_USE_MPI
   const ParNonlinearForm *p_nlf = dynamic_cast<const ParNonlinearForm *>(oper);
   MFEM_VERIFY(!(parallel && p_nlf == NULL), "Invalid Operator subclass.");
   if (parallel)
   {
      fes = p_nlf->FESpace();
      energy_in = p_nlf->GetEnergy(x);
   }
#endif
   const bool serial = !parallel;
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   MFEM_VERIFY(!(serial && nlf == NULL), "Invalid Operator subclass.");
   if (serial)
   {
      fes = nlf->FESpace();
      energy_in = nlf->GetEnergy(x);
   }

   // Get the local prolongation of the solution vector.
   Vector x_out_loc(fes->GetVSize(),
                    (temp_mt == MemoryType::DEFAULT) ? Device::GetDeviceMemoryType() : temp_mt);
   if (serial)
   {
      const SparseMatrix *cP = fes->GetConformingProlongation();
      if (!cP) { x_out_loc = x; }
      else     { cP->Mult(x, x_out_loc); }
   }
#ifdef MFEM_USE_MPI
   else
   {
      fes->GetProlongationMatrix()->Mult(x, x_out_loc);
   }
#endif

   double scale = 1.0;
   double avg_surf_fit_err, max_surf_fit_err = 0.0;
   if (surf_fit_max_threshold > 0.0)
   {
      GetSurfaceFittingError(x_out_loc, avg_surf_fit_err, max_surf_fit_err);
      if (max_surf_fit_err < surf_fit_max_threshold &&
          surf_fit_converge_based_on_error)
      {
         if (print_options.iterations)
         {
            mfem::out << "TMOPNewtonSolver converged "
                      "based on the surface fitting error.\n";
         }
         scale = 0.0;
         return scale;
      }
   }
   if (adapt_inc_count >= max_adapt_inc_count)
   {
      if (print_options.iterations)
      {
         mfem::out << "TMOPNewtonSolver converged "
                   "based on max number of times surface fitting weight can "
                   "be increased. \n";
      }
      scale = 0.0;
      return scale;
   }

   // Check if the starting mesh (given by x) is inverted. Note that x hasn't
   // been modified by the Newton update yet.
   const double min_detT_in = ComputeMinDet(x_out_loc, *fes);
   const bool untangling = (min_detT_in <= 0.0) ? true : false;
   const double untangle_factor = 1.5;
   if (print_options.iterations)
   {
      mfem::out << "Min det of input mesh in line search: = " <<
                min_detT_in  << std::endl;
   }
   if (untangling)
   {
      // Needed for the line search below. The untangling metrics see this
      // reference to detect deteriorations.
      MFEM_VERIFY(min_det_ptr != NULL, " Initial mesh was valid, but"
                  " intermediate mesh is invalid. Contact TMOP Developers.");
      MFEM_VERIFY(min_detJ_threshold == 0.0,
                  "This setup is not supported. Contact TMOP Developers.");
      *min_det_ptr = untangle_factor * min_detT_in;
   }

   const bool have_b = (b.Size() == Height());

   Vector x_out(x.Size());
   bool x_out_ok = false;
   double energy_out = 0.0, min_detT_out;
   const double norm_in = Norm(r);

   const double detJ_factor = (solver_type == 1) ? 0.1 : 0.25;
   compute_metric_quantile_flag = false;
   // TODO:
   // - Customized line search for worst-quality optimization.
   // - What is the Newton exit criterion for worst-quality optimization?

   double scale_save = 0.0;
   double norm_out = 0.0;
   double avg_fit_err, max_fit_err = 0.0;


   // Perform the line search.
   for (int i = 0; i < 12; i++)
   {
      avg_fit_err = 0.0;
      max_fit_err = 0.0;
      // Update the mesh and get the L-vector in x_out_loc.
      add(x, -scale, c, x_out);
      if (serial)
      {
         const SparseMatrix *cP = fes->GetConformingProlongation();
         if (!cP) { x_out_loc = x_out; }
         else     { cP->Mult(x_out, x_out_loc); }
      }
#ifdef MFEM_USE_MPI
      else { fes->GetProlongationMatrix()->Mult(x_out, x_out_loc); }
#endif

      // Check the changes in detJ.
      min_detT_out = ComputeMinDet(x_out_loc, *fes);
      if (untangling == false && min_detT_out <= 1.0*min_detJ_threshold)
      {
         // No untangling, and detJ got negative (or small) -- no good.
         if (print_options.iterations)
         {
            mfem::out << "Scale = " << scale << " " << min_detT_out <<
                      " Neg det(J) found.\n";
         }
         scale *= detJ_factor; continue;
      }
      if (untangling == true && min_detT_out < *min_det_ptr)
      {
         // Untangling, and detJ got even more negative -- no good.
         if (print_options.iterations)
         {
            mfem::out << "Scale = " << scale << " Neg det(J) decreased.\n";
         }
         scale *= detJ_factor; continue;
      }
      if (worst_skewness > 0.0)
      {
         Vector worst_skew = ComputeWorstSkew(x_out_loc, *fes);
         if (worst_skew(0) > worst_skewness)
         {
            if (print_options.iterations)
            {
               mfem::out << "Scale = " << scale << " Worst skewness breached.\n";
               mfem::out << worst_skew(0)*180.0/M_PI << " " << worst_skewness*180.0/M_PI <<
                         std::endl;
            }
            scale *= detJ_factor; continue;
         }
      }

      // Skip the energy and residual checks when we're untangling. The
      // untangling metrics change their denominators, which can affect the
      // energy and residual, so their increase/decrease is not relevant.
      if (untangling) { x_out_ok = true; break; }

      // Check the changes in total energy.
      ProcessNewState(x_out);

      if (surf_fit_max_threshold > 0.0)
      {
         GetSurfaceFittingError(x_out_loc, avg_fit_err, max_fit_err);
      }
      if (surf_fit_max_threshold > 0.0 && max_fit_err >= 1.2*max_surf_fit_err)
      {
         if (print_options.iterations)
         {
            mfem::out << "Scale = " << scale <<  " " <<
                      max_fit_err << " " << max_surf_fit_err << " Surf fit err increased.\n";
         }
         //         scale *= std::min(0.25, max_surf_fit_err/max_fit_err); continue;
         scale *= 0.25; continue;
      }

      if (serial)
      {
         energy_out = nlf->GetGridFunctionEnergy(x_out_loc);
      }
#ifdef MFEM_USE_MPI
      else
      {
         energy_out = p_nlf->GetParGridFunctionEnergy(x_out_loc);
      }
#endif
      if (energy_out > energy_in + 0.2*fabs(energy_in) ||
          std::isnan(energy_out) != 0)
      {
         if (print_options.iterations)
         {
            mfem::out << "Scale = " << scale << " Increasing energy: "
                      << energy_in << " --> " << energy_out << '\n';
         }
         //         scale *= std::min(0.25, energy_in/energy_out); continue;
         scale *= 0.25; continue;
      }

      // Check the changes in the Newton residual.
      oper->Mult(x_out, r);
      if (have_b) { r -= b; }
      norm_out = Norm(r);

      if (norm_out > 1.1*norm_in)
      {
         if (print_options.iterations)
         {
            mfem::out << "Scale = " << scale << " Norm increased: "
                      << norm_in << " --> " << norm_out << '\n';
         }
         //         scale *= std::min(0.25, norm_in/norm_out); continue;
         scale *= 0.25; continue;
      }
      else if (norm_out > 1.0*norm_in)
      {
         if (scale_save > 0.0)
         {
            if (print_options.iterations)
            {
               mfem::out << "Scale = " << scale << " Norm increase accepted: "
                         << norm_in << " --> " << norm_out << '\n';
            }
            x_out_ok = true; break;
         }
         else
         {
            if (print_options.iterations)
            {
               mfem::out << "Scale = " << scale << " Norm increased: "
                         << norm_in << " --> " << norm_out << '\n';
            }
            scale_save = scale;
            scale *= 0.25; continue;
         }
      }
      else
      {
         if (print_options.iterations)
         {
            mfem::out << "Scale = " << scale << " Norm decreased: "
                      << norm_in << " --> " << norm_out << '\n';
         }
         x_out_ok = true; break;
      }
   } // end line search

   if (untangling)
   {
      // Update the global min detJ. Untangling metrics see this min_det_ptr.
      if (min_detT_out > 0.0)
      {
         *min_det_ptr = 0.0;
         if (print_options.summary || print_options.iterations ||
             print_options.first_and_last)
         { mfem::out << "The mesh has been untangled at the used points!\n"; }
      }
      else { *min_det_ptr = untangle_factor * min_detT_out; }
   }

   if (print_options.summary || print_options.iterations ||
       print_options.first_and_last)
   {
      if (untangling)
      {
         mfem::out << "Min det(T) change: "
                   << min_detT_in << " -> " << min_detT_out
                   << " with " << scale << " scaling.\n";
      }
      else
      {
         mfem::out << "Energy decrease: "
                   << energy_in << " --> " << energy_out << " or "
                   << (energy_in - energy_out) / energy_in * 100.0
                   << "% with " << scale << " scaling.\n";
      }
   }

   if (x_out_ok == false) { scale = 0.0; }

   if (surf_fit_scale_factor > 0.0) { update_surf_fit_coeff = true; }
   compute_metric_quantile_flag = true;

   if (print_options.iterations)
   {
      mfem::out << "Line search info energy/norm/mindet in and out: " << " " <<
                energy_in << " " << energy_out << " " <<
                norm_in << " " << norm_out << " " <<
                min_detT_in << " " << min_detT_out << std::endl;
      mfem::out << "Line search info avg/max fitting error in and out: " << " " <<
                avg_surf_fit_err << " " << max_surf_fit_err << " " <<
                avg_fit_err << " " << max_fit_err << " " << std::endl;
   }

   return scale;
}

void TMOPNewtonSolver::UpdateSurfaceFittingWeight(double factor) const
{
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   const Array<NonlinearFormIntegrator*> &integs = *nlf->GetDNFI();
   TMOP_Integrator *ti  = NULL;
   TMOPComboIntegrator *co = NULL;
   for (int i = 0; i < integs.Size(); i++)
   {
      ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti)
      {
         ti->UpdateSurfaceFittingWeight(factor);
      }
      co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co)
      {
         Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
         for (int j = 0; j < ati.Size(); j++)
         {
            ati[j]->UpdateSurfaceFittingWeight(factor);
         }
      }
   }
}

void TMOPNewtonSolver::SaveSurfaceFittingWeight() const
{
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   const Array<NonlinearFormIntegrator*> &integs = *nlf->GetDNFI();
   TMOP_Integrator *ti  = NULL;
   TMOPComboIntegrator *co = NULL;
   for (int i = 0; i < integs.Size(); i++)
   {
      ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti)
      {
         ti->SaveSurfaceFittingWeight();
      }
      co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co)
      {
         Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
         for (int j = 0; j < ati.Size(); j++)
         {
            ati[j]->SaveSurfaceFittingWeight();
         }
      }
   }
}

void TMOPNewtonSolver::GetSurfaceFittingWeight(Array<double> &weights) const
{
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   const Array<NonlinearFormIntegrator*> &integs = *nlf->GetDNFI();
   TMOP_Integrator *ti  = NULL;
   TMOPComboIntegrator *co = NULL;
   weights.SetSize(0);
   double weight;

   for (int i = 0; i < integs.Size(); i++)
   {
      ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti && ti->IsSurfaceFittingEnabled())
      {
         weight = ti->GetSurfaceFittingWeight();
         weights.Append(weight);
      }
      co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co)
      {
         Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
         for (int j = 0; j < ati.Size(); j++)
         {
            if (ati[j]->IsSurfaceFittingEnabled())
            {
               weight = ati[j]->GetSurfaceFittingWeight();
               weights.Append(weight);
            }
         }
      }
   }
}

void TMOPNewtonSolver::GetSurfaceFittingError(const Vector &x_loc,
                                              double &err_avg,
                                              double &err_max) const
{
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   const Array<NonlinearFormIntegrator*> &integs = *nlf->GetDNFI();
   TMOP_Integrator *ti  = NULL;
   TMOPComboIntegrator *co = NULL;

   err_avg = 0.0;
   err_max = 0.0;
   double err_avg_loc, err_max_loc;
   for (int i = 0; i < integs.Size(); i++)
   {
      ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti)
      {
         if (ti->IsSurfaceFittingEnabled())
         {
            ti->GetSurfaceFittingErrors(x_loc, err_avg_loc, err_max_loc);
            err_avg = std::fmax(err_avg_loc, err_avg);
            err_max = std::fmax(err_max_loc, err_max);
         }
      }
      co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co)
      {
         Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
         for (int j = 0; j < ati.Size(); j++)
         {
            if (ati[j]->IsSurfaceFittingEnabled())
            {
               ati[j]->GetSurfaceFittingErrors(x_loc, err_avg_loc, err_max_loc);
               err_avg = std::fmax(err_avg_loc, err_avg);
               err_max = std::fmax(err_max_loc, err_max);
            }
         }
      }
   }
}

void TMOPNewtonSolver::ProcessNewState(const Vector &x) const
{
   const NonlinearForm *nlf = dynamic_cast<const NonlinearForm *>(oper);
   const Array<NonlinearFormIntegrator*> &integs = *nlf->GetDNFI();

   // Reset the update flags of all TargetConstructors. This is done to avoid
   // repeated updates of shared TargetConstructors.
   TMOP_Integrator *ti  = NULL;
   TMOPComboIntegrator *co = NULL;
   DiscreteAdaptTC *dtc = NULL;
   for (int i = 0; i < integs.Size(); i++)
   {
      ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti)
      {
         dtc = ti->GetDiscreteAdaptTC();
         if (dtc) { dtc->ResetUpdateFlags(); }
      }
      co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co)
      {
         Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
         for (int j = 0; j < ati.Size(); j++)
         {
            dtc = ati[j]->GetDiscreteAdaptTC();
            if (dtc) { dtc->ResetUpdateFlags(); }
         }
      }
   }

   Vector x_loc;
   const FiniteElementSpace *x_fes = nullptr;
   if (parallel)
   {
#ifdef MFEM_USE_MPI
      const ParNonlinearForm *pnlf =
         dynamic_cast<const ParNonlinearForm *>(oper);

      x_fes = pnlf->ParFESpace();
      x_loc.SetSize(x_fes->GetVSize());
      x_fes->GetProlongationMatrix()->Mult(x, x_loc);
#endif
   }
   else
   {
      x_fes = nlf->FESpace();
      const Operator *P = nlf->GetProlongation();
      if (P)
      {
         x_loc.SetSize(P->Height());
         P->Mult(x,x_loc);
      }
      else { x_loc = x; }
   }

   double min_det = ComputeMinDet(x_loc, *x_fes);
   // Get array with surface fitting weights.
   Array<double> fitweights;
   GetSurfaceFittingWeight(fitweights);

   for (int i = 0; i < integs.Size(); i++)
   {
      ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti)
      {
         ti->UpdateAfterMeshPositionChange(x_loc, *x_fes);
         if (compute_metric_quantile_flag)
         {
            ti->ComputeUntangleMetricQuantiles(x_loc, *x_fes);
         }
      }
      co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co)
      {
         Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
         for (int j = 0; j < ati.Size(); j++)
         {
            ati[j]->UpdateAfterMeshPositionChange(x_loc, *x_fes);
            if (compute_metric_quantile_flag)
            {
               ati[j]->ComputeUntangleMetricQuantiles(x_loc, *x_fes);
            }
         }
      }
   }


   // Constant coefficient associated with the surface fitting terms if
   // adaptive surface fitting is enabled. The idea is to increase the
   // coefficient if the surface fitting error does not sufficiently
   // decrease between subsequent TMOPNewtonSolver iterations.

   if (update_surf_fit_coeff)
   {
      // Get surface fitting errors.
      GetSurfaceFittingError(x_loc, surf_fit_err_avg, surf_fit_err_max);

      if (print_options.iterations)
      {
         mfem::out << "Avg/Max surface fitting error: " <<
                   surf_fit_err_avg << " " <<
                   surf_fit_err_max << "\n";
         mfem::out << "Min/Max surface fitting weight: " <<
                   fitweights.Min() << " " << fitweights.Max() << "\n";
      }

      bool max_for_threshold = false;
      double change_surf_fit_err = surf_fit_err_avg_prvs-surf_fit_err_avg;
      double rel_change_surf_fit_err = change_surf_fit_err/surf_fit_err_avg_prvs;
      if (max_for_threshold)
      {
         change_surf_fit_err = surf_fit_err_avg_prvs-surf_fit_err_max;
         rel_change_surf_fit_err = change_surf_fit_err/surf_fit_err_avg_prvs;
      }

      // Increase the surface fitting coefficient if the surface fitting error
      // does not decrease sufficiently.
      SaveSurfaceFittingWeight();

      if (rel_change_surf_fit_err < surf_fit_rel_change_threshold &&
          fitweights.Max() < weights_max_limit &&
          (surf_fit_converge_based_on_error ||
           surf_fit_err_max > surf_fit_max_threshold))
      {
         //      if (rel_change_surf_fit_err < surf_fit_rel_change_threshold)
         //      {
         //         if (fitweights.Max() < weights_max_limit && min_det > min_detJ_threshold)
         //         {
         if (print_options.iterations)
         {
            //               mfem::out << norm_ratio << " " << fitweights.Max() <<  " " <<
            //                         surf_fit_scale_factor <<
            //                         " morm ratio-max weight-factor\n";
         }
         double scale_factor = std::min(surf_fit_scale_factor,
                                        weights_max_limit/fitweights.Max());
         UpdateSurfaceFittingWeight(scale_factor);
         adapt_inc_count += 1;
         //         }
      }
      else
      {
         adapt_inc_count = 0;
      }
      surf_fit_err_avg_prvs = surf_fit_err_avg;
      if (max_for_threshold)
      {
         surf_fit_err_avg_prvs = surf_fit_err_max;
      }
      update_surf_fit_coeff = false;
   }
}

double GetWorstJacobianSkewness(DenseMatrix &J)
{
   double size;
   double skew = -100000;
   const int dim = J.Size();
   if (dim == 2)
   {
      Vector col1, col2;
      J.GetColumn(0, col1);
      J.GetColumn(1, col2);

      skew = std::atan2(J.Det(), col1 * col2);
      skew = std::fabs(skew);
   }
   else
   {
      Vector skewv(dim);
      Vector col1, col2, col3;
      J.GetColumn(0, col1);
      J.GetColumn(1, col2);
      J.GetColumn(2, col3);
      double len1 = col1.Norml2(),
             len2 = col2.Norml2(),
             len3 = col3.Norml2();

      Vector col1unit = col1,
             col2unit = col2,
             col3unit = col3;
      col1unit *= 1.0/len1;
      col2unit *= 1.0/len2;
      col3unit *= 1.0/len3;

      Vector crosscol12, crosscol13, crosscol23;
      col1.cross3D(col2, crosscol12);
      col1.cross3D(col3, crosscol13);
      col2.cross3D(col3, crosscol23);
      skewv(0) = std::atan2(crosscol12.Norml2(),col1*col2);
      skewv(1) = std::atan2(crosscol13.Norml2(),col1*col3);
      skewv(2) = std::atan2(col1*crosscol23,crosscol12*crosscol13);
      for (int d = 0; d < dim; d++)
      {
         skew = std::max(skew, std::fabs(skewv(d)));
      }
   }

   return skew;
}

Vector TMOPNewtonSolver::ComputeWorstSkew(const Vector &x_loc,
                                          const FiniteElementSpace &fes) const
{
   double skewmax = -1000;
   double skewmin = 1000;
   const Mesh *mesh = fes.GetMesh();
   const int NE = mesh->GetNE();
   const int dim = mesh->Dimension();
   DenseMatrix Jac(dim);
   Array<int> xdofs;
   for (int i = 0; i < NE; i++)
   {

      const int dof = fes.GetFE(i)->GetDof();
      DenseMatrix dshape(dof, dim), pos(dof, dim);
      Vector posV(pos.Data(), dof * dim);
      const int gt = fes.GetFE(i)->GetGeomType();

      fes.GetElementVDofs(i, xdofs);
      x_loc.GetSubVector(xdofs, posV);

      const IntegrationRule &irule = GetIntegrationRule(*fes.GetFE(i));
      const int nsp = irule.GetNPoints();
      for (int j = 0; j < nsp; j++)
      {
         fes.GetFE(i)->CalcDShape(irule.IntPoint(j), dshape);
         MultAtB(pos, dshape, Jac);
         double skew_q = GetWorstJacobianSkewness(Jac);
         //         skewmax = std::max(skewmax, skew_q);
         //         skewmin = std::min(skewmin, skew_q);
      }

      const IntegrationRule &ir2 = fes.GetFE(i)->GetNodes();
      for (int j = 0; j < ir2.GetNPoints(); j++)
      {
         fes.GetFE(i)->CalcDShape(ir2.IntPoint(j), dshape);
         MultAtB(pos, dshape, Jac);
         double skew_q = GetWorstJacobianSkewness(Jac);
         skewmax = std::max(skewmax, skew_q);
         skewmin = std::min(skewmin, skew_q);
      }
   }

   double skewmax_all = skewmax;
   double skewmin_all = skewmin;

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      auto p_nlf = dynamic_cast<const ParNonlinearForm *>(oper);
      MPI_Allreduce(&skewmax, &skewmax_all, 1, MPI_DOUBLE, MPI_MAX,
                    p_nlf->ParFESpace()->GetComm());
      MPI_Allreduce(&skewmin, &skewmin_all, 1, MPI_DOUBLE, MPI_MIN,
                    p_nlf->ParFESpace()->GetComm());
   }
#endif

   Vector skewall(2);
   skewall(0) = skewmax_all;
   skewall(1) = skewmin_all;
   return skewall;
}

double TMOPNewtonSolver::ComputeMinDet(const Vector &x_loc,
                                       const FiniteElementSpace &fes) const
{
   double min_detJ = infinity();
   const int NE = fes.GetNE(), dim = fes.GetMesh()->Dimension();
   Array<int> xdofs;
   DenseMatrix Jpr(dim);
   const bool mixed_mesh = fes.GetMesh()->GetNumGeometries(dim) > 1;
   if (NE > 0)
   {
      if (dim == 1 || mixed_mesh || UsesTensorBasis(fes) == false)
      {
         for (int i = 0; i < NE; i++)
         {
            const int dof = fes.GetFE(i)->GetDof();
            DenseMatrix dshape(dof, dim), pos(dof, dim);
            Vector posV(pos.Data(), dof * dim);

            fes.GetElementVDofs(i, xdofs);
            x_loc.GetSubVector(xdofs, posV);

            const IntegrationRule &irule = GetIntegrationRule(*fes.GetFE(i));
            const int nsp = irule.GetNPoints();
            for (int j = 0; j < nsp; j++)
            {
               fes.GetFE(i)->CalcDShape(irule.IntPoint(j), dshape);
               MultAtB(pos, dshape, Jpr);
               min_detJ = std::min(min_detJ, Jpr.Det());
            }

            const IntegrationRule &ir2 = fes.GetFE(i)->GetNodes();
            for (int j = 0; j < ir2.GetNPoints(); j++)
            {
               fes.GetFE(i)->CalcDShape(ir2.IntPoint(j), dshape);
               MultAtB(pos, dshape, Jpr);
               min_detJ = std::min(min_detJ, Jpr.Det());
            }
         }
      }
      else
      {
         min_detJ = dim == 2 ? MinDetJpr_2D(&fes, x_loc) :
                    dim == 3 ? MinDetJpr_3D(&fes, x_loc) : 0.0;
      }
   }
   double min_detT_all = min_detJ;

   if (fes.GetMesh()->GetNE() > 0)
   {
      const DenseMatrix &Wideal =
         Geometries.GetGeomToPerfGeomJac(fes.GetFE(0)->GetGeomType());
      min_detT_all /= Wideal.Det();
   }
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      auto p_nlf = dynamic_cast<const ParNonlinearForm *>(oper);
      MPI_Allreduce(&min_detJ, &min_detT_all, 1, MPI_DOUBLE, MPI_MIN,
                    p_nlf->ParFESpace()->GetComm());
   }
#endif

   return min_detT_all;
}

#ifdef MFEM_USE_MPI
// Metric values are visualized by creating an L2 finite element functions and
// computing the metric values at the nodes.
void vis_tmop_metric_p(int order, TMOP_QualityMetric &qm,
                       const TargetConstructor &tc, ParMesh &pmesh,
                       char *title, int position)
{
   L2_FECollection fec(order, pmesh.Dimension(), BasisType::GaussLobatto);
   ParFiniteElementSpace fes(&pmesh, &fec, 1);
   ParGridFunction metric(&fes);
   InterpolateTMOP_QualityMetric(qm, tc, pmesh, metric);
   socketstream sock;
   if (pmesh.GetMyRank() == 0)
   {
      sock.open("localhost", 19916);
      sock << "solution\n";
   }
   pmesh.PrintAsOne(sock);
   metric.SaveAsOne(sock);
   if (pmesh.GetMyRank() == 0)
   {
      sock << "window_title '"<< title << "'\n"
           << "window_geometry "
           << position << " " << 0 << " " << 600 << " " << 600 << "\n"
           << "keys jRmclA\n";
   }
}
#endif

// Metric values are visualized by creating an L2 finite element functions and
// computing the metric values at the nodes.
void vis_tmop_metric_s(int order, TMOP_QualityMetric &qm,
                       const TargetConstructor &tc, Mesh &mesh,
                       char *title, int position)
{
   L2_FECollection fec(order, mesh.Dimension(), BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec, 1);
   GridFunction metric(&fes);
   InterpolateTMOP_QualityMetric(qm, tc, mesh, metric);
   osockstream sock(19916, "localhost");
   sock << "solution\n";
   mesh.Print(sock);
   metric.Save(sock);
   sock.send();
   sock << "window_title '"<< title << "'\n"
        << "window_geometry "
        << position << " " << 0 << " " << 600 << " " << 600 << "\n"
        << "keys jRmclA\n";
}

}
