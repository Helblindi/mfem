// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project (17-SC-20-SC)
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "volume_fractions.hpp"

using namespace std;

namespace mfem
{

bool IsInCircle(const Vector &x){
    double radius = 0.2;
    Vector center(2);
    center(0) = 0.5;
    center(1) = 0.5;
    return ((pow(x(0)-center(0),2.0)+pow(x(1)-center(1),2.0)) >= pow(radius,2.0)) ? true : false;
}
  
  bool IsInSphere(const Vector &x){
    double radius = 0.3;
    Vector center(3);
    center(0) = 0.5;
    center(1) = 0.5;
    center(2) = 0.5;
    return ( ( pow(x(0)-center(0),2.0) + pow(x(1)-center(1),2.0) + pow(x(2)-center(2),2.0) ) >= pow(radius,2.0) ) ? true : false;
}
  
  bool IsInElement(const Vector &x, const int &type_){
    if (type_ == 1){
      return IsInCircle(x);
    }
    if (type_ == 2){
      return IsInSphere(x);
    } 
    else{
      return false;
    }
      
  }
  
  //  void UpdateAlpha(const AnalyticalSurface &analyticalShape, ParGridFunction &alpha, ParFiniteElementSpace &h1_fes, const int &type)
  void UpdateAlpha(const ShiftedFaceMarker &analyticalShape, ParGridFunction &alpha, ParFiniteElementSpace &h1_fes, const int &type)
  {
    IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
    auto pfes = alpha.ParFESpace();
    const IntegrationRule &ir = IntRulesLo.Get(pfes->GetFE(0)->GetGeomType(), 20);
    const int NE = alpha.ParFESpace()->GetNE(), nqp = ir.GetNPoints();
    for (int e = 0; e < NE; e++)
      {
	ElementTransformation &Tr = *(pfes->GetElementTransformation(e));
	double volume_1 = 0.0, volume = 0.0;
	for (int q = 0; q < nqp; q++)
	  {
	    const IntegrationPoint &ip = ir.IntPoint(q);
	    Tr.SetIntPoint(&ip);
	    Vector x(3);
	    Tr.Transform(ip,x);
	    const bool checkIntegrationPtLocation = IsInElement(x, type);
	    volume   += ip.weight * Tr.Weight();
	    volume_1 += ip.weight * Tr.Weight() * ((checkIntegrationPtLocation) ? 1.0 : 0.0);
	  }  
	alpha(e) = volume_1 / volume;
	if (alpha(e) == 0.0){
	  const FiniteElement *FElem = h1_fes.GetFE(e);
	  const IntegrationRule &ir = FElem->GetNodes();
	  ElementTransformation &T = *(h1_fes.GetElementTransformation(e));
	  for (int j = 0; j < ir.GetNPoints(); j++)
	    {
	      const IntegrationPoint &ip = ir.IntPoint(j);
	      Vector x(3);
	      T.Transform(ip,x);
	      const bool checkIntegrationPtLocation = IsInElement(x, type);
	      if(checkIntegrationPtLocation){
		alpha(e) = 1.0/nqp;
		break;
	      }
	    }
	}
     }
  }
  /* void UpdateAlpha(const AnalyticalSurface &analyticalShape,
		   ParGridFunction &alpha, ParFiniteElementSpace &h1_fes)
  {
    IntegrationRules IntRulesLo(0, Quadrature1D::GaussLobatto);
    auto pfes = alpha.ParFESpace();
    const IntegrationRule &ir = IntRulesLo.Get(pfes->GetFE(0)->GetGeomType(), 20);
    const int NE = alpha.ParFESpace()->GetNE(), nqp = ir.GetNPoints();
    for (int e = 0; e < NE; e++)
      {
	ElementTransformation &Tr = *(pfes->GetElementTransformation(e));
	double volume_1 = 0.0, volume = 0.0;
	for (int q = 0; q < nqp; q++)
	  {
	    const IntegrationPoint &ip = ir.IntPoint(q);
	    Tr.SetIntPoint(&ip);
	    Vector x(3);
	    Tr.Transform(ip,x);
	    const bool checkIntegrationPtLocation = analyticalShape.IsInElement(x);
	    volume   += ip.weight * Tr.Weight();
	    volume_1 += ip.weight * Tr.Weight() * ((checkIntegrationPtLocation) ? 1.0 : 0.0);
	  }
	if (volume_1 > 0.0 ){
	  alpha(e) = 1.0;
	}
	else{
	  alpha(e) = 0.0;
	}
      }
      }*/
}
 // namespace mfem
