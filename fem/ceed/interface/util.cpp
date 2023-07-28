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

#include "util.hpp"

#include "../../../general/device.hpp"
#include "../../../linalg/dtensor.hpp"
#include "../../gridfunc.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#if !defined(_WIN32) || !defined(_MSC_VER)
typedef struct stat struct_stat;
#else
#define stat(dir, buf) _stat(dir, buf)
#define S_ISDIR(mode) (((mode) & _S_IFMT) == _S_IFDIR)
typedef struct _stat struct_stat;
#endif

namespace mfem
{

bool DeviceCanUseCeed()
{
   return Device::Allows(Backend::CEED_MASK);
}

namespace ceed
{

void RemoveBasisAndRestriction(const mfem::FiniteElementSpace *fes)
{
#ifdef MFEM_USE_CEED
#ifdef MFEM_USE_OPENMP
   #pragma omp parallel
#endif
   {
      auto itb = internal::ceed_basis_map.begin();
      while (itb != internal::ceed_basis_map.end())
      {
         if (std::get<0>(itb->first) == fes)
         {
            CeedBasisDestroy(&itb->second);
            itb = internal::ceed_basis_map.erase(itb);
         }
         else
         {
            itb++;
         }
      }
      auto itr = internal::ceed_restr_map.begin();
      while (itr != internal::ceed_restr_map.end())
      {
         if (std::get<0>(itr->first) == fes)
         {
            CeedElemRestrictionDestroy(&itr->second);
            itr = internal::ceed_restr_map.erase(itr);
         }
         else
         {
            itr++;
         }
      }
   }
#endif
}

#ifdef MFEM_USE_CEED

void InitVector(const mfem::Vector &v, Ceed ceed, CeedVector &cv)
{
   CeedVectorCreate(ceed, v.Size(), &cv);
   CeedScalar *cv_ptr;
   CeedMemType mem;
   CeedGetPreferredMemType(ceed, &mem);
   if (Device::Allows(Backend::DEVICE_MASK) && mem == CEED_MEM_DEVICE)
   {
      cv_ptr = const_cast<CeedScalar *>(v.Read());
   }
   else
   {
      cv_ptr = const_cast<CeedScalar *>(v.HostRead());
      mem = CEED_MEM_HOST;
   }
   CeedVectorSetArray(cv, mem, CEED_USE_POINTER, cv_ptr);
}

// Assumes a tensor-product operator with one active field
int CeedOperatorGetActiveField(CeedOperator oper, CeedOperatorField *field)
{
   int ierr;
   Ceed ceed;
   ierr = CeedOperatorGetCeed(oper, &ceed); PCeedChk(ierr);

   CeedQFunction qf;
   bool isComposite;
   ierr = CeedOperatorIsComposite(oper, &isComposite); PCeedChk(ierr);
   CeedOperator *subops;
   if (isComposite)
   {
#if CEED_VERSION_GE(0, 10, 2)
      ierr = CeedCompositeOperatorGetSubList(oper, &subops); PCeedChk(ierr);
#else
      ierr = CeedOperatorGetSubList(oper, &subops); PCeedChk(ierr);
#endif
      ierr = CeedOperatorGetQFunction(subops[0], &qf); PCeedChk(ierr);
   }
   else
   {
      ierr = CeedOperatorGetQFunction(oper, &qf); PCeedChk(ierr);
   }
   CeedInt numinputfields, numoutputfields;
   ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
   CeedOperatorField *inputfields;
   if (isComposite)
   {
      ierr = CeedOperatorGetFields(subops[0], &numinputfields, &inputfields,
                                   &numoutputfields, NULL); PCeedChk(ierr);
   }
   else
   {
      ierr = CeedOperatorGetFields(oper, &numinputfields, &inputfields,
                                   &numoutputfields, NULL); PCeedChk(ierr);
   }

   CeedVector if_vector;
   bool found = false;
   int found_index = -1;
   for (int i = 0; i < numinputfields; ++i)
   {
      ierr = CeedOperatorFieldGetVector(inputfields[i], &if_vector); PCeedChk(ierr);
      if (if_vector == CEED_VECTOR_ACTIVE)
      {
         if (found)
         {
            return CeedError(ceed, 1, "Multiple active vectors in CeedOperator!");
         }
         found = true;
         found_index = i;
      }
   }
   if (!found)
   {
      return CeedError(ceed, 1, "No active vector in CeedOperator!");
   }
   *field = inputfields[found_index];

   return 0;
}

std::string ceed_path;

const std::string &GetCeedPath()
{
   if (ceed_path.empty())
   {
      const char *install_dir = MFEM_INSTALL_DIR "/include/mfem/fem/ceed";
      const char *source_dir = MFEM_SOURCE_DIR "/fem/ceed";
      struct_stat m_stat;
      if (stat(install_dir, &m_stat) == 0 && S_ISDIR(m_stat.st_mode))
      {
         ceed_path = install_dir;
      }
      else if (stat(source_dir, &m_stat) == 0 && S_ISDIR(m_stat.st_mode))
      {
         ceed_path = source_dir;
      }
      else
      {
         MFEM_ABORT("Cannot find libCEED kernels in MFEM_INSTALL_DIR or "
                    "MFEM_SOURCE_DIR");
      }
      // Could be useful for debugging:
      // out << "Using libCEED dir: " << ceed_path << std::endl;
   }
   return ceed_path;
}

#endif

} // namespace ceed

} // namespace mfem
