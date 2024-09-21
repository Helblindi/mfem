# Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Sets the following variables:
#   - HIOP_FOUND
#   - HIOP_INCLUDE_DIRS
#   - HIOP_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(HIOP HIOP HIOP_DIR
  "include" "hiopInterface.hpp"
  "lib" "hiop"
  "Paths to headers required by HIOP."
  "Libraries required by HIOP.")

# Is there a way to get the namespace libraries HiOp::hiop_tpl from the mfem_find_package command?
# Ideally I would like to not have to call find_package(hiop) twice essentially
find_package(hiop REQUIRED NAMES hiop HiOp HINTS "${HIOP_DIR}")

# this test fails with parallel MFEM since mpi.h is not available (cxx compiler is used for some reason)
#  CHECK_BUILD HIOP_VERSION_OK TRUE
#"
##include <hiopInterface.hpp>
#using namespace hiop;
#int main(int argc, char *argv[])
#{
#   MPI_Init(&argc, &argv);
#   MPI_Comm comm = MPI_COMM_WORLD;
#
#   return 0;
#}
#")
