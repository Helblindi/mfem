//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m ../data/square-disc.mesh
//               ex1 -m ../data/star.mesh
//               ex1 -m ../data/star-mixed.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh
//               ex1 -m ../data/fichera-mixed.mesh
//               ex1 -m ../data/toroid-wedge.mesh
//               ex1 -m ../data/octahedron.mesh -o 1
//               ex1 -m ../data/periodic-annulus-sector.msh
//               ex1 -m ../data/periodic-torus-sector.msh
//               ex1 -m ../data/square-disc-p2.vtk -o 2
//               ex1 -m ../data/square-disc-p3.mesh -o 3
//               ex1 -m ../data/square-disc-nurbs.mesh -o -1
//               ex1 -m ../data/star-mixed-p2.mesh -o 2
//               ex1 -m ../data/disc-nurbs.mesh -o -1
//               ex1 -m ../data/pipe-nurbs.mesh -o -1
//               ex1 -m ../data/fichera-mixed-p2.mesh -o 2
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
//               ex1 -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               ex1 -pa -d cuda
//               ex1 -fa -d cuda
//               ex1 -pa -d raja-cuda
//             * ex1 -pa -d raja-hip
//               ex1 -pa -d occa-cuda
//               ex1 -pa -d raja-omp
//               ex1 -pa -d occa-omp
//               ex1 -pa -d ceed-cpu
//               ex1 -pa -d ceed-cpu -o 4 -a
//               ex1 -pa -d ceed-cpu -m ../data/square-mixed.mesh
//               ex1 -pa -d ceed-cpu -m ../data/fichera-mixed.mesh
//             * ex1 -pa -d ceed-cuda
//             * ex1 -pa -d ceed-hip
//               ex1 -pa -d ceed-cuda:/gpu/cuda/shared
//               ex1 -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/square-mixed.mesh
//               ex1 -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/fichera-mixed.mesh
//               ex1 -m ../data/beam-hex.mesh -pa -d cuda
//               ex1 -m ../data/beam-tet.mesh -pa -d ceed-cpu
//               ex1 -m ../data/beam-tet.mesh -pa -d ceed-cuda:/gpu/cuda/ref
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   bool fa = false;
   const char *device_config = "hip";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
#ifdef MFEM_USE_CEED
   args.AddOption(&algebraic_ceed, "-a", "--algebraic", "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();


   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   bool delete_fec;
   if (order > 0)
   {
      fec = new DG_FECollection(order, dim);
      delete_fec = true;
   }
   else if (mesh.GetNodes())
   {
      fec = mesh.GetNodes()->OwnFEC();
      delete_fec = false;
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim);
      delete_fec = true;
   }
   FiniteElementSpace fespace(&mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace.GetTrueVSize() << endl;


   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(&fespace);
   x = 0.0;

   Vector rhs(x.Size());
   rhs.Randomize();

   Vector x_ref = rhs;
   Vector x_magma = x_ref;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.

   const int NE = mesh.GetNE();
   const int ndofs = x_ref.Size() / NE;

   MassIntegrator mass_int;
   Vector VecMassMats(ndofs * ndofs * NE);
   mass_int.AssembleEA(fespace, VecMassMats, false);

   //Batch solver...
   DenseTensor LUMassMats(ndofs, ndofs, NE);
   DenseTensor MassMats(ndofs, ndofs, NE);
   Array<int> P(rhs.Size());

   std::cout<<"sizes "<<LUMassMats.TotalSize()
            <<" "<<VecMassMats.Size()<<std::endl;

   for(int i=0; i<ndofs * ndofs * NE; ++i)
   {
     LUMassMats.HostWrite()[i] = VecMassMats.HostRead()[i];
     MassMats.HostWrite()[i] = VecMassMats.HostRead()[i];
   }

   //Compute reference solution
   mfem::BatchLUFactor(LUMassMats, P);

   {
     std::cout<<"\n reference pivot array ... "<<std::endl;
     int elem = 1;
     for(int i=0; i<ndofs; ++i) {
       int indx = i + ndofs * elem;
       //std::cout<<x_magma.HostReadWrite()[indx]<<
       //" "<<x_ref.HostReadWrite()[indx]<<std::endl;
       std::cout<<P.HostReadWrite()[indx]<<std::endl;
     }
   }


   mfem::BatchLUSolve(LUMassMats, P, x_ref);

   //Compute magma version with variable batch solver
   magma_int_t magma_device = 0;
   magma_queue_t magma_queue;

   magma_setdevice(magma_device);
   magma_queue_create(magma_device, &magma_queue);


   std::cout<<"Number of matrices NE = "<<NE<<std::endl;

   //Number of rows and columns of each matrix
   Array<magma_int_t> num_of_rows(NE);
   Array<magma_int_t> num_of_cols(NE);

   Array<double *> magma_LUMassMats(NE);


   for(int i=0; i<NE; ++i)
   {
     num_of_rows[i] = ndofs;
     num_of_cols[i] = ndofs;
     magma_LUMassMats.HostWrite()[i] = &MassMats.ReadWrite()[i*ndofs*ndofs];
   }

   Array<magma_int_t> ldda = num_of_cols;

   Array<magma_int_t *> dipiv_array(NE);
   Array<magma_int_t> info_array(NE);

   P.HostReadWrite();
   P = 0.0;
   for(int i=0; i<NE; ++i)
   {
     dipiv_array.HostWrite()[i] = &P.ReadWrite()[i*ndofs];
   }

   magma_dgetrf_vbatched(num_of_rows.ReadWrite(), num_of_cols.ReadWrite(),
                         magma_LUMassMats.ReadWrite(), ldda.ReadWrite(),
                         dipiv_array.ReadWrite(), info_array.ReadWrite(),
                         NE, magma_queue);


   {
     std::cout<<"ndofs = "<<ndofs<<std::endl;
     std::cout<<"magma mat "<<std::endl;
     int elem = 8;

     for(int r=0; r<ndofs; ++r) {
       for(int c=0; c<ndofs; ++c) {
         int idx = c + ndofs * r  + ndofs*ndofs*elem;
         std::cout<<MassMats.HostReadWrite()[idx]<<" ";
       }
       std::cout<<" "<<std::endl;
     }

     std::cout<<" "<<std::endl;

     std::cout<<"reference mat "<<std::endl;
     for(int r=0; r<ndofs; ++r) {
       for(int c=0; c<ndofs; ++c) {
         int idx = c + ndofs * r  + ndofs*ndofs*elem;
         std::cout<<LUMassMats.HostReadWrite()[idx]<<" ";
       }
       std::cout<<" "<<std::endl;
     }

   }


   {
     std::cout<<"\n magma pivot array ... "<<std::endl;
     int elem = 1;
     for(int i=0; i<ndofs; ++i) {
       int indx = i + ndofs * elem;
       //std::cout<<x_magma.HostReadWrite()[indx]<<
       //" "<<x_ref.HostReadWrite()[indx]<<std::endl;
       std::cout<<P.HostReadWrite()[indx]<<std::endl;
     }
   }

   //rhs only has 1 column
   Array<int> rhs_num_of_cols(NE);
   rhs_num_of_cols = 1;

   //may need to perfom pivoting myself...?

   //x_magma_ptrs to x_magma
   Array<double *> x_magma_ptrs(NE);
   for(int i=0; i<NE; ++i) {
     x_magma_ptrs.HostWrite()[i] = &x_magma.ReadWrite()[i*ndofs];
   }

   Array<magma_int_t> trSolve_ldda(NE + 1);
   Array<magma_int_t> trSolve_lddb(NE + 1);
   for(int i=0; i<NE+1; ++i) {
     trSolve_ldda[i] = ndofs; //??
     trSolve_lddb[i] = ndofs; //??
   }


   //L(Ux) = b //lower solve
   magmablas_dtrsm_vbatched(MagmaLeft, MagmaLower, MagmaNoTrans,
                            MagmaNonUnit,
                            num_of_rows.ReadWrite(),
                            rhs_num_of_cols.ReadWrite(),
                            1.0,
                            magma_LUMassMats.ReadWrite(), trSolve_ldda.ReadWrite(),
                            x_magma_ptrs.ReadWrite(), trSolve_lddb.ReadWrite(),
                            NE, magma_queue);
#if 0
   //Ux = b //upper
   magmablas_dtrsm_vbatched(MagmaLeft, MagmaUpper, MagmaNoTrans,
                            MagmaUnit,
                            num_of_rows.ReadWrite(),
                            rhs_num_of_cols.ReadWrite(),
                            1.0,
                            magma_LUMassMats.ReadWrite(), trSolve_ldda.ReadWrite(),
                            x_magma_ptrs.ReadWrite(), trSolve_lddb.ReadWrite(),
                            NE, magma_queue);
#endif


   //mfem::BatchLUSolve(MassMats, P, x_magma);

   x_magma -= x_ref;

   double error = x_magma.Norml2();

   std::cout<<"error = "<<error<<std::endl;


   // 15. Free the used memory.
   if (delete_fec)
   {
      delete fec;
   }

   return 0;
}
