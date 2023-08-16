#include "schwarz.hpp"

// Some utility functions

bool is_a_patch(int iv, Array<int> patch_ids)
{
   return (patch_ids.FindSorted(iv) != -1);
}

bool owned(int tdof, int * offs)
{
   return  (offs[0] <= tdof && tdof < offs[1]);
}

void GetColumnValues(const int tdof_i, const Array<int> & tdof_j,
                     SparseMatrix & diag,
                     SparseMatrix & offd, const int * cmap, const int * row_start,  Array<int> &cols,
                     Array<double> &vals)
{
   int row = tdof_i - row_start[0];
   int row_size = diag.RowSize(row);

   int *col = diag.GetRowColumns(row);
   double *cval = diag.GetRowEntries(row);
   for (int j = 0; j < row_size; j++)
   {
      int icol = col[j]+ row_start[0];
      int jj = tdof_j.FindSorted(icol);
      if (jj != -1)
      {
         double dval = cval[j];
         cols.Append(jj);
         vals.Append(dval);
      }
   }
   int crow_size = offd.RowSize(row);
   int *ccol = offd.GetRowColumns(row);
   double *ccval = offd.GetRowEntries(row);
   for (int j = 0; j < crow_size; j++)
   {
      int icol = cmap[ccol[j]];
      int jj = tdof_j.FindSorted(icol);
      if (jj != -1)
      {
         double dval = ccval[j];
         cols.Append(jj);
         vals.Append(dval);
      }
   }
}

int GetNumColumns(const int tdof_i, const Array<int> & tdof_j,
                  SparseMatrix & diag,
                  SparseMatrix & offd, const int * cmap, const int * row_start)
{
   int row = tdof_i - row_start[0];
   int row_size = diag.RowSize(row);

   int *col = diag.GetRowColumns(row);
   int k = -1;
   for (int j = 0; j < row_size; j++)
   {
      int icol = col[j]+ row_start[0];
      int jj = tdof_j.FindSorted(icol);
      if (jj != -1)
      {
         k++;
      }
   }
   int crow_size = offd.RowSize(row);
   int *ccol = offd.GetRowColumns(row);
   for (int j = 0; j < crow_size; j++)
   {
      int icol = cmap[ccol[j]];
      int jj = tdof_j.FindSorted(icol);
      if (jj != -1)
      {
         k++;
      }
   }
   return k;
}

void GetOffdColumnValues(const Array<int> & tdof_i, const Array<int> & tdof_j,
                         SparseMatrix & offd, const int * cmap,
                         const int * row_start, SparseMatrix * PatchMat)
{
   int ndof = tdof_i.Size();
   for (int i = 0; i<ndof; i++)
   {
      int row = tdof_i[i] - row_start[0];
      int row_size = offd.RowSize(row);
      int *ccol = offd.GetRowColumns(row);
      double *ccval = offd.GetRowEntries(row);
      for (int j = 0; j < row_size; j++)
      {
         int icol = cmap[ccol[j]];
         int jj = tdof_j.FindSorted(icol);
         if (jj != -1)
         {
            double dval = ccval[j];
            PatchMat->Set(i,jj,dval);
         }
      }
   }
}

SparseMatrix * GetLocalRestriction(const Array<int> & tdof_i,
                                   const int * row_start,
                                   const int num_rows, const int num_cols)
{
   SparseMatrix * R = new SparseMatrix(num_cols,num_rows);
   for (int i=0; i<num_cols; i++)
   {
      int ii = tdof_i[i] - row_start[0];
      R->Set(i,ii,1.0);
   }
   R->Finalize();
   return R;
}

void GetLocal2GlobalMap(const Array<int> & tdof_i, const int * row_start,
                        const int num_rows, const int num_cols, Array<int> & l2gmap)
{
   l2gmap.SetSize(num_cols);
   for (int i=0; i<num_cols; i++)
   {
      int ii = tdof_i[i] - row_start[0];
      l2gmap[i] = ii;
   }
}

void GetArrayIntersection(const Array<int> & A, const Array<int> & B,
                          Array<int>  & C)
{
   int i = 0, j = 0;
   while (i != A.Size() && j != B.Size())
   {
      if (A[i] == B[j])
      {
         C.Append(A[i]);
         i++;
         j++;
      }
      else if (A[i] > B[j])
      {
         j++;
      }
      else
      {
         i++;
      }
   }
}


VertexPatchInfo::VertexPatchInfo(ParMesh *pmesh_, int ref_levels_)
   : pmesh(pmesh_), ref_levels(ref_levels_)
{
   int dim = pmesh->Dimension();
   // 1. Define an auxiliary parallel H1 finite element space on the parallel mesh.
   FiniteElementCollection * aux_fec = new H1_FECollection(1, dim);
   ParFiniteElementSpace * aux_fespace = new ParFiniteElementSpace(pmesh, aux_fec);
   int mycdofoffset =
      aux_fespace->GetMyDofOffset(); // dof offset for the coarse mesh

   // 2. Store the cDofTrueDof Matrix. Required after the refinements
   HypreParMatrix *cDofTrueDof = new HypreParMatrix(
      *aux_fespace->Dof_TrueDof_Matrix());

   // 3. Perform the refinements (if any) and Get the final Prolongation operator
   HypreParMatrix *Pr = nullptr;
   for (int i = 0; i < ref_levels; i++)
   {
      const ParFiniteElementSpace cfespace(*aux_fespace);
      pmesh->UniformRefinement();
      // Update fespace
      aux_fespace->Update();
      OperatorHandle Tr(Operator::Hypre_ParCSR);
      aux_fespace->GetTrueTransferOperator(cfespace, Tr);
      Tr.SetOperatorOwner(false);
      HypreParMatrix *P;
      Tr.Get(P);
      if (!Pr)
      {
         Pr = P;
      }
      else
      {
         Pr = ParMult(P, Pr);
      }
   }
   if (Pr) { Pr->Threshold(0.0); }

   // 4. Get the DofTrueDof map on this mesh and convert the prolongation matrix
   // to correspond to global dof numbering (from true dofs to dofs)
   HypreParMatrix *DofTrueDof = aux_fespace->Dof_TrueDof_Matrix();
   HypreParMatrix *A = nullptr;
   if (Pr)
   {
      A = ParMult(DofTrueDof, Pr);
   }
   else
   {
      // If there is no refinement then the prolongation is the identity
      A = DofTrueDof;
   }
   HypreParMatrix * cDofTrueDofT = cDofTrueDof->Transpose();
   HypreParMatrix *B = ParMult(A, cDofTrueDofT);
   delete cDofTrueDofT;
   // 5. Now we compute the vertices that are owned by the process
   SparseMatrix cdiag, coffd;
   cDofTrueDof->GetDiag(cdiag);
   Array<int> cown_vertices;
   int cnv = 0;
   for (int k = 0; k < cdiag.Height(); k++)
   {
      int nz = cdiag.RowSize(k);
      int i = mycdofoffset + k;
      if (nz != 0)
      {
         cnv++;
         cown_vertices.SetSize(cnv);
         cown_vertices[cnv - 1] = i;
      }
   }
   // 6. Compute total number of patches
   MPI_Comm comm = pmesh->GetComm();
   mynrpatch = cown_vertices.Size();
   // Compute total number of patches.

   MPI_Allreduce(&mynrpatch, &nrpatch, 1, MPI_INT, MPI_SUM, comm);

   patch_global_dofs_ids.SetSize(nrpatch);
   // Create a list of patches identifiers to all procs
   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   int count[num_procs];

   MPI_Allgather(&mynrpatch, 1, MPI_INT, &count[0], 1, MPI_INT, comm);
   int displs[num_procs];
   displs[0] = 0;
   for (int i = 1; i < num_procs; i++)
   {
      displs[i] = displs[i - 1] + count[i - 1];
   }

   int * cownvert_ptr = nullptr;
   int * dof_rank_id_ptr = nullptr;
   Array<int> dof_rank_id;
   if (cown_vertices.Size() >0)
   {
      cownvert_ptr = &cown_vertices[0];
      dof_rank_id.SetSize(cown_vertices.Size());
      dof_rank_id = myid;
      dof_rank_id_ptr = &dof_rank_id[0];
   }
   // send also the rank number for each global dof
   host_rank.SetSize(nrpatch);
   MPI_Allgatherv(cownvert_ptr, mynrpatch, MPI_INT, &patch_global_dofs_ids[0],
                  count, displs, MPI_INT, comm);
   MPI_Allgatherv(dof_rank_id_ptr, mynrpatch, MPI_INT, &host_rank[0], count,
                  displs, MPI_INT, comm);

   int size = patch_global_dofs_ids[nrpatch - 1] + 1;
   patch_natural_order_idx.SetSize(size);
   // TODO: isn't this size too big? Can a map replace this array? In serial it is fine, but it seems to depend on the global number of patches, which may be fine for moderate problem sizes but could get large and hinder scalability.
   //cout << "DYLAN DBG: patch_natural_order_idx size " << size << ", nrpatch " << nrpatch << ", aux dofs "
   //<< aux_fespace->GlobalVSize() << endl;

   // initialize with -1
   patch_natural_order_idx = -1;
   for (int i = 0; i < nrpatch; i++)
   {
      int k = patch_global_dofs_ids[i];
      patch_natural_order_idx[k] = i;
   }

   int nvert = pmesh->GetNV();
   // first find all the contributions of the vertices
   vert_contr.resize(nvert);
   SparseMatrix H1pr_diag;
   B->GetDiag(H1pr_diag);
   for (int i = 0; i < nvert; i++)
   {
      int row = i;
      int row_size = H1pr_diag.RowSize(row);
      int *col = H1pr_diag.GetRowColumns(row);
      for (int j = 0; j < row_size; j++)
      {
         int jv = col[j] + mycdofoffset;
         if (is_a_patch(jv, patch_global_dofs_ids))
         {
            vert_contr[i].Append(jv);
         }
      }
   }

   SparseMatrix H1pr_offd;
   int *cmap;
   B->GetOffd(H1pr_offd, cmap);
   for (int i = 0; i < nvert; i++)
   {
      int row = i;
      int row_size = H1pr_offd.RowSize(row);
      int *col = H1pr_offd.GetRowColumns(row);
      for (int j = 0; j < row_size; j++)
      {
         int jv = cmap[col[j]];
         if (is_a_patch(jv, patch_global_dofs_ids))
         {
            vert_contr[i].Append(jv);
         }
      }
   }

   Array<int> edge_vertices;
   int nedge = pmesh->GetNEdges();
   edge_contr.resize(nedge);
   for (int ie = 0; ie < nedge; ie++)
   {
      pmesh->GetEdgeVertices(ie, edge_vertices);
      int nv = edge_vertices.Size(); // always 2 but ok
      // The edge will contribute to the same patches as its vertices
      for (int iv = 0; iv < nv; iv++)
      {
         int ivert = edge_vertices[iv];
         edge_contr[ie].Append(vert_contr[ivert]);
      }
      edge_contr[ie].Sort();
      edge_contr[ie].Unique();
   }
   // -----------------------------------------------------------------------
   // done with edges. Now the faces
   // -----------------------------------------------------------------------
   Array<int> face_vertices;
   int nface = pmesh->GetNFaces();
   face_contr.resize(nface);
   for (int ifc = 0; ifc < nface; ifc++)
   {
      pmesh->GetFaceVertices(ifc, face_vertices);
      int nv = face_vertices.Size();
      // The face will contribute to the same patches as its vertices
      for (int iv = 0; iv < nv; iv++)
      {
         int ivert = face_vertices[iv];
         face_contr[ifc].Append(vert_contr[ivert]);
      }
      face_contr[ifc].Sort();
      face_contr[ifc].Unique();
   }
   // -----------------------------------------------------------------------
   // Finally the elements
   // -----------------------------------------------------------------------
   Array<int> elem_vertices;
   int nelem = pmesh->GetNE();
   elem_contr.resize(nelem);
   for (int iel = 0; iel < nelem; iel++)
   {
      pmesh->GetElementVertices(iel, elem_vertices);
      int nv = elem_vertices.Size();
      // The element will contribute to the same patches as its vertices
      for (int iv = 0; iv < nv; iv++)
      {
         int ivert = elem_vertices[iv];
         elem_contr[iel].Append(vert_contr[ivert]);
      }
      elem_contr[iel].Sort();
      elem_contr[iel].Unique();
   }
   if (Pr) { delete A; }
   delete B;
   if (Pr) { delete Pr; }
   delete cDofTrueDof;
   delete aux_fespace;
   delete aux_fec;
}

void ParaViewPrintAttributes(const string &fname,
                             Mesh &mesh,
                             int entity_dim,
                             const Array<int> *el_number,
                             const Array<int> *vert_number)
{
   ofstream out(fname + ".vtu");

   out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\"";
   out << " byte_order=\"" << VTKByteOrder() << "\">\n";
   out << "<UnstructuredGrid>\n";

   const string fmt_str = "ascii";

   int dim = mesh.Dimension();
   int ne = 0;
   if (entity_dim == 1)
   {
      if (dim == 1)
      {
         ne = mesh.GetNE();
      }
      else
      {
         ne = mesh.GetNEdges();
      }
   }
   else if (entity_dim == 2)
   {
      if (dim == 2)
      {
         ne = mesh.GetNE();
      }
      else
      {
         ne = mesh.GetNFaces();
      }
   }
   else if (entity_dim == 3)
   {
      ne = mesh.GetNE();
   }
   int np = mesh.GetNV();

   auto get_geom = [mesh,entity_dim,dim](int i)
   {
      if (entity_dim == 1)
      {
         return Geometry::SEGMENT;
      }
      else if (entity_dim == 2 && dim > 2)
      {
         return mesh.GetFaceGeometry(i);
      }
      else
      {
         return mesh.GetElementGeometry(i);
      }
   };

   auto get_verts = [mesh,entity_dim,dim](int i, Array<int> &v)
   {
      if (entity_dim == dim)
      {
         mesh.GetElementVertices(i, v);
      }
      else if (entity_dim == 1)
      {
         mesh.GetEdgeVertices(i, v);
      }
      else if (entity_dim == 2)
      {
         mesh.GetFaceVertices(i, v);
      }
   };

   out << "<Piece NumberOfPoints=\"" << np << "\" NumberOfCells=\""
       << ne << "\">\n";

   // print out the points
   out << "<Points>\n";
   out << "<DataArray type=\"" << "Float64"
       << "\" NumberOfComponents=\"3\" format=\"" << fmt_str << "\">\n";
   for (int i = 0; i < np; i++)
   {
      const double *v = mesh.GetVertex(i);
      for (int d = 0; d < 3; ++ d)
      {
         if (d < mesh.SpaceDimension())
         {
            out << v[d] << " ";
         }
         else
         {
            out << "0.0 ";
         }
      }
      out << '\n';
   }
   out << "</DataArray>" << endl;
   out << "</Points>" << endl;

   out << "<Cells>" << endl;
   out << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\""
       << fmt_str << "\">" << endl;
   for (int i = 0; i < ne; i++)
   {
      Array<int> v;
      Geometry::Type geom = get_geom(i);
      get_verts(i, v);
      const int *p = VTKGeometry::VertexPermutation[geom];
      for (int j = 0; j < v.Size(); ++j)
      {
         out << v[p ? p[j] : j] << " ";
      }
      out << '\n';
   }
   out << "</DataArray>" << endl;

   out << "<DataArray type=\"Int32\" Name=\"offsets\" format=\""
       << fmt_str << "\">" << endl;
   // offsets
   int coff = 0;
   for (int i = 0; i < ne; ++i)
   {
      Array<int> v;
      get_verts(i, v);
      coff += v.Size();
      out << coff << '\n';
   }
   out << "</DataArray>" << endl;
   out << "<DataArray type=\"UInt8\" Name=\"types\" format=\""
       << fmt_str << "\">" << endl;
   // cell types
   for (int i = 0; i < ne; i++)
   {
      Geometry::Type geom = get_geom(i);
      out << VTKGeometry::Map[geom] << '\n';
   }
   out << "</DataArray>" << endl;
   out << "</Cells>" << endl;

   out << "<CellData Scalars=\"attribute\">" << endl;

   if (el_number)
   {
      string array_name;
      if (entity_dim == dim)
      {
         array_name = "element number";
      }
      else if (entity_dim == 2)
      {
         array_name = "face number";
      }
      else if (entity_dim == 1)
      {
         array_name = "edge number";
      }
      out << "<DataArray type=\"Int32\" Name=\""
          << array_name << "\" format=\""
          << fmt_str << "\">" << endl;
      for (int i = 0; i < ne; i++)
      {
         out << (*el_number)[i] << '\n';
      }
      out << "</DataArray>" << endl;
   }
   out << "</CellData>" << endl;

   if (vert_number)
   {
      out << "<PointData>" << endl;
      out << "<DataArray type=\"Int32\" Name=\"vertex number\" format=\""
          << fmt_str << "\">" << endl;
      for (int i = 0; i < np; i++)
      {
         out << (*vert_number)[i] << '\n';
      }
      out << "</DataArray>" << endl;
      out << "</PointData>" << endl;
   }

   out << "</Piece>\n"; // need to close the piece open in the PrintVTU method
   out << "</UnstructuredGrid>\n";
   out << "</VTKFile>" << endl;
}

SparseBooleanMatrix::SparseBooleanMatrix(SparseBooleanMatrix const& M)
   : n(M.GetSize()), a(M.GetSize())
{
   for (int row=0; row<n; ++row)
   {
      for (auto col : M.GetRow(row))
      {
         a[row].push_back(col);
      }
   }
}

void SparseBooleanMatrix::SetEntry(int row, int col)
{
   bool found = false;
   for (auto i : a[row])
   {
      if (i == col)
      {
         found = true;
      }
   }

   if (!found)
   {
      a[row].push_back(col);
   }
}

SparseBooleanMatrix* SparseBooleanMatrix::Transpose() const
{
   SparseBooleanMatrix *t = new SparseBooleanMatrix(n);
   for (int r=0; r<n; ++r)
   {
      for (auto c : a[r])
      {
         t->SetEntry(c, r);
      }
   }

   return t;
}

SparseBooleanMatrix* SparseBooleanMatrix::Mult(SparseBooleanMatrix const& M)
const
{
   MFEM_VERIFY(M.GetSize() == n, "");
   SparseBooleanMatrix *p = new SparseBooleanMatrix(n);

   // TODO: this could be optimized, but for now just using it for
   // matrices with 2 nonzeros per row.

   SparseBooleanMatrix *Mt = M.Transpose();

   for (int r=0; r<n; ++r)
   {
      std::set<int> row;
      for (auto c : a[r])
      {
         row.insert(c);
      }

      // TODO: this is O(n^2), which can be improved.
      for (int c=0; c<n; ++c)
      {
         for (auto i : Mt->GetRow(c))  // entry (c,i) is in Mt, so (i,c) is in M
         {
            std::set<int>::iterator it = row.find(i);
            if (it != row.end())
            {
               p->SetEntry(r, c);
            }
         }
      }
   }

   delete Mt;

   return p;
}

SparseMatrix* GetAnisotropicGraph_with_distance(ParMesh *pmesh,
                                                VectorCoefficient &BCoef)
{
   const int distance = 1;
   double threshold = 1.5;

   // TODO: generalize to 3D
   const int dim = 2;
   Vector tvec(dim);

   int bdim = BCoef.GetVDim();
   Vector bvec(bdim);
   bvec = 0.0;

   const int nv = pmesh->GetNV();

   Array<bool> node_flag(nv);
   node_flag = false;

   // TODO: generalize for parallel case
   SparseBooleanMatrix A(nv);
   SparseMatrix *G = new SparseMatrix(nv);

   // Set an entry of A for each edge in pmesh, connecting the two vertices
   std::map<std::pair<int,int>, int> edge_by_verts;
   int nedges = pmesh->GetNEdges();

   for (int i=0; i<nedges; ++i)
   {
      Array<int> vert;
      pmesh->GetEdgeVertices(i, vert);
      MFEM_VERIFY(vert.Size() == 2, "");
      A.SetEntry(vert[0], vert[1]);
      A.SetEntry(vert[1], vert[0]);

      edge_by_verts[std::make_pair(std::min(vert[0], vert[1]),
                                                                    std::max(vert[0], vert[1]))] = i;
   }

   if (distance == 1)  // Not using algebraic distance
   {
      // Connect vertices of quadrilateral elements not sharing an edge
      for (int i=0; i<pmesh->GetNE(); ++i)
      {
         Array<int> vert;
         pmesh->GetElementVertices(i, vert);
         MFEM_VERIFY(vert.Size() == 4, "");
         for (int j=0; j<2; ++j) // Loop over diagonals
         {
            A.SetEntry(vert[j], vert[j+2]);
            A.SetEntry(vert[j+2], vert[j]);

            edge_by_verts[std::make_pair(std::min(vert[j], vert[j+2]),
                                                                          std::max(vert[j], vert[j+2]))]
               = nedges + 2 * i + j;
         }
      }
   }

   SparseBooleanMatrix *dis_A = new SparseBooleanMatrix(A);
   for (int i=0; i<distance-1; ++i)
   {
      SparseBooleanMatrix *prod = dis_A->Mult(A);
      delete dis_A;
      dis_A = prod;
   }

   // Loop over all nonzero entries in dis_A
   for (int v0=0; v0<dis_A->GetSize(); ++v0)
   {
      for (auto v1 : dis_A->GetRow(v0))
      {
         // Entry (v0,v1) is in dis_A

         double bnorm = 0.0;
         double tnorm = 0.0;
         double ip = 0.0;

         {
            ElementTransformation* T = NULL;
            IntegrationPoint eip;

            int edge_no = edge_by_verts[std::make_pair(std::min(v0,v1),
                                                       std::max(v0,v1))];
            if (edge_no < nedges)
            {
               T = pmesh->GetEdgeTransformation(edge_no);
               eip.x = 0.5;
               T->SetIntPoint(&eip);

               const DenseMatrix &J = T->Jacobian();
               J.GetColumn(0, tvec);
            }
            else
            {
               int elem_no = (edge_no - nedges) / 2;
               int diag_no = (edge_no - nedges) % 2;

               T = pmesh->GetElementTransformation(elem_no);
               eip.x = 0.5; eip.y = 0.5;
               T->SetIntPoint(&eip);

               const DenseMatrix &J = T->Jacobian();

               // col1 + col0 or col1 - col0
               tvec[0] = J(0,1) + (1.0 - 2.0 * diag_no) * J(0,0);
               tvec[1] = J(1,1) + (1.0 - 2.0 * diag_no) * J(1,0);
            }
            BCoef.Eval(bvec, *T, eip);
         }

         if (false)
         {
            // Rotate bvec
            const double tmp = bvec[0];
            bvec[0] = -bvec[1];
            bvec[1] = tmp;
         }

         for (int i=0; i<dim; ++i)
         {
            bnorm += bvec[i] * bvec[i];
            tnorm += tvec[i] * tvec[i];
            ip += bvec[i] * tvec[i];
         }

         bnorm = sqrt(bnorm);
         tnorm = sqrt(tnorm);

         const double cos_theta = ip / (bnorm * tnorm);
         const double edge_weight = 1.0 / (fabs(cos_theta) + 1.0e-6);
         if (edge_weight > threshold)
         {
            G->Set(v0, v1, edge_weight);
            node_flag[v0] = true;
            node_flag[v1] = true;
         }
      }
   }

   delete dis_A;

   // Find any nodes not in G
   for (int i=0; i<nv; ++i)
   {
      if (!node_flag[i])
      {
         G->Set(i,i,-1.0);   // Weight -1 indicates it is not connected to another node
      }
   }

   G->Finalize();
   return G;
}

class AnisotropyPseudoIntegrator : public BilinearFormIntegrator
{
private:
   VectorCoefficient &BCoef_;

   mutable Vector bVec_;
   mutable Vector tVec_;

   double edgeWeight()
   {
      const int dim = tVec_.Size();
      double tnorm = 0.0, bnorm = 0.0, ip = 0.0;

      for (int i=0; i<dim; ++i)
      {
         bnorm += bVec_[i] * bVec_[i];
         tnorm += tVec_[i] * tVec_[i];
         ip += bVec_[i] * tVec_[i];
      }

      bnorm = sqrt(bnorm);
      tnorm = sqrt(tnorm);

      const double cos_theta = ip / (bnorm * tnorm);
      const double edge_weight = 1.0 / (fabs(cos_theta) + 1.0e-6);
      return edge_weight;
   }

   void segmentWeights(double threshold,
                       ElementTransformation &Trans,
                       DenseMatrix &elmat)
   {
      IntegrationPoint ip;
      {
         // edge 0 - 1
         ip.x = 0.5; ip.y = 0.0;
         Trans.SetIntPoint(&ip);
         const DenseMatrix &J = Trans.Jacobian();
         J.GetColumn(0, tVec_);
         BCoef_.Eval(bVec_, Trans, ip);

         const double edge_weight = 0.5 * edgeWeight();
         if (edge_weight > 0.5 * threshold)
         {
            elmat(0,1) = edge_weight;
            elmat(1,0) = edge_weight;
         }
      }
   }

   void triangleWeights(double threshold,
                        ElementTransformation &Trans,
                        DenseMatrix &elmat)
   {
      IntegrationPoint ip;
      {
         // edge 0 - 1
         ip.x = 0.5; ip.y = 0.0;
         Trans.SetIntPoint(&ip);
         const DenseMatrix &J = Trans.Jacobian();
         J.GetColumn(0, tVec_);
         BCoef_.Eval(bVec_, Trans, ip);

         const double edge_weight = 0.5 * edgeWeight();
         if (edge_weight > 0.5 * threshold)
         {
            elmat(0,1) = edge_weight;
            elmat(1,0) = edge_weight;
         }
      }
      {
         // edge 0 - 2
         ip.x = 0.0; ip.y = 0.5;
         Trans.SetIntPoint(&ip);
         const DenseMatrix &J = Trans.Jacobian();
         J.GetColumn(1, tVec_);
         BCoef_.Eval(bVec_, Trans, ip);

         const double edge_weight = 0.5 * edgeWeight();
         if (edge_weight > 0.5 * threshold)
         {
            elmat(0,2) = edge_weight;
            elmat(2,0) = edge_weight;
         }
      }
      {
         // edge 1 - 2
         ip.x = 0.5; ip.y = 0.5;
         Trans.SetIntPoint(&ip);
         const DenseMatrix &J = Trans.Jacobian();
         tVec_[0] = J(0,1) - J(0,0);
         tVec_[1] = J(1,1) - J(1,0);
         BCoef_.Eval(bVec_, Trans, ip);

         const double edge_weight = 0.5 * edgeWeight();
         if (edge_weight > 0.5 * threshold)
         {
            elmat(1,2) = edge_weight;
            elmat(2,1) = edge_weight;
         }
      }
   }

   void squareWeights(double threshold,
                      ElementTransformation &Trans,
                      DenseMatrix &elmat)
   {
      IntegrationPoint ip;
      {
         // edge 0 - 1
         ip.x = 0.5; ip.y = 0.0;
         Trans.SetIntPoint(&ip);
         const DenseMatrix &J = Trans.Jacobian();
         J.GetColumn(0, tVec_);
         BCoef_.Eval(bVec_, Trans, ip);

         const double edge_weight = 0.5 * edgeWeight();
         if (edge_weight > 0.5 * threshold)
         {
            elmat(0,1) = edge_weight;
            elmat(1,0) = edge_weight;
         }
      }
      {
         // edge 2 - 3
         ip.x = 0.5; ip.y = 1.0;
         Trans.SetIntPoint(&ip);
         const DenseMatrix &J = Trans.Jacobian();
         J.GetColumn(0, tVec_);
         BCoef_.Eval(bVec_, Trans, ip);

         const double edge_weight = 0.5 * edgeWeight();
         if (edge_weight > 0.5 * threshold)
         {
            elmat(2,3) = edge_weight;
            elmat(3,2) = edge_weight;
         }
      }
      {
         // edge 0 - 3
         ip.x = 0.0; ip.y = 0.5;
         Trans.SetIntPoint(&ip);
         const DenseMatrix &J = Trans.Jacobian();
         J.GetColumn(1, tVec_);
         BCoef_.Eval(bVec_, Trans, ip);

         const double edge_weight = 0.5 * edgeWeight();
         if (edge_weight > 0.5 * threshold)
         {
            elmat(0,3) = edge_weight;
            elmat(3,0) = edge_weight;
         }
      }
      {
         // edge 1 - 2
         ip.x = 1.0; ip.y = 0.5;
         Trans.SetIntPoint(&ip);
         const DenseMatrix &J = Trans.Jacobian();
         J.GetColumn(1, tVec_);
         BCoef_.Eval(bVec_, Trans, ip);

         const double edge_weight = 0.5 * edgeWeight();
         if (edge_weight > 0.5 * threshold)
         {
            elmat(1,2) = edge_weight;
            elmat(2,1) = edge_weight;
         }
      }
      {
         // edge 0 - 2
         ip.x = 0.5; ip.y = 0.5;
         Trans.SetIntPoint(&ip);
         const DenseMatrix &J = Trans.Jacobian();
         tVec_[0] = J(0,1) + J(0,0);
         tVec_[1] = J(1,1) + J(1,0);
         BCoef_.Eval(bVec_, Trans, ip);

         const double edge_weight = edgeWeight();
         if (edge_weight > threshold)
         {
            elmat(0,2) = edge_weight;
            elmat(2,0) = edge_weight;
         }
      }
      {
         // edge 1 - 3
         ip.x = 0.5; ip.y = 0.5;
         Trans.SetIntPoint(&ip);
         const DenseMatrix &J = Trans.Jacobian();
         tVec_[0] = J(0,1) - J(0,0);
         tVec_[1] = J(1,1) - J(1,0);
         BCoef_.Eval(bVec_, Trans, ip);

         const double edge_weight = edgeWeight();
         if (edge_weight > threshold)
         {
            elmat(1,3) = edge_weight;
            elmat(3,1) = edge_weight;
         }
      }
   }

public:
   AnisotropyPseudoIntegrator(VectorCoefficient &BCoef)
      : BCoef_(BCoef), bVec_(BCoef.GetVDim())
   {}

   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat)
   {
      const double threshold = 1.5;

      int nd = el.GetDof();
      int dim = el.GetDim();
      Geometry::Type type = el.GetGeomType();

      tVec_.SetSize(dim);
      elmat.SetSize(nd); elmat = 0.0;

      switch (type)
      {
         case Geometry::SEGMENT:
            segmentWeights(threshold, Trans, elmat);
            break;
         case Geometry::TRIANGLE:
            triangleWeights(threshold, Trans, elmat);
            break;
         case Geometry::SQUARE:
            squareWeights(threshold, Trans, elmat);
            break;
         default:
            break;
      }
   }

};

HypreParMatrix* GetParAnisotropicGraph(ParFiniteElementSpace *fespace,
                                       VectorCoefficient &BCoef)
{
   ParBilinearForm g(fespace);
   g.AddDomainIntegrator(new AnisotropyPseudoIntegrator(BCoef));
   g.AddBoundaryIntegrator(new AnisotropyPseudoIntegrator(BCoef));
   g.Assemble();
   g.Finalize();

   HypreParMatrix * G = g.ParallelAssemble();
   return G;
}

std::vector<int>* GetPathCover(SparseMatrix *G, int& npath)
{
   const int nnode = G->Size();
   int nedge = 0;

   // Set nedge = G.number_of_edges()

   // TODO: keep it like this, or simplify by not storing -1 on diagonal for unflagged nodes?

   for (int r=0; r<nnode; ++r)
   {
      const int s = G->RowSize(r);
      int *cols = G->GetRowColumns(r);
      for (int i=0; i<s; ++i)
      {
         if (cols[i] != r)
         {
            nedge++;
         }
      }
   }

   MFEM_VERIFY(nedge > 0, "GetPathCover: nedge <= 0");

   std::vector<double> weight(nedge);
   std::vector<std::vector<int>> edgeList(nedge);

   std::vector<std::vector<int>> pathNeighbor(nnode);
   std::vector<int> *pathFlag = new std::vector<int>();
   pathFlag->assign(nnode, -2);

   for (int r=0; r<nnode; ++r)
   {
      pathNeighbor[r].assign(2, -2);  // Initialize to -2
   }

   int cnt = 0;
   for (int r=0; r<nnode; ++r)
   {
      const int s = G->RowSize(r);
      int *cols = G->GetRowColumns(r);
      double *vals = G->GetRowEntries(r);
      for (int i=0; i<s; ++i)
      {
         if (cols[i] != r)
         {
            weight[cnt] = vals[i];
            edgeList[cnt].resize(2);
            edgeList[cnt][0] = r;
            edgeList[cnt][1] = cols[i];
            cnt++;
         }
      }
   }

   MFEM_VERIFY(nedge == cnt, "GetPathCover: nedge != cnt");

   // Find edge indices in order of descending weight
   std::vector<int> isort(nedge);
   {
      for (int i=0; i<nedge; ++i)
      {
         isort[i] = i;
      }

      std::sort(isort.begin(), isort.end(), [&](const int& a, const int& b)
      {
         return (weight[a] > weight[b]);
      }
               );  // descending order
   }

   npath = 0;

   // TODO: how can pathNeighbor[u][1] be set but not pathNeighbor[u][0]?

   // Loop over edges in order of descending weight
   for (int i=0; i<nedge; ++i)
   {
      const int e = isort[i];
      const int u = edgeList[e][0];
      const int v = edgeList[e][1];

      // If neither node v0 nor node v1 is in a path, create a new path
      if (pathNeighbor[u][0] == -2 && pathNeighbor[u][1] == -2 &&
          pathNeighbor[v][0] == -2 && pathNeighbor[v][1] == -2)
      {
         pathNeighbor[u][0] = v;
         pathNeighbor[v][0] = u;
         (*pathFlag)[u] = npath;
         (*pathFlag)[v] = npath;
         npath++;
      }
      // node u is the end point of a path && node v is not in any path, append node v
      else if ((pathNeighbor[u][0] != -2 | pathNeighbor[u][1] != -2) &&
               pathNeighbor[v][0] == -2 && pathNeighbor[v][1] == -2)
      {
         if (pathNeighbor[u][0] == -2)
         {
            pathNeighbor[u][0] = v;
            pathNeighbor[v][0] = u;
            (*pathFlag)[v] = (*pathFlag)[u];
         }
         else
         {
            pathNeighbor[u][1] = v;
            pathNeighbor[v][0] = u;
            (*pathFlag)[v] = (*pathFlag)[u];
         }
      }
      // node v is the end point of a path and node u is not in any path, append node u
      else if (pathNeighbor[u][0] == -2 && pathNeighbor[u][1] == -2 &&
               (pathNeighbor[v][0] == -2 | pathNeighbor[v][1] == -2))
      {
         if (pathNeighbor[v][0] == -2)
         {
            pathNeighbor[v][0] = u;
            pathNeighbor[u][0] = v;
            (*pathFlag)[u] = (*pathFlag)[v];
         }
         else
         {
            pathNeighbor[v][1] = u;
            pathNeighbor[u][0] = v;
            (*pathFlag)[u] = (*pathFlag)[v];
         }
      }
      // both node u and node v are the end points of a path
      else if ((pathNeighbor[u][0] == -2 | pathNeighbor[u][1] == -2) &&
               (pathNeighbor[v][0] == -2 | pathNeighbor[v][1] == -2))
      {
         // node u and v are endpoints of different path, merge paths
         if ((*pathFlag)[u] != (*pathFlag)[v])
         {
            // connect node u and v
            if (pathNeighbor[u][0] == -2 && pathNeighbor[v][0] == -2)
            {
               pathNeighbor[u][0] = v;
               pathNeighbor[v][0] = u;
            }
            else if (pathNeighbor[u][0] == -2 && pathNeighbor[v][1] == -2)
            {
               pathNeighbor[u][0] = v;
               pathNeighbor[v][1] = u;
            }
            else if (pathNeighbor[u][1] == -2 && pathNeighbor[v][0] == -2)
            {
               pathNeighbor[u][1] = v;
               pathNeighbor[v][0] = u;
            }
            else
            {
               pathNeighbor[u][1] = v;
               pathNeighbor[v][1] = u;
            }

            // Merge the paths
            // TODO: more efficient implementation of this
            //pathFlag[pathFlag == pathFlag[v]] = pathFlag[u] # this can be done more efficiently
            const int pfv = (*pathFlag)[v];
            for (int j=0; j<nnode; ++j)
            {
               if ((*pathFlag)[j] == pfv)
               {
                  (*pathFlag)[j] = (*pathFlag)[u];
               }
            }
         }
      }
   } // loop over edges

   // Find the number of unique paths
   std::set<int> pathFlags;
   for (int j=0; j<nnode; ++j)
   {
      pathFlags.insert((*pathFlag)[j]);
   }

   npath = pathFlags.size();

   std::map<int, int> uniquePath;

   cnt = 0;
   for (auto p : pathFlags)
   {
      uniquePath[p] = cnt;
      cnt++;
   }

   // Reset path indices to range from 0 to npath-1

   int maxFlag = 0;
   for (int j=0; j<nnode; ++j)
   {
      (*pathFlag)[j] = uniquePath[(*pathFlag)[j]];
      maxFlag = std::max(maxFlag, (*pathFlag)[j]);
   }

   MFEM_VERIFY(npath == maxFlag + 1, "GetPathCover: npath != maxFlag + 1");

   return pathFlag;
}

int SetCoarseVertexLinePatches_GraphBased(ParMesh *pmesh,
                                          ParFiniteElementSpace *fespace,
                                          VectorCoefficient &BCoef,
                                          Array<int> & cdofToGlobalLine)
{
   SparseMatrix *G = GetAnisotropicGraph_with_distance(pmesh, BCoef);
   //G = solver.GetAnisotropicGraph_with_distance(mesh, Gmesh, coord, b, distance, threshold)
   if (false)
   {
      MPI_Comm comm = pmesh->GetComm();
      int num_procs, myid;
      MPI_Comm_size(comm, &num_procs);
      MPI_Comm_rank(comm, &myid);

      ostringstream oss;
      oss << "G_sp." << myid;
      ofstream ofs(oss.str());
      G->Print(ofs);
   }

   int npaths = 0;
   std::vector<int> *pathFlags = GetPathCover(G, npaths);

   delete G;

   MFEM_VERIFY(pathFlags->size() == pmesh->GetNV(), "");

   for (int i=0; i<pmesh->GetNV(); ++i)
   {
      Array<int> dofs;
      fespace->GetVertexDofs(i, dofs);
      MFEM_VERIFY(dofs.Size() == 1, "");

      cdofToGlobalLine[dofs[0]] = (*pathFlags)[i];
   }

   // ParaView output
   bool paraview = false;
   if (paraview)
   {
      MPI_Comm comm = pmesh->GetComm();
      int num_procs, myid;
      MPI_Comm_size(comm, &num_procs);
      MPI_Comm_rank(comm, &myid);

      Array<int> v(pmesh->GetNV()), edge(pmesh->GetNEdges());
      v = 0;
      edge = 0;

      for (int i=0; i<v.Size(); ++i)
      {
         v[i] = (*pathFlags)[i];
      }

      ostringstream oss;
      oss << "pv_patches_" << myid;
      ParaViewPrintAttributes(oss.str(), *pmesh, 1, &edge, &v);
   }

   return npaths;
}

int SetCoarseVertexLinePatches_xline(ParMesh *pmesh,
                                     ParFiniteElementSpace *fespace, Array<int> & cdofToGlobalLine)
{
   Vector linecrd;
   double tol = 1.0e-3;

   // TODO: input linecrd rather than hard-coding it as done here.
   const int nlines = 33;
   linecrd.SetSize(nlines);
   for (int i=0; i<nlines; ++i)
   {
      linecrd[i] = ((double) i) / ((double) (nlines - 1));
   }

   for (int i=0; i<pmesh->GetNV(); ++i)
   {
      const double y = pmesh->GetVertex(i)[1];
      int globalLineIndex = -1;
      for (int j=0; j<nlines; ++j)
      {
         if (fabs(linecrd[j] - y) < tol)
         {
            MFEM_VERIFY(globalLineIndex == -1, "");
            globalLineIndex = j;
         }
      }

      Array<int> dofs;
      fespace->GetVertexDofs(i, dofs);
      MFEM_VERIFY(dofs.Size() == 1, "");

      cdofToGlobalLine[dofs[0]] = globalLineIndex;
   }

   return nlines;
}

int SetCoarseVertexLinePatches_xy45line(ParMesh *pmesh,
                                        ParFiniteElementSpace *fespace, Array<int> & cdofToGlobalLine)
{
   Vector linecrd;
   double tol = 1.0e-3;

   // TODO: input linecrd rather than hard-coding it as done here.
   const int n = 33;
   const int nlines = (2*n) - 1;

   linecrd.SetSize(n);
   for (int i=0; i<n; ++i)
   {
      linecrd[i] = ((double) i) / ((double) (n-1));
   }

   for (int i=0; i<pmesh->GetNV(); ++i)
   {
      const double x = pmesh->GetVertex(i)[0];
      const double y = pmesh->GetVertex(i)[1];
      int globalLineIndex = -1;

      // First, loop over lines y = x - linecrd[i]
      for (int j=0; j<n; ++j)
      {
         if (fabs(linecrd[j] + y - x) < tol)
         {
            MFEM_VERIFY(globalLineIndex == -1, "");
            globalLineIndex = j;
         }
      }

      // Second, loop over lines y = x + linecrd[i+1]
      for (int j=0; j<n-1; ++j)
      {
         if (fabs(linecrd[j+1] + x - y) < tol)
         {
            MFEM_VERIFY(globalLineIndex == -1, "");
            globalLineIndex = n + j;
         }
      }

      Array<int> dofs;
      fespace->GetVertexDofs(i, dofs);
      MFEM_VERIFY(dofs.Size() == 1, "");

      cdofToGlobalLine[dofs[0]] = globalLineIndex;
   }

   return nlines;
}

void ReadPatches(string filename, vector<vector<int>> &patches)
{
   ifstream file(filename);
   string line, word;
   int count = 0;

   while (getline(file, line))
   {
      stringstream s(line);
      vector<int> patch;

      while (getline(s, word, ','))
      {
         if (word.empty())
         {
            break;
         }

         patch.push_back(stoi(word) - 1);  // Subtract 1 from Matlab indexing
      }

      cout << "Patch " << patches.size() << " has size " << patch.size() << endl;

      patches.push_back(patch);
   }

   file.close();
}

int SetCoarseVertexLinePatches_Matlab(ParMesh *pmesh,
                                      ParFiniteElementSpace *fespace,
                                      Array<int> & cdofToGlobalLine)
{
   Vector linecrd;
   double tol = 1.0e-3;

   vector<vector<int>> patches;
   ReadPatches("cover.csv", patches);
   const int nlines = patches.size();
   cout << "Read " << nlines << " from cover.csv" << endl;

   for (int i=0; i<nlines; ++i)
   {
      for (int j=0; j<patches[i].size(); ++j)
      {
         Array<int> dofs;
         fespace->GetVertexDofs(patches[i][j], dofs);
         MFEM_VERIFY(dofs.Size() == 1, "");

         cdofToGlobalLine[dofs[0]] = i;
      }
   }

   return nlines;
}

class PathNeighbor
{
private:
   // int num_nodes_;
   HYPRE_BigInt node_start_;
   HYPRE_BigInt node_end_;

   std::vector<std::vector<HYPRE_BigInt>> localPathNeighbor;
   mutable std::map<HYPRE_BigInt,std::vector<HYPRE_BigInt>> remotePathNeighbor;

public:
   PathNeighbor(HYPRE_BigInt node_start, int num_nodes)
      : // num_nodes_(num_nodes),
        node_start_(node_start),
        node_end_(node_start + num_nodes),
        localPathNeighbor(num_nodes)
   {}

   const std::vector<HYPRE_BigInt> &operator[](HYPRE_BigInt i) const
   {
      if (i >= node_start_ && i < node_end_)
      {
         return localPathNeighbor[i - node_start_];
      }
      else
      {
         if (remotePathNeighbor.find(i) == remotePathNeighbor.end())
         {
            remotePathNeighbor[i].resize(2);
            remotePathNeighbor[i][0] = -2;
            remotePathNeighbor[i][1] = -2;
         }
         return remotePathNeighbor[i];
      }
   }

   std::vector<HYPRE_BigInt> &operator[](HYPRE_BigInt i)
   {
      if (i >= node_start_ && i < node_end_)
      {
         return localPathNeighbor[i - node_start_];
      }
      else
      {
         if (remotePathNeighbor.find(i) == remotePathNeighbor.end())
         {
            remotePathNeighbor[i].resize(2);
            remotePathNeighbor[i][0] = -2;
            remotePathNeighbor[i][1] = -2;
         }
         return remotePathNeighbor[i];
      }
   }
};

class PathFlag
{
private:
   int num_nodes_;
   HYPRE_BigInt node_start_;
   HYPRE_BigInt node_end_;
   bool sync_needed_;

   std::vector<int> localPathFlag;
   mutable std::map<HYPRE_BigInt,int> remotePathFlag;

public:
   PathFlag(HYPRE_BigInt node_start, int num_nodes)
      : num_nodes_(num_nodes),
        node_start_(node_start),
        node_end_(node_start + num_nodes),
        sync_needed_(false)
   {
      localPathFlag.assign(num_nodes, -2);
   }

   bool SyncNeeded() const { return sync_needed_; }
   void Synchronized() { sync_needed_ = false; }

   std::vector<int> & GetLocalPathFlag() { return localPathFlag; }
   std::map<HYPRE_BigInt,int> & GetRemotePathFlag() { return remotePathFlag; }

   // Merge v into u
   void Merge(HYPRE_BigInt v, HYPRE_BigInt u)
   {
      std::map<HYPRE_BigInt,int>::iterator it;
      const int pfv = (*this)[v];
      const int pfu = (*this)[u];

      for (int i=0; i<num_nodes_; i++)
      {
         if (localPathFlag[i] == pfv)
         {
            localPathFlag[i] = pfu;
         }
      }
      for (it=remotePathFlag.begin(); it!=remotePathFlag.end(); it++)
      {
         if (it->second == pfv)
         {
            it->second = pfu;
            sync_needed_ = true;
         }
      }
   }

   void Renumber(HYPRE_BigInt u, int new_pfu)
   {
      std::map<HYPRE_BigInt,int>::iterator it;
      const int pfu = (*this)[u];

      if (pfu == new_pfu) { return; }

      for (int i=0; i<num_nodes_; i++)
      {
         if (localPathFlag[i] == pfu)
         {
            localPathFlag[i] = new_pfu;
         }
      }
      for (it=remotePathFlag.begin(); it!=remotePathFlag.end(); it++)
      {
         if (it->second == pfu)
         {
            it->second = new_pfu;
            sync_needed_ = true;
         }
      }
   }

   const int operator[](HYPRE_BigInt i) const
   {
      if (i >= node_start_ && i < node_end_)
      {
         return localPathFlag[i - node_start_];
      }
      else
      {
         if (remotePathFlag.find(i) == remotePathFlag.end())
         {
            remotePathFlag[i] = -2;
         }
         return remotePathFlag[i];
      }
   }

   int &operator[](HYPRE_BigInt i)
   {
      if (i >= node_start_ && i < node_end_)
      {
         return localPathFlag[i - node_start_];
      }
      else
      {
         sync_needed_ = true;
         if (remotePathFlag.find(i) == remotePathFlag.end())
         {
            remotePathFlag[i] = -2;
         }
         return remotePathFlag[i];
      }
   }
};

std::vector<int>* GetPathCover(ParFiniteElementSpace *fespace,
                               HypreParMatrix *Gpar, int& npath)
{
   MPI_Comm comm = fespace->GetComm();
   int num_procs = fespace->GetNRanks();
   int myid = fespace->GetMyRank();

   HYPRE_BigInt row_start = fespace->GetMyTDofOffset();
   /*
   HYPRE_BigInt row_start = 0;
   {
      const HYPRE_BigInt *row_part = Gpar->RowPart();

      if (HYPRE_AssumedPartitionCheck())
      {
         row_start = row_part[0];
      }
      else
      {
         row_start = row_part[myid];
      }
   }
   */

   SparseMatrix Gsp;
   cout << myid << " Calling Gpar->MergeDiagAndOffd(Gsp)" << endl;
   Gpar->MergeDiagAndOffd(Gsp);
   cout << myid << " Calling Gsp.Threshold(1e-6)" << endl;
   Gsp.Threshold(1e-6);
   if (true)
   {
      ostringstream oss;
      oss << "G_par_merge." << myid;
      ofstream ofs(oss.str());
      Gsp.Print(ofs);
   }
   // cout << "Calling GetPathCover(&Gsp, npath)" << endl;
   // return GetPathCover(&Gsp, npath);

   const int nnode = Gsp.Size();
   int nedge = 0;

   for (int r=0; r<nnode; ++r)
   {
      const int s = Gsp.RowSize(r);
      int *cols = Gsp.GetRowColumns(r);
      for (int i=0; i<s; ++i)
      {
         if (cols[i] != r + row_start)
         {
            nedge++;
         }
      }
   }
   cout << myid << " GetPathCover 1: " << nedge << endl;
   std::vector<double> weight(nedge+1);
   std::vector<std::vector<HYPRE_BigInt>> edgeList(nedge);

   // std::vector<std::vector<int>> localPathNeighbor(nnode);
   // std::map<int,std::vector<int>> remotePathNeighbor;
   PathNeighbor pathNeighbor(row_start, nnode);
   PathFlag pathFlag(row_start, nnode);
   ParGridFunction pathGF(fespace);
   pathGF = -2;

   std::vector<int> *pathFlagPtr = new std::vector<int>();
   pathFlagPtr->assign(nnode, -2);
   cout << myid << " GetPathCover 2" << endl;
   for (int r=0; r<nnode; ++r)
   {
      pathNeighbor[r + row_start].assign(2, -2);  // Initialize to -2
   }
   cout << myid << " GetPathCover 3" << endl;
   int cnt = 0;
   for (int r=0; r<nnode; ++r)
   {
      const int s = Gsp.RowSize(r);
      int *cols = Gsp.GetRowColumns(r);
      double *vals = Gsp.GetRowEntries(r);
      for (int i=0; i<s; ++i)
      {
         if (cols[i] != r + row_start)
         {
            weight[cnt] = vals[i];
            edgeList[cnt].resize(2);
            edgeList[cnt][0] = r + row_start;
            edgeList[cnt][1] = cols[i];
            cnt++;
         }
      }
   }
   weight[nedge] = 0.0;

   MFEM_VERIFY(nedge == cnt, "GetPathCover: nedge != cnt");
   cout << myid << " GetPathCover 4" << endl;
   // Find edge indices in order of descending weight
   std::vector<int> isort(nedge+1);
   {
      for (int i=0; i<=nedge; ++i)
      {
         isort[i] = i;
      }

      std::sort(isort.begin(), isort.end(),
                [&](const int& a, const int& b)
      {
         return (weight[a] > weight[b]);
      }
               );  // descending order
   }

   double max_weight = 0.0;
   {
      double loc_max_weight = weight[isort[0]];
      MPI_Allreduce(&loc_max_weight, &max_weight, 1, MPI_DOUBLE, MPI_MAX, comm);
      cout << myid << " Weights: " << loc_max_weight << " " << max_weight
           << endl;
   }
   double pause_weight = 0.99 * max_weight;

   cout << myid << " GetPathCover 5" << endl;
   npath = row_start;

   // Loop over edges in order of descending weight
   // for (int i=0; i<nedge; ++i)
   int ie = 0;
   bool more_edges = true;
   while (ie < nedge || more_edges)
   {
      if (ie < nedge)
      {
         const int e = isort[ie];
         const HYPRE_BigInt u = edgeList[e][0];
         const HYPRE_BigInt v = edgeList[e][1];
         cout << myid << " edge " << e << " " << u << " " << v
              << " pathFlag.SyncNeeded() " << pathFlag.SyncNeeded() << endl;

         // If neither node v0 nor node v1 is in a path, create a new path
         if (pathNeighbor[u][0] == -2 && pathNeighbor[u][1] == -2 &&
             pathNeighbor[v][0] == -2 && pathNeighbor[v][1] == -2)
         {
            pathNeighbor[u][0] = v;
            pathNeighbor[v][0] = u;
            pathFlag[u] = npath;
            pathFlag[v] = npath;
            npath++;
         }
         // node u is the end point of a path && node v is not in any path,
         // append node v
         else if ((pathNeighbor[u][0] != -2 | pathNeighbor[u][1] != -2) &&
                  pathNeighbor[v][0] == -2 && pathNeighbor[v][1] == -2)
         {
            if (pathNeighbor[u][0] == -2)
            {
               pathNeighbor[u][0] = v;
               pathNeighbor[v][0] = u;
               pathFlag[v] = pathFlag[u];
            }
            else
            {
               pathNeighbor[u][1] = v;
               pathNeighbor[v][0] = u;
               pathFlag[v] = pathFlag[u];
            }
         }
         // node v is the end point of a path and node u is not in any path,
         // append node u
         else if (pathNeighbor[u][0] == -2 && pathNeighbor[u][1] == -2 &&
                  (pathNeighbor[v][0] == -2 | pathNeighbor[v][1] == -2))
         {
            if (pathNeighbor[v][0] == -2)
            {
               pathNeighbor[v][0] = u;
               pathNeighbor[u][0] = v;
               pathFlag[u] = pathFlag[v];
            }
            else
            {
               pathNeighbor[v][1] = u;
               pathNeighbor[u][0] = v;
               pathFlag[u] = pathFlag[v];
            }
         }
         // both node u and node v are the end points of a path
         else if ((pathNeighbor[u][0] == -2 | pathNeighbor[u][1] == -2) &&
                  (pathNeighbor[v][0] == -2 | pathNeighbor[v][1] == -2))
         {
            // node u and v are endpoints of different path, merge paths
            if (pathFlag[u] != pathFlag[v])
            {
               // connect node u and v
               if (pathNeighbor[u][0] == -2 && pathNeighbor[v][0] == -2)
               {
                  pathNeighbor[u][0] = v;
                  pathNeighbor[v][0] = u;
               }
               else if (pathNeighbor[u][0] == -2 && pathNeighbor[v][1] == -2)
               {
                  pathNeighbor[u][0] = v;
                  pathNeighbor[v][1] = u;
               }
               else if (pathNeighbor[u][1] == -2 && pathNeighbor[v][0] == -2)
               {
                  pathNeighbor[u][1] = v;
                  pathNeighbor[v][0] = u;
               }
               else
               {
                  pathNeighbor[u][1] = v;
                  pathNeighbor[v][1] = u;
               }

               // Merge the paths
               // TODO: more efficient implementation of this
               //pathFlag[pathFlag == pathFlag[v]] = pathFlag[u] # this can be done more efficiently
               pathFlag.Merge(v, u);
            }
            else
            {
               cout << myid << " skipping edge " << e
                    << " between vertices " << u << " and " << v << " with weight " << weight[e] <<
                    endl;
            }
         }
         cout << myid << " edge " << e << " " << u << " " << v
              << " pathFlag.SyncNeeded() " << pathFlag.SyncNeeded() << endl;

         ie++;
      }

      if (weight[isort[ie]] < pause_weight)
      {
         {
            bool loc_more_edges = ie < nedge;
            MPI_Allreduce(&loc_more_edges, &more_edges, 1,
                          MPI_CXX_BOOL, MPI_LOR, comm);

         }

         bool sync_needed = false;
         {
            bool loc_sync_needed = pathFlag.SyncNeeded();
            MPI_Allreduce(&loc_sync_needed, &sync_needed, 1,
                          MPI_CXX_BOOL, MPI_LOR, comm);

            cout << myid << " loc_sync_needed " << loc_sync_needed << ", sync_needed " <<
                 sync_needed << endl;
         }

         // Exchange path info across processor boundaries
         //
         if (sync_needed)
         {
            std::map<HYPRE_BigInt,int> &remotePathFlag =
               pathFlag.GetRemotePathFlag();
            std::map<HYPRE_BigInt,int>::iterator it;

            while (sync_needed)
            {
               for (int p=0; p<num_procs; p++)
               {
                  int len = (myid == p && pathFlag.SyncNeeded()) ?
                            (2 * remotePathFlag.size()) : 0;
                  MPI_Bcast(&len, 1, MPI_INTEGER, p, comm);

                  if (len > 0)
                  {
                     HYPRE_BigInt *buf = new HYPRE_BigInt[len];
                     if (myid == p)
                     {
                        int ib = 0;
                        for (it=remotePathFlag.begin();
                             it!=remotePathFlag.end(); it++, ib+=2)
                        {
                           buf[ib] = it->first;
                           buf[ib+1] = it->second;
                        }

                        pathFlag.Synchronized();
                     }
                     MPI_Bcast(buf, len, HYPRE_MPI_BIG_INT, p, comm);

                     if (myid != p)
                     {
                        for (int ib=0; ib<len; ib +=2)
                        {
                           HYPRE_BigInt u = buf[ib];
                           int pfu = (int)buf[ib+1];
                           if (u >= row_start && u < row_start+nnode)
                           {
                              pathFlag.Renumber(u,pfu);
                           }
                        }
                     }
                     delete [] buf;
                  }
               }

               {
                  bool loc_sync_needed = pathFlag.SyncNeeded();
                  MPI_Allreduce(&loc_sync_needed, &sync_needed, 1,
                                MPI_CXX_BOOL, MPI_LOR, comm);
               }

            }
            // pathFlag.Synchronized();
         }

         if (more_edges)
         {
            double loc_max_weight = weight[isort[ie]];
            MPI_Allreduce(&loc_max_weight, &max_weight, 1,
                          MPI_DOUBLE, MPI_MAX, comm);
            cout << myid << " Weights: " << loc_max_weight << " "
                 << max_weight << endl;
            pause_weight = 0.99 * max_weight;
         }

      }
   } // loop over edges

   std::vector<int> & localPathFlag = pathFlag.GetLocalPathFlag();
   std::map<HYPRE_BigInt,int> &remotePathFlag =
      pathFlag.GetRemotePathFlag();

   {
      ostringstream oss; oss << "paths." << myid;
      ofstream ofs(oss.str());
      ofs << "{";
      for (int i=0; i<localPathFlag.size(); i++)
      {
         ofs << "{" << i + row_start << "," << localPathFlag[i] << "}";
         if (i < localPathFlag.size()-1) { ofs << ","; }
      }
      std::map<HYPRE_BigInt,int>::iterator it;
      for (it=remotePathFlag.begin(); it!=remotePathFlag.end();)
      {
         ofs << "{" << it->first << "," << it->second << "}";
         it++;
         if (it != remotePathFlag.end()) { ofs << ","; }
      }
      ofs << "}";
   }

   int loc_nrem = remotePathFlag.size();
   int glb_nrem;
   MPI_Allreduce(&loc_nrem, &glb_nrem, 1, MPI_INT, MPI_MAX, comm);

   if (glb_nrem > 0)
   {
      cout << myid << " nrem " << loc_nrem << " " << glb_nrem << endl;
      /*
      for (int p=0; p<num_procs; p++)
      {
      int nkeys;
      HYPRE_BigInt *buf = NULL;

      if (p == myid)
      {

        nkeys = remotePathFlag.size();
        buf = new HYPRE_BigInt[2*nkeys];

        std::map<HYPRE_BigInt,int>::iterator it = remotePathFlag.begin();

        for (int j=0; it!=remotePathFlag.end(); it++, j++)
        {
      sendbuf[2*j+0] = it->first;
      sendbuf[2*j+1] = it->second;
      cout << it->first << " -> " << it->second << endl;
        }
      }

      int
        MPI_Scatter(&nkeys, 1, MPI_INTEGER,
         &nkeys, 1, MPI_INTEGER);

      delete [] buf;
      }
      else
      {
      }
      */
   }
   /*
   // Update pause_weight
   double loc_max_weight = weight[isort[ie]];
   MPI_Allreduce(&loc_max_weight, &max_weight, 1, MPI_DOUBLE, MPI_MAX,
       comm);
   cout << myid << " Weights: " << loc_max_weight << " " << max_weight
   << endl;
   pause_weight = 0.99 * max_weight;
   if (max_weight < 0.1) { sync_remote = false; }
   */
   cout << myid << " GetPathCover 6" << endl;
   MPI_Barrier(comm);

   return pathFlagPtr;
}

int SetCoarseVertexLinePatches_ParGraphBased(ParMesh *pmesh,
                                             ParFiniteElementSpace *fespace,
                                             VectorCoefficient &BCoef,
                                             Array<int> & cdofToGlobalLine)
{
   HypreParMatrix *G = GetParAnisotropicGraph(fespace, BCoef);
   if (true)
   {
      G->Print("G_par");
   }

   int npaths = 0;
   std::vector<int> *pathFlags = GetPathCover(fespace, G, npaths);

   delete G;

   MFEM_VERIFY(pathFlags->size() == pmesh->GetNV(),
               "SetCoarseVertexLinePatches_ParGraphBased: pathFlags->size() != pmesh->GetNV()");

   for (int i=0; i<pmesh->GetNV(); ++i)
   {
      Array<int> dofs;
      fespace->GetVertexDofs(i, dofs);
      MFEM_VERIFY(dofs.Size() == 1, "");

      cdofToGlobalLine[dofs[0]] = (*pathFlags)[i];
   }

   return npaths;
}

LinePatchInfo::LinePatchInfo(ParMesh *pmesh_, VectorCoefficient &BCoef_,
                             int ref_levels_)
   : pmesh(pmesh_), BCoef(BCoef_), ref_levels(ref_levels_)
{
   int dim = pmesh->Dimension();

   // 1. Define an auxiliary parallel H1 finite element space on the parallel mesh.
   FiniteElementCollection * aux_fec = new H1_FECollection(1, dim);
   ParFiniteElementSpace * aux_fespace = new ParFiniteElementSpace(pmesh, aux_fec);
   int mycdofoffset =
      aux_fespace->GetMyDofOffset(); // dof offset for the coarse mesh

   // Set map from coarse vertex DOFs to global line patches
   Array<int> cdofToGlobalLine(aux_fespace->GetNDofs());
   cdofToGlobalLine = -1;
   //const int nlines = SetCoarseVertexLinePatches_xline(pmesh, aux_fespace, cdofToGlobalLine);
   //const int nlines = SetCoarseVertexLinePatches_xy45line(pmesh, aux_fespace, cdofToGlobalLine);

   //int nlines = SetCoarseVertexLinePatches_Matlab(pmesh, aux_fespace, cdofToGlobalLine);
   int nlines = SetCoarseVertexLinePatches_GraphBased(pmesh, aux_fespace,
                                                      BCoef,
                                                      cdofToGlobalLine);

   // 2. Store the cDofTrueDof Matrix. Required after the refinements
   HypreParMatrix *cDofTrueDof = new HypreParMatrix(
      *aux_fespace->Dof_TrueDof_Matrix());

   //cdofToGlobalLine.Print(std::cout);
   if (cdofToGlobalLine.Min() == -1)
   {
      // Check how many nodes are not in a patch
      int cnt=0;
      for (int i=0; i<cdofToGlobalLine.Size(); ++i)
      {
         if (cdofToGlobalLine[i] < 0)
         {
            cdofToGlobalLine[i] = nlines;
            nlines++;
            cnt++;
         }
      }

      cout << "WARNING: " << cnt << " out of " << cdofToGlobalLine.Size() <<
           " nodes are not in a patch and will made into separate patches " << endl;
   }

   MFEM_VERIFY(cDofTrueDof->Height() == cdofToGlobalLine.Size(), "");
   MFEM_VERIFY(cdofToGlobalLine.Min() == 0,
               ""); // TODO: in parallel, just check >= 0

   // 3. Perform the refinements (if any) and Get the final Prolongation operator
   HypreParMatrix *Pr = nullptr;
   for (int i = 0; i < ref_levels; i++)
   {
      const ParFiniteElementSpace cfespace(*aux_fespace);
      pmesh->UniformRefinement();
      // Update fespace
      aux_fespace->Update();
      OperatorHandle Tr(Operator::Hypre_ParCSR);
      aux_fespace->GetTrueTransferOperator(cfespace, Tr);
      Tr.SetOperatorOwner(false);
      HypreParMatrix *P;
      Tr.Get(P);
      if (!Pr)
      {
         Pr = P;
      }
      else
      {
         Pr = ParMult(P, Pr);
      }
   }
   if (Pr) { Pr->Threshold(0.0); }

   // 4. Get the DofTrueDof map on this mesh and convert the prolongation matrix
   // to correspond to global dof numbering (from true dofs to dofs)
   HypreParMatrix *DofTrueDof = aux_fespace->Dof_TrueDof_Matrix();
   HypreParMatrix *A = nullptr;
   if (Pr)
   {
      A = ParMult(DofTrueDof, Pr);
   }
   else
   {
      // If there is no refinement then the prolongation is the identity
      A = DofTrueDof;
   }
   HypreParMatrix * cDofTrueDofT = cDofTrueDof->Transpose();
   HypreParMatrix *B = ParMult(A, cDofTrueDofT);
   delete cDofTrueDofT;
   // 5. Now we compute the vertices that are owned by the process
   SparseMatrix cdiag, coffd;
   cDofTrueDof->GetDiag(cdiag);

   MFEM_VERIFY(cdiag.Height() == cdofToGlobalLine.Size(), "");

   Array<int> cown_vertices;
   Array<int> cown_vertices_line;
   std::set<int> touched_lines;
   int cnv = 0;
   for (int k = 0; k < cdiag.Height(); k++)
   {
      int nz = cdiag.RowSize(k);
      int i = mycdofoffset + k;
      if (nz != 0)
      {
         cnv++;
         cown_vertices.SetSize(cnv);
         cown_vertices[cnv - 1] = i;

         cown_vertices_line.SetSize(cnv);
         cown_vertices_line[cnv - 1] = cdofToGlobalLine[k];
         touched_lines.insert(cdofToGlobalLine[k]);
      }
   }

   MPI_Comm comm = pmesh->GetComm();
   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   // Determine ownership of patches, based on minimum MPI rank touching the line patches.
   Array<int> owned_lines;
   {
      std::vector<int> touchingRank(nlines);
      //std::vector<int> owningRank(nlines);

      host_rank.SetSize(nlines);

      touchingRank.assign(nlines, num_procs);
      //owningRank.assign(nlines, num_procs);
      for (std::set<int>::const_iterator it = touched_lines.begin();
           it != touched_lines.end(); ++it)
      {
         touchingRank[*it] = myid;
      }

      MPI_Allreduce(touchingRank.data(), host_rank.GetData(), nlines, MPI_INT,
                    MPI_MIN, comm);

      mynrpatch = 0;
      for (int i=0; i<nlines; ++i)
      {
         if (host_rank[i] == myid)
         {
            mynrpatch++;
         }
      }

      owned_lines.SetSize(mynrpatch);

      mynrpatch = 0;
      for (int i=0; i<nlines; ++i)
      {
         if (host_rank[i] == myid)
         {
            owned_lines[mynrpatch] = i;
            mynrpatch++;
         }
      }
   }

   // 6. Compute total number of patches
   //mynrpatch = cown_vertices.Size();
   mynrpatch = owned_lines.Size();
   // Compute total number of patches.

   MPI_Allreduce(&mynrpatch, &nrpatch, 1, MPI_INT, MPI_SUM, comm);

   MFEM_VERIFY(nrpatch == nlines, "nrpatch not equal to nlines");

   //patch_global_dofs_ids.SetSize(nrpatch);

   /*
   // Create a list of patches identifiers to all procs

   int count[num_procs];

   MPI_Allgather(&mynrpatch, 1, MPI_INT, &count[0], 1, MPI_INT, comm);
   int displs[num_procs];
   displs[0] = 0;
   for (int i = 1; i < num_procs; i++)
   {
      displs[i] = displs[i - 1] + count[i - 1];
   }

   int * cownvert_ptr = nullptr;
   int * dof_rank_id_ptr = nullptr;
   Array<int> dof_rank_id;
   if (cown_vertices.Size() >0)
   {
      cownvert_ptr = &cown_vertices[0];
      dof_rank_id.SetSize(cown_vertices.Size());
      dof_rank_id = myid;
      dof_rank_id_ptr = &dof_rank_id[0];
   }
   // send also the rank number for each global dof
   host_rank.SetSize(nrpatch);
   MPI_Allgatherv(cownvert_ptr, mynrpatch, MPI_INT, &patch_global_dofs_ids[0],
                  count, displs, MPI_INT, comm);
   MPI_Allgatherv(dof_rank_id_ptr, mynrpatch, MPI_INT, &host_rank[0], count,
                  displs, MPI_INT, comm);

   int size = patch_global_dofs_ids[nrpatch - 1] + 1;
   patch_natural_order_idx.SetSize(size);
   // TODO: isn't this size too big? Can a map replace this array? In serial it is fine, but it seems to depend on the global number of patches, which may be fine for moderate problem sizes but could get large and hinder scalability.
   //cout << "DYLAN DBG: patch_natural_order_idx size " << size << ", nrpatch " << nrpatch << ", aux dofs "
   //<< aux_fespace->GlobalVSize() << endl;

   // initialize with -1
   patch_natural_order_idx = -1;
   for (int i = 0; i < nrpatch; i++)
   {
      int k = patch_global_dofs_ids[i];
      patch_natural_order_idx[k] = i;
   }
   */

   patch_natural_order_idx.SetSize(nrpatch);
   for (int i = 0; i < nrpatch; i++)
   {
      patch_natural_order_idx[i] = i;
   }

   int nvert = pmesh->GetNV();
   // first find all the contributions of the vertices
   vert_contr.resize(nvert);
   SparseMatrix H1pr_diag;
   B->GetDiag(H1pr_diag);
   for (int i = 0; i < nvert; i++)
   {
      int row = i;
      int row_size = H1pr_diag.RowSize(row);
      int *col = H1pr_diag.GetRowColumns(row);
      for (int j = 0; j < row_size; j++)
      {
         const int globalLine = cdofToGlobalLine[col[j]];
         vert_contr[i].Append(globalLine);
         /*
               int jv = col[j] + mycdofoffset;
               if (is_a_patch(jv, patch_global_dofs_ids))
               {
                  vert_contr[i].Append(jv);
               }
         */
      }
   }

   SparseMatrix H1pr_offd;
   int *cmap;
   B->GetOffd(H1pr_offd, cmap);
   for (int i = 0; i < nvert; i++)
   {
      int row = i;
      int row_size = H1pr_offd.RowSize(row);
      int *col = H1pr_offd.GetRowColumns(row);
      for (int j = 0; j < row_size; j++)
      {
         const int globalLine = cdofToGlobalLine[col[j]];
         vert_contr[i].Append(globalLine);

         /*
               int jv = cmap[col[j]];
               if (is_a_patch(jv, patch_global_dofs_ids))
               {
                  vert_contr[i].Append(jv);
               }
         */
      }
   }

   Array<int> edge_vertices;
   int nedge = pmesh->GetNEdges();
   edge_contr.resize(nedge);
   for (int ie = 0; ie < nedge; ie++)
   {
      pmesh->GetEdgeVertices(ie, edge_vertices);
      int nv = edge_vertices.Size(); // always 2 but ok
      // The edge will contribute to the same patches as its vertices
      for (int iv = 0; iv < nv; iv++)
      {
         int ivert = edge_vertices[iv];
         edge_contr[ie].Append(vert_contr[ivert]);
      }
      edge_contr[ie].Sort();
      edge_contr[ie].Unique();
   }
   // -----------------------------------------------------------------------
   // done with edges. Now the faces
   // -----------------------------------------------------------------------
   Array<int> face_vertices;
   int nface = pmesh->GetNFaces();
   face_contr.resize(nface);
   for (int ifc = 0; ifc < nface; ifc++)
   {
      pmesh->GetFaceVertices(ifc, face_vertices);
      int nv = face_vertices.Size();
      // The face will contribute to the same patches as its vertices
      for (int iv = 0; iv < nv; iv++)
      {
         int ivert = face_vertices[iv];
         face_contr[ifc].Append(vert_contr[ivert]);
      }
      face_contr[ifc].Sort();
      face_contr[ifc].Unique();
   }
   // -----------------------------------------------------------------------
   // Finally the elements
   // -----------------------------------------------------------------------
   Array<int> elem_vertices;
   int nelem = pmesh->GetNE();
   elem_contr.resize(nelem);
   for (int iel = 0; iel < nelem; iel++)
   {
      pmesh->GetElementVertices(iel, elem_vertices);
      int nv = elem_vertices.Size();
      // The element will contribute to the same patches as its vertices
      for (int iv = 0; iv < nv; iv++)
      {
         int ivert = elem_vertices[iv];
         elem_contr[iel].Append(vert_contr[ivert]);
      }
      elem_contr[iel].Sort();
      elem_contr[iel].Unique();
   }
   if (Pr) { delete A; }
   delete B;
   if (Pr) { delete Pr; }
   delete cDofTrueDof;
   delete aux_fespace;
   delete aux_fec;
}

PatchDofInfo::PatchDofInfo(ParMesh *pmesh_, VectorCoefficient &BCoef_,
                           int ref_levels_, ParFiniteElementSpace *fespace)
{
   //VertexPatchInfo *patch_nodes = new VertexPatchInfo(pmesh_, ref_levels_);
   LinePatchInfo *patch_nodes = new LinePatchInfo(pmesh_, BCoef_, ref_levels_);

   int num_procs, myid;
   comm = pmesh_->GetComm();
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   // Build a list on each processor identifying the truedofs in each patch
   // First the vertices
   nrpatch = patch_nodes->nrpatch;
   host_rank = patch_nodes->host_rank;

   patch_local_tdofs.resize(nrpatch);
   int * offs = fespace->GetTrueDofOffsets();
   int nrvert = fespace->GetNV();
   for (int i = 0; i < nrvert; i++)
   {
      int np = patch_nodes->vert_contr[i].Size();
      if (np == 0) { continue; }
      Array<int> vertex_dofs;
      fespace->GetVertexDofs(i, vertex_dofs);
      int nv = vertex_dofs.Size();
      for (int j = 0; j < np; j++)
      {
         int k = patch_nodes->vert_contr[i][j];
         int kk = patch_nodes->patch_natural_order_idx[k];
         for (int l = 0; l < nv; l++)
         {
            int m = fespace->GetGlobalTDofNumber(vertex_dofs[l]);
            if (owned(m,offs)) { patch_local_tdofs[kk].Append(m); }
         }
      }
   }

   int nedge = fespace->GetMesh()->GetNEdges();
   for (int i = 0; i < nedge; i++)
   {
      int np = patch_nodes->edge_contr[i].Size();
      if (np == 0) { continue; }
      Array<int> edge_dofs;
      fespace->GetEdgeInteriorDofs(i, edge_dofs);
      int nv = edge_dofs.Size();
      for (int j = 0; j < np; j++)
      {
         int k = patch_nodes->edge_contr[i][j];
         int kk = patch_nodes->patch_natural_order_idx[k];
         for (int l = 0; l < nv; l++)
         {
            int m = fespace->GetGlobalTDofNumber(edge_dofs[l]);
            if (owned(m,offs)) { patch_local_tdofs[kk].Append(m); }
         }
      }
   }
   int nface = fespace->GetMesh()->GetNFaces();
   for (int i = 0; i < nface; i++)
   {
      int np = patch_nodes->face_contr[i].Size();
      if (np == 0) { continue; }
      Array<int> face_dofs;
      fespace->GetFaceInteriorDofs(i, face_dofs);
      int nv = face_dofs.Size();
      for (int j = 0; j < np; j++)
      {
         int k = patch_nodes->face_contr[i][j];
         int kk = patch_nodes->patch_natural_order_idx[k];
         for (int l = 0; l < nv; l++)
         {
            int m = fespace->GetGlobalTDofNumber(face_dofs[l]);
            if (owned(m,offs)) { patch_local_tdofs[kk].Append(m); }
         }
      }
   }
   int nelem = fespace->GetNE();
   for (int i = 0; i < nelem; i++)
   {
      int np = patch_nodes->elem_contr[i].Size();
      if (np == 0) { continue; }
      Array<int> elem_dofs;
      fespace->GetElementInteriorDofs(i, elem_dofs);
      int nv = elem_dofs.Size();
      for (int j = 0; j < np; j++)
      {
         int k = patch_nodes->elem_contr[i][j];
         int kk = patch_nodes->patch_natural_order_idx[k];
         for (int l = 0; l < nv; l++)
         {
            int m = fespace->GetGlobalTDofNumber(elem_dofs[l]);
            if (owned(m,offs)) { patch_local_tdofs[kk].Append(m); }
         }
      }
   }
   delete patch_nodes;

   patch_tdofs.resize(nrpatch);
   for (int i = 0; i < nrpatch; i++)
   {
      Array<int> count(num_procs);
      int size = patch_local_tdofs[i].Size();

      count[myid] = size;
      MPI_Allgather(&size, 1, MPI_INT, &count[0], 1, MPI_INT, comm);
      Array<int>displs(num_procs);
      displs[0] = 0;
      for (int j = 1; j < num_procs; j++)
      {
         displs[j] = displs[j-1] + count[j-1];
      }
      int tot_size = displs[num_procs - 1] + count[num_procs - 1];
      // Get a group identifier for comm.
      MPI_Group world_group_id;
      MPI_Comm new_comm = MPI_COMM_NULL;
      MPI_Group new_group_id;
      MPI_Comm_group (comm, &world_group_id);
      // count the ranks that do not have zero length
      int num_ranks = 0;
      for (int k = 0; k<num_procs; k++)
      {
         if (count[k] != 0) {num_ranks++;}
      }
      Array<int> new_count(num_ranks);
      Array<int> new_displs(num_ranks);
      int sub_comm_ranks[num_ranks];
      num_ranks = 0;
      for (int j = 0; j <num_procs ; j++ )
      {
         if (count[j] != 0)
         {
            sub_comm_ranks[num_ranks] = j;
            new_count[num_ranks] = count[j];
            new_displs[num_ranks] = displs[j];
            num_ranks++;
         }
      }
      MPI_Group_incl(world_group_id, num_ranks, sub_comm_ranks, &new_group_id);
      MPI_Comm_create(comm, new_group_id, &new_comm);
      if (size != 0)
      {
         patch_tdofs[i].SetSize(tot_size);
         MPI_Allgatherv(&patch_local_tdofs[i][0],size,MPI_INT,
                        &patch_tdofs[i][0],new_count,new_displs,MPI_INT,new_comm);
      }
      MPI_Group_free(&world_group_id);
      MPI_Group_free(&new_group_id);
      if (new_comm != MPI_COMM_NULL) { MPI_Comm_free(&new_comm); }
   }
}

PatchAssembly::PatchAssembly(ParMesh *cpmesh_, VectorCoefficient &BCoef_,
                             int ref_levels_, ParFiniteElementSpace *fespace_,
                             HypreParMatrix * A_)
   :  A(A_), fespace(fespace_)
{
   comm = A->GetComm();
   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);
   compute_trueoffsets();
   SparseMatrix diag;
   SparseMatrix offd;
   int *cmap;
   A->GetDiag(diag);
   A->GetOffd(offd,cmap);
   int *row_start = A->GetRowStarts();
   SparseMatrix offdT;
   int *cmapT;
   HypreParMatrix * At = A->Transpose();
   At->GetOffd(offdT,cmapT);
   int *row_startT = At->GetRowStarts();
   diag.SortColumnIndices();

   patch_tdof_info = new PatchDofInfo(cpmesh_, BCoef_, ref_levels_,fespace);

   nrpatch = patch_tdof_info->nrpatch;
   host_rank.SetSize(nrpatch); host_rank = -1;
   // This can be changed later. For now the required lists are
   // constructed from the whole list
   patch_other_tdofs.resize(nrpatch);
   patch_owned_other_tdofs.resize(nrpatch);
   for (int ip = 0; ip < nrpatch; ip++)
   {
      int ndof = patch_tdof_info->patch_tdofs[ip].Size();

      if (ndof !=0 )
      {
         host_rank[ip] = patch_tdof_info->host_rank[ip];
         // host_rank[ip] = get_rank(patch_tdof_info->patch_tdofs[ip][0]);
         for (int i=0; i<ndof; i++)
         {
            int tdof = patch_tdof_info->patch_tdofs[ip][i];
            int tdof_rank = get_rank(tdof);
            if (tdof_rank != host_rank[ip])
            {
               patch_other_tdofs[ip].Append(tdof);
            }
         }
         GetArrayIntersection(patch_other_tdofs[ip],
                              patch_tdof_info->patch_local_tdofs[ip], patch_owned_other_tdofs[ip]);
      }
   }
   // For the construction of the matrix of a patch we follow the following procedure.
   // The matrix will be split to a 2x2 block matrix where:
   // Block (0,0) is constructed by the dofs owned by the processor (using diag and RAP)
   // Block (0,1) is constructed by the dofs owned by the processor (using offd)
   // Block (1,0) is the Transpose of (0,1) (for now the support is only for symmetric matrices)
   // Block (1,1) has to be communicated among processors. Its constructed by the dofs not owned by the processor.

   Array<SparseMatrix * > PatchMat00(nrpatch);
   l2gmaps.resize(nrpatch);

   //--------------------------------------------------------------------------------------
   // Construction of (0,0): This is done with RAP
   //--------------------------------------------------------------------------------------

   for (int ip = 0; ip < nrpatch; ip++)
   {
      PatchMat00[ip]=nullptr;
      if (myid == host_rank[ip])
      {
         int num_cols = patch_tdof_info->patch_local_tdofs[ip].Size();
         int num_rows = diag.Height();
         // loop through rows
         // l2gmaps[ip].SetSize(num_cols);
         GetLocal2GlobalMap(patch_tdof_info->patch_local_tdofs[ip],
                            row_start, num_rows, num_cols, l2gmaps[ip]);
         // Build prolongation (temporary to perform RAP)
         SparseMatrix Prl(num_rows,num_cols);
         for (int i=0; i<num_cols; ++i)
         {
            int ii = l2gmaps[ip][i];
            Prl.Set(ii,i,1.0);
         }
         Prl.Finalize();
         PatchMat00[ip] = RAP(Prl,diag,Prl);
      }
   }

   //--------------------------------------------------------------------------------------
   // Construction of (0,1) and its transpose
   //--------------------------------------------------------------------------------------
   // The matrix PatchMat10 is the same as PatchMat01 only for symmetric problems
   // For the case of FOSLS the offdiagonal matrices are not symmetric because of the essential BC
   // Therefore Patch10 has to be computed in the same way with (0,1) but using the traspose of A
   // loop through patches
   Array<SparseMatrix * > PatchMat01(nrpatch);
   Array<SparseMatrix * > PatchMat10(nrpatch);
   for (int ip = 0; ip < nrpatch; ++ip)
   {
      PatchMat01[ip] = nullptr;
      PatchMat10[ip] = nullptr;
      if (myid == host_rank[ip])
      {
         int num_rows = patch_tdof_info->patch_local_tdofs[ip].Size();
         int num_cols = patch_other_tdofs[ip].Size();
         if (num_rows*num_cols !=0)
         {
            PatchMat01[ip] = new SparseMatrix(num_rows, num_cols);
            GetOffdColumnValues(patch_tdof_info->patch_local_tdofs[ip],
                                patch_other_tdofs[ip],offd, cmap,row_start, PatchMat01[ip]);
            PatchMat01[ip]->Finalize();
            // Now the transpose
            SparseMatrix Mat(num_rows, num_cols);
            GetOffdColumnValues(patch_tdof_info->patch_local_tdofs[ip],
                                patch_other_tdofs[ip],offdT, cmapT,row_startT, &Mat);
            Mat.Finalize();
            PatchMat10[ip] = Transpose(Mat);
         }
      }
   }
   delete patch_tdof_info;
   delete At;

   //--------------------------------------------------------------------------------------
   // Construction of (1,1)
   //--------------------------------------------------------------------------------------
   // 1. Send info (sendbuff, sentcounts, send sdispls)
   // Each proccess computes and groups together an array of sendbuff. There will be
   // one sendbuff for all patches
   Array<int> send_count(num_procs);
   Array<int> send_displ(num_procs);
   Array<int> recv_count(num_procs);
   Array<int> recv_displ(num_procs);
   send_count = 0; send_displ = 0;
   recv_count = 0; recv_displ = 0;

   for (int ip = 0; ip < nrpatch; ip++)
   {
      // loop through the dofs and identify their ranks
      int sendnum_rows = patch_owned_other_tdofs[ip].Size();
      for (int i =0; i<sendnum_rows; i++)
      {
         int tdof = patch_owned_other_tdofs[ip][i];
         int tdof_rank = get_rank(tdof);
         if (myid == tdof_rank)
         {
            int k = GetNumColumns(tdof,patch_other_tdofs[ip],diag, offd, cmap, row_start);
            // pass one more that holds how many
            send_count[host_rank[ip]] += k+2;
         }
      }
   }

   // comunicate so that recv_count is constructed
   MPI_Alltoall(&send_count[0],1,MPI_INT,&recv_count[0],1,MPI_INT,comm);
   for (int k=0; k<num_procs-1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
   }
   int sbuff_size = send_count.Sum();
   int rbuff_size = recv_count.Sum();
   // now allocate space for the send buffer
   Array<double> sendbuf(sbuff_size);  sendbuf = 0;
   Array<int> sendmap(sbuff_size);  sendmap = 0;
   Array<int> soffs(num_procs); soffs = 0;

   // now the data will be placed according to process offsets
   for (int ip = 0; ip < nrpatch; ip++)
   {
      // loop through the dofs and identify their ranks
      int sendnum_rows = patch_owned_other_tdofs[ip].Size();
      for (int i = 0; i<sendnum_rows; i++)
      {
         int tdof = patch_owned_other_tdofs[ip][i];
         // find its rank
         int tdof_rank = get_rank(tdof);
         if (myid == tdof_rank)
         {
            Array<int>cols;
            Array<double>vals;
            GetColumnValues(tdof,patch_other_tdofs[ip],diag,offd, cmap,row_start, cols,
                            vals);
            int j = send_displ[host_rank[ip]] + soffs[host_rank[ip]];
            int size = cols.Size();
            // Pass one more to hold the size
            soffs[host_rank[ip]] += size+1;
            // need to save and communicate these offsets for extraction from recv_buff
            // // For now we do the copy (will be changed later)
            sendbuf[j] = 0.0;
            sendmap[j] = size;
            for (int k=0; k<size ; k++)
            {
               sendbuf[j+k+1] = vals[k];
               sendmap[j+k+1] = cols[k];
            }
         }
      }
   }

   // communication
   Array<double> recvbuf(rbuff_size);
   Array<int> recvmap(rbuff_size);

   double * sendbuf_ptr = nullptr;
   double * recvbuf_ptr = nullptr;
   int * sendmap_ptr = nullptr;
   int * recvmap_ptr = nullptr;
   if (sbuff_size !=0 )
   {
      sendbuf_ptr = &sendbuf[0];
      sendmap_ptr = &sendmap[0];
   }
   if (rbuff_size !=0 )
   {
      recvbuf_ptr = &recvbuf[0];
      recvmap_ptr = &recvmap[0];
   }

   MPI_Alltoallv(sendbuf_ptr, send_count, send_displ, MPI_DOUBLE, recvbuf_ptr,
                 recv_count, recv_displ, MPI_DOUBLE, comm);

   MPI_Alltoallv(sendmap_ptr, send_count, send_displ, MPI_INT, recvmap_ptr,
                 recv_count, recv_displ, MPI_INT, comm);

   Array<SparseMatrix * > PatchMat11(nrpatch);

   Array<int> roffs(num_procs);
   roffs = 0;
   // Now each process will construct the SparseMatrix
   for (int ip = 0; ip < nrpatch; ip++)
   {
      PatchMat11[ip] = nullptr;
      if (myid == host_rank[ip])
      {
         int ndof = patch_other_tdofs[ip].Size();
         PatchMat11[ip] = new SparseMatrix(ndof,ndof);
         // extract the data from receiv buffer
         // loop through rows
         for (int i=0; i<ndof; i++)
         {
            // pick up the dof and find its tdof_rank
            int tdof = patch_other_tdofs[ip][i];
            int tdof_rank= get_rank(tdof);
            // offset
            int k = recv_displ[tdof_rank] + roffs[tdof_rank];
            roffs[tdof_rank] += recvmap[k]+1;
            // copy to the matrix
            for (int j =0; j<recvmap[k]; j++)
            {
               int jj = recvmap[k+j+1];
               PatchMat11[ip]->Set(i,jj,recvbuf[k+j+1]);
            }
         }
         PatchMat11[ip]->Finalize();
      }
   }

   Array<BlockMatrix * > BlkPatchMat(nrpatch);
   PatchMat.SetSize(nrpatch);

   for (int ip = 0; ip < nrpatch; ip++)
   {
      //initialise to nullptr
      PatchMat[ip] = nullptr;
      if (myid == host_rank[ip])
      {
         if (PatchMat11[ip]->Height() !=0)
         {
            Array<int>block_offsets(3);
            block_offsets[0] = 0;
            block_offsets[1] = PatchMat00[ip]->Height();
            block_offsets[2] = PatchMat11[ip]->Height();
            block_offsets.PartialSum();
            BlkPatchMat[ip] = new BlockMatrix(block_offsets);
            BlkPatchMat[ip]->SetBlock(0,0,PatchMat00[ip]);
            BlkPatchMat[ip]->SetBlock(0,1,PatchMat01[ip]);
            BlkPatchMat[ip]->SetBlock(1,0,PatchMat10[ip]);
            BlkPatchMat[ip]->SetBlock(1,1,PatchMat11[ip]);
            // Convert to sparse
            PatchMat[ip] = BlkPatchMat[ip]->CreateMonolithic();
            delete BlkPatchMat[ip];
         }
         else
         {
            PatchMat[ip] = new SparseMatrix(*PatchMat00[ip]);
         }
         delete PatchMat00[ip];
         delete PatchMat10[ip];
         delete PatchMat01[ip];
         delete PatchMat11[ip];
         PatchMat[ip]->Threshold(1e-13);
      }
   }
   PatchMat00.DeleteAll();
   PatchMat01.DeleteAll();
   PatchMat10.DeleteAll();
   PatchMat11.DeleteAll();
}


PatchRestriction::PatchRestriction(PatchAssembly * P_) : P(P_)
{
   comm = P->comm;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);
   nrpatch = P->nrpatch;
   host_rank = P->host_rank;

   send_count.SetSize(num_procs);
   send_displ.SetSize(num_procs);
   recv_count.SetSize(num_procs);
   recv_displ.SetSize(num_procs);

   send_count = 0; send_displ = 0;
   recv_count = 0; recv_displ = 0;

   // Precompute send_counts
   for (int ip = 0; ip < nrpatch; ip++)
   {
      int sendnum_rows = P->patch_owned_other_tdofs[ip].Size();
      for (int i =0; i<sendnum_rows; i++)
      {
         int tdof = P->patch_owned_other_tdofs[ip][i];
         int tdof_rank = P->get_rank(tdof);
         if (myid == tdof_rank)
         {
            send_count[host_rank[ip]]++;
         }
      }
   }

   // comunicate so that recv_count is constructed
   MPI_Alltoall(&send_count[0],1,MPI_INT,&recv_count[0],1,MPI_INT,comm);
   //
   for (int k=0; k<num_procs-1; k++)
   {
      send_displ[k+1] = send_displ[k] + send_count[k];
      recv_displ[k+1] = recv_displ[k] + recv_count[k];
   }
   sbuff_size = send_count.Sum();
   rbuff_size = recv_count.Sum();
}

void PatchRestriction::Mult(const Vector & r, Array<BlockVector*> & res)
{
   int *row_start = P->A->GetRowStarts();
   std::vector<Vector> res0(nrpatch); // residual on the processor
   std::vector<Vector> res1(nrpatch); // residual off the processor
   //  Part of the residual on the processor
   for (int ip = 0; ip < nrpatch; ip++)
   {
      if (myid == host_rank[ip])
      {
         r.GetSubVector(P->l2gmaps[ip], res0[ip]);
      }
   }
   // now allocate space for the send buffer
   Array<double> sendbuf(sbuff_size);  sendbuf = 0;
   Array<int> soffs(num_procs); soffs = 0;
   // now the data will be placed according to process offsets
   for (int ip = 0; ip < nrpatch; ip++)
   {
      int sendnum_rows = P->patch_owned_other_tdofs[ip].Size();
      for (int i = 0; i<sendnum_rows; i++)
      {
         int tdof = P->patch_owned_other_tdofs[ip][i];
         // find its rank
         int tdof_rank = P->get_rank(tdof);
         if (myid == tdof_rank)
         {
            int j = send_displ[host_rank[ip]] + soffs[host_rank[ip]];
            soffs[host_rank[ip]]++;
            int k = tdof - row_start[0];
            sendbuf[j] = r[k];
         }
      }
   }
   // communication
   Array<double> recvbuf(rbuff_size);
   double * sendbuf_ptr = nullptr;
   double * recvbuf_ptr = nullptr;
   if (sbuff_size !=0 ) { sendbuf_ptr = &sendbuf[0]; }
   if (rbuff_size !=0 ) { recvbuf_ptr = &recvbuf[0]; }

   MPI_Alltoallv(sendbuf_ptr, send_count, send_displ, MPI_DOUBLE, recvbuf_ptr,
                 recv_count, recv_displ, MPI_DOUBLE, comm);
   Array<int> roffs(num_procs);
   roffs = 0;
   // Now each process will construct the res1 vector
   for (int ip = 0; ip < nrpatch; ip++)
   {
      if (myid == host_rank[ip])
      {
         int ndof = P->patch_other_tdofs[ip].Size();
         res1[ip].SetSize(ndof);
         // extract the data from receiv buffer
         // loop through rows
         for (int i=0; i<ndof; i++)
         {
            // pick up the dof and find its tdof_rank
            int tdof = P->patch_other_tdofs[ip][i];
            int tdof_rank= P->get_rank(tdof);
            // offset
            int k = recv_displ[tdof_rank] + roffs[tdof_rank];
            roffs[tdof_rank]++;
            res1[ip][i] = recvbuf[k];
         }
      }
   }

   res.SetSize(nrpatch);
   for (int ip=0; ip<nrpatch; ip++)
   {
      res[ip] = nullptr;
      if (myid == host_rank[ip])
      {
         Array<int> block_offs(3);
         block_offs[0] = 0;
         block_offs[1] = res0[ip].Size();
         block_offs[2] = res1[ip].Size();
         block_offs.PartialSum();
         res[ip] = new BlockVector(block_offs);
         res[ip]->SetVector(res0[ip], 0);
         res[ip]->SetVector(res1[ip], res0[ip].Size());
      }
   }
}

void PatchRestriction::MultTranspose(const Array<BlockVector *> & sol,
                                     Vector & z)
{
   int *row_start = P->A->GetRowStarts();
   std::vector<Vector> sol0(nrpatch);
   std::vector<Vector> sol1(nrpatch);
   // Step 3: Propagate the information to the global solution vector
   // (the recv_buff becomes the sendbuff and vice-versa)
   Array<double> sendbuf(sbuff_size);  sendbuf = 0.0;
   Array<double> recvbuf(rbuff_size);  recvbuf = 0.0;
   Array<int> roffs(num_procs); roffs = 0;
   Array<int> soffs(num_procs); soffs = 0;
   for (int ip = 0; ip < nrpatch; ip++)
   {
      if (myid == host_rank[ip])
      {
         sol1[ip] = sol[ip]->GetBlock(1);
         int ndof = P->patch_other_tdofs[ip].Size();
         // loop through rows
         for (int i=0; i<ndof; i++)
         {
            //  pick up the dof and find its tdof_rank
            int tdof = P->patch_other_tdofs[ip][i];
            int tdof_rank= P->get_rank(tdof);
            // offset
            int k = recv_displ[tdof_rank] + roffs[tdof_rank];
            roffs[tdof_rank]++;
            recvbuf[k] = sol1[ip][i];
         }
      }
   }
   // now communication
   double * sendbuf_ptr = nullptr;
   double * recvbuf_ptr = nullptr;
   if (sbuff_size !=0 ) { sendbuf_ptr = &sendbuf[0]; }
   if (rbuff_size !=0 ) { recvbuf_ptr = &recvbuf[0]; }

   MPI_Alltoallv(recvbuf_ptr, recv_count, recv_displ, MPI_DOUBLE, sendbuf_ptr,
                 send_count, send_displ, MPI_DOUBLE, comm);

   // 1. Accummulate for the solution to other prosessors
   for (int ip = 0; ip < nrpatch; ip++)
   {
      int sendnum_rows = P->patch_owned_other_tdofs[ip].Size();
      for (int i = 0; i<sendnum_rows; i++)
      {
         int tdof = P->patch_owned_other_tdofs[ip][i];
         // find its rank
         int tdof_rank = P->get_rank(tdof);
         if (myid == tdof_rank)
         {
            int j = send_displ[host_rank[ip]] + soffs[host_rank[ip]];
            soffs[host_rank[ip]]++;
            int k = tdof - row_start[0];
            z[k] += sendbuf[j];
         }
      }
   }
   // 2. Accummulate for the solution on the processor
   for (int ip = 0; ip < nrpatch; ip++)
   {
      if (myid == host_rank[ip])
      {
         sol0[ip] = sol[ip]->GetBlock(0);
         z.AddElementVector(P->l2gmaps[ip],sol0[ip]);
      }
   }
}

void PatchAssembly::compute_trueoffsets()
{
   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);
   tdof_offsets.resize(num_procs);
   int mytoffset = fespace->GetMyTDofOffset();
   MPI_Allgather(&mytoffset,1,MPI_INT,&tdof_offsets[0],1,MPI_INT,comm);
}

int PatchAssembly::get_rank(int tdof)
{
   int size = tdof_offsets.size();
   if (size == 1) {return 0;}
   std::vector<int>::iterator up;
   up=std::upper_bound (tdof_offsets.begin(), tdof_offsets.end(),
                        tdof); //          ^
   return std::distance(tdof_offsets.begin(),up)-1;
}

PatchAssembly::~PatchAssembly()
{
   // // if(patch_tdof_info) delete patch_tdof_info;
   for (int ip = 0; ip < nrpatch; ip++)
   {
      if (PatchMat[ip]) { delete PatchMat[ip]; }
   }
   PatchMat.DeleteAll();
}

void CheckSPD(SparseMatrix *S)
{
   MFEM_VERIFY(S->Height() == S->Width(), "");
   const int n = S->Height();

   DenseMatrix D(n);
   Vector ev(n);

   S->ToDenseMatrix(D);
   D.Eigenvalues(ev);

   const double minev = ev.Min();
   //cout << "Min eig " << minev << endl;

   MFEM_VERIFY(minev > 0.0, "");
}

SchwarzSmoother::SchwarzSmoother(ParMesh * cpmesh_, VectorCoefficient & BCoef_,
                                 int ref_levels_,
                                 ParFiniteElementSpace *fespace_,
                                 HypreParMatrix * A_)
   : Solver(A_->Height(), A_->Width()), A(A_)
{
   P = new PatchAssembly(cpmesh_, BCoef_, ref_levels_, fespace_, A_);

   comm = A->GetComm();
   nrpatch = P->nrpatch;
   host_rank.SetSize(nrpatch);
   host_rank = P->host_rank;
   PatchInv.SetSize(nrpatch);
   for (int ip = 0; ip < nrpatch; ip++)
   {
      PatchInv[ip] = nullptr;
      if (P->PatchMat[ip])
      {
#ifdef MFEM_USE_SUITESPARSE
         PatchInv[ip] = new UMFPackSolver;
         PatchInv[ip]->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_AMD;
         std::cout << "SchwarzSmoother using UMFPackSolver" << std::endl;

         PatchInv[ip]->SetOperator(*P->PatchMat[ip]);
#else
         PatchInv[ip] = new GMRESSolver;
         PatchInv[ip]->iterative_mode = false;
         std::cout << "SchwarzSmoother using GMRESSolver size " <<
                   P->PatchMat[ip]->Height() << " x " << P->PatchMat[ip]->Width() << " sym " <<
                   P->PatchMat[ip]->IsSymmetric() << ", global size " << A_->Height() << std::endl;
         PatchInv[ip]->SetRelTol(1e-12);
         PatchInv[ip]->SetMaxIter(100);

         HypreParMatrix *hmat = nullptr;
         {
            // Set preconditioner based on hypre
            HYPRE_BigInt *row_starts = new HYPRE_BigInt(2);
            HYPRE_BigInt *col_starts = new HYPRE_BigInt(2);
            row_starts[0] = 0;
            row_starts[1] = P->PatchMat[ip]->NumRows();

            col_starts[0] = 0;
            col_starts[1] = P->PatchMat[ip]->NumCols();

            //hmat = new HypreParMatrix(MPI_COMM_WORLD, row_starts, col_starts, P->PatchMat[ip]);
            hmat = new HypreParMatrix(MPI_COMM_WORLD, P->PatchMat[ip]->NumRows(),
                                      row_starts, P->PatchMat[ip]);

            HypreILU *hilu = new HypreILU();
            PatchInv[ip]->SetPreconditioner(*hilu);
         }

         PatchInv[ip]->SetOperator(*hmat);

#endif
         //CheckSPD(P->PatchMat[ip]);
         //MFEM_VERIFY(P->PatchMat[ip]->IsSymmetric() < 1.0e-12, "");
         //PatchInv[ip]->SetPrintLevel(1);
      }
   }
   R = new PatchRestriction(P);
}

void SchwarzSmoother::Mult(const Vector &r, Vector &z) const
{
   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);
   Array<int> tdof_i;

   z = 0.0;
   Vector rnew(r);
   Vector znew(z);

   for (int iter = 0; iter < maxit; iter++)
   {
      znew = 0.0;
      Array<BlockVector * > res;
      R->Mult(rnew,res);

      Array<BlockVector*> sol(nrpatch);
      for (int ip=0; ip<nrpatch; ip++)
      {
         sol[ip] = nullptr;
         if (myid == host_rank[ip])
         {
            Array<int> block_offs(3);
            block_offs[0] = 0;
            block_offs[1] = res[ip]->GetBlock(0).Size();
            block_offs[2] = res[ip]->GetBlock(1).Size();
            block_offs.PartialSum();
            sol[ip] = new BlockVector(block_offs);
            PatchInv[ip]->Mult(*res[ip], *sol[ip]);
         }
      }
      for (auto p:res) { delete p; }
      res.DeleteAll();
      R->MultTranspose(sol,znew);
      for (auto p:sol) { delete p; }
      sol.DeleteAll();

      znew *= theta; // relaxation parameter
      z+= znew;
      // Update residual
      Vector raux(znew.Size());
      A->Mult(znew,raux);
      rnew -= raux;
   } // end of loop through smoother iterations
}

SchwarzSmoother::~SchwarzSmoother()
{
   delete P;
   delete R;
   for (int ip = 0; ip < nrpatch; ip++)
   {
      if (PatchInv[ip]) { delete PatchInv[ip]; }
   }
   PatchInv.DeleteAll();
}

