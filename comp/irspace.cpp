#include "irspace.hpp"
#include <diffop_impl.hpp>


namespace ngcomp
{

  class IRFiniteElement : public FiniteElement
  {
    ELEMENT_TYPE et;
    IntegrationRule ir;
  public:
    IRFiniteElement (ELEMENT_TYPE aet, int aorder)
      : FiniteElement(0, aorder), et(aet), ir(aet, 2*aorder)
    {
      ndof = ir.Size();
    }
    ELEMENT_TYPE ElementType() const override { return et; }
    
    void Interpolate (const ElementTransformation & trafo, 
                      const CoefficientFunction & func, SliceMatrix<> coefs, LocalHeap & lh) const override
    {
      HeapReset hr(lh);
      BaseMappedIntegrationRule & mir = trafo(ir, lh);
      func.Evaluate (mir, coefs);
    }

  };


  class IRDiffOp : public DiffOp<IRDiffOp>
  {
  public:
    enum { DIM = 1 };
    enum { DIM_SPACE = 0 };
    enum { DIM_ELEMENT = 0 };
    enum { DIM_DMAT = 1 };
    enum { DIFFORDER = 0 };

    static IVec<0> GetDimensions() { return IVec<0>(); };
    static string Name() { return "IdIR"; }
    static bool SupportsVB (VorB checkvb) { return true; }

    template <typename MIP, typename MAT>
    static void GenerateMatrix (const FiniteElement & fel, const MIP & mip,
				MAT && mat, LocalHeap & lh)
    {
      mat.Row(0).Range(0, fel.GetNDof()) = 0;
      mat(0, mip.IP().Nr()) = 1;
    }

    using DiffOp<IRDiffOp>::ApplyIR;
    template <class MIR, class TMY>
    static void ApplyIR (const FiniteElement & fel, const MIR & mir,
                         BareSliceVector<double> x, TMY y,
			 LocalHeap & lh)
    {
      y.Col(0).Range(0, fel.GetNDof()) = x;
    }

    using DiffOp<IRDiffOp>::ApplyTransIR;    
    template <class MIR>
    static void ApplyTransIR (const FiniteElement & fel, 
			      const MIR & mir,
			      FlatMatrix<double> x, BareSliceVector<double> y,
			      LocalHeap & lh)
    {
      y.Range(0, fel.GetNDof()) = x.Col(0);
    }
    
    
    // SIMD rules may be Ranges of the element rule (SymbolicBFI evaluates big rules in blocks),
    // point i of the Range is dof mir.IR().FirstNr()+i
    template <typename FEL, typename MIR>
    static void GenerateMatrixSIMDIR (const FEL & fel, const MIR & mir, BareSliceMatrix<SIMD<double>> mat)
    {
      constexpr int SW = SIMD<double>::Size();
      int ndof = fel.GetNDof();
      int first = mir.IR().FirstNr();
      mat.AddSize(ndof, mir.Size()) = SIMD<double> (0.0);
      SliceMatrix<double> hmat(ndof, SW*mir.Size(), SW*mat.Dist(), (double*)mat.Data());
      for (int i = 0; i < SW*mir.Size() && first+i < ndof; i++)
        hmat(first+i, i) = 1;
      // cout << "diffopir, genmat:" << endl << mat.AddSize(ndof, mir.Size()) << endl;
    }
    
    using DiffOp<IRDiffOp> :: ApplySIMDIR;
    template <typename FEL, class MIR, class TVX>
    static void ApplySIMDIR (const FEL & fel, const MIR & mir,
                             const TVX & x, BareSliceMatrix<SIMD<double>> y)
    {
      constexpr int SW = SIMD<double>::Size();
      int ndof = fel.GetNDof();
      int first = mir.IR().FirstNr();
      // padding points of the last SIMD point repeat the last dof
      for (size_t i = 0; i < mir.Size(); i++)
        y(0, i) = SIMD<double> ([&] (int j) { return x(min(first+int(i)*SW+j, ndof-1)); });
    }

    using DiffOp<IRDiffOp> :: AddTransSIMDIR;    
    template <typename FEL, class MIR, class TVY>
    static void AddTransSIMDIR (const FEL & fel, const MIR & mir,
                                BareSliceMatrix<SIMD<double>> x, TVY & y)
    // LocalHeap & lh)
    {
      constexpr int SW = SIMD<double>::Size();
      int ndof = fel.GetNDof();
      int first = mir.IR().FirstNr();
      for (size_t i = 0; i < mir.Size(); i++)
        for (int j = 0; j < SW && first+int(i)*SW+j < ndof; j++)
          y(first+i*SW+j) = x(0,i)[j];
    }
    
    
  };





  IntegrationRuleSpace::IntegrationRuleSpace (shared_ptr<MeshAccess> ama, const Flags & flags, bool checkflags)
      : FESpace (ama, flags)
    {
      type = "irspace";
      evaluator[VOL] = make_shared<T_DifferentialOperator<IRDiffOp>>();
      
      if (dimension > 1)
        evaluator[VOL] = make_shared<BlockDifferentialOperator> (evaluator[VOL], dimension);
    }

  void IntegrationRuleSpace::Update()
  {
    //cout << "update irspace" << endl;
    firsteldof.SetSize(ma->GetNE()+1);
    size_t ndof = 0;
    for (auto i : Range(ma->GetNE()))
      {
        firsteldof[i] = ndof;
        if (!DefinedOn(ElementId(VOL, i)))
            continue;

        IntegrationRule ir(ma->GetElType( { VOL, i } ), 2*order);
        ndof += ir.Size();
      }
    firsteldof.Last() = ndof;
    //cout << "firstel = " << firsteldof << endl;
    SetNDof(ndof);

    UpdateCouplingDofArray();
  }

  void IntegrationRuleSpace::UpdateCouplingDofArray()
  {
    // all dofs are local dofs
    ctofdof.SetSize(ndof);
    ctofdof = LOCAL_DOF;
  }
  
  
  FiniteElement & IntegrationRuleSpace::GetFE (ElementId ei, Allocator & lh) const
  {
    if (ei.VB() == VOL && DefinedOn(ei))
      return *new (lh) IRFiniteElement(ma->GetElType(ei), order);
    else
      return SwitchET (ma->GetElType(ei), [&] (auto et) -> FiniteElement&
                       {
                         return *new (lh) DummyFE<et.ElementType()> ();
                       });
  }
  
  void IntegrationRuleSpace::GetDofNrs (ElementId ei, Array<int> & dnums) const
  {
    if (ei.VB() == VOL)
      dnums = IntRange(firsteldof[ei.Nr()], firsteldof[ei.Nr()+1]);
    else
      dnums.SetSize0();
  }

  std::map<ELEMENT_TYPE, IntegrationRule> IntegrationRuleSpace::GetIntegrationRules() const
  {
    std::map<ELEMENT_TYPE, IntegrationRule> irs;
    irs[ET_TRIG] = IntegrationRule(ET_TRIG, 2*order);
    irs[ET_QUAD] = IntegrationRule(ET_QUAD, 2*order);
    irs[ET_HEX] = IntegrationRule(ET_HEX, 2*order);
    irs[ET_TET] = IntegrationRule(ET_TET, 2*order);
    irs[ET_PRISM] = IntegrationRule(ET_PRISM, 2*order);
    irs[ET_PYRAMID] = IntegrationRule(ET_PYRAMID, 2*order);
    return irs;
  }




    IntegrationRuleSpaceSurface::IntegrationRuleSpaceSurface (shared_ptr<MeshAccess> ama, const Flags & flags, bool checkflags)
      : FESpace (ama, flags)
    {
      type = "irspacesurface";
      evaluator[VOL] = make_shared<T_DifferentialOperator<IRDiffOp>>();
      evaluator[BND] = make_shared<T_DifferentialOperator<IRDiffOp>>();
      
      if (dimension > 1)
        {
          evaluator[VOL] = make_shared<BlockDifferentialOperator> (evaluator[VOL], dimension);
          evaluator[BND] = make_shared<BlockDifferentialOperator> (evaluator[BND], dimension);
        }
    }

  void IntegrationRuleSpaceSurface::Update()
  {
    //cout << "update irspace" << endl;
    firsteldof.SetSize(ma->GetNSE()+1);
    size_t ndof = 0;
    for (auto i : Range(ma->GetNSE()))
      {
        firsteldof[i] = ndof;
        if (!DefinedOn(ElementId(BND, i)))
            continue;

        IntegrationRule ir(ma->GetElType( { BND, i } ), 2*order);
        ndof += ir.Size();
      }
    firsteldof.Last() = ndof;
    //cout << "firstel = " << firsteldof << endl;
    SetNDof(ndof);

    UpdateCouplingDofArray();
  }

  void IntegrationRuleSpaceSurface::UpdateCouplingDofArray()
  {
    // all dofs are local dofs
    ctofdof.SetSize(ndof);
    ctofdof = LOCAL_DOF;
  }
  
  
  FiniteElement & IntegrationRuleSpaceSurface::GetFE (ElementId ei, Allocator & lh) const
  {
    if (ei.VB() == BND && DefinedOn(ei))
      return *new (lh) IRFiniteElement(ma->GetElType(ei), order);
    else
      return SwitchET (ma->GetElType(ei), [&] (auto et) -> FiniteElement&
                       {
                         return *new (lh) DummyFE<et.ElementType()> ();
                       });
  }
  
  void IntegrationRuleSpaceSurface::GetDofNrs (ElementId ei, Array<int> & dnums) const
  {
    if (ei.VB() == BND)
      dnums = IntRange(firsteldof[ei.Nr()], firsteldof[ei.Nr()+1]);
    else
      dnums.SetSize0();
  }

  std::map<ELEMENT_TYPE, IntegrationRule> IntegrationRuleSpaceSurface::GetIntegrationRules() const
  {
    std::map<ELEMENT_TYPE, IntegrationRule> irs;
    irs[ET_SEGM] = IntegrationRule(ET_SEGM, 2*order);
    irs[ET_TRIG] = IntegrationRule(ET_TRIG, 2*order);
    irs[ET_QUAD] = IntegrationRule(ET_QUAD, 2*order);
    return irs;
  }

  
  static RegisterFESpace<IntegrationRuleSpace> init ("irspace");
  static RegisterFESpace<IntegrationRuleSpaceSurface> initsurf ("irspacesurface");
}
