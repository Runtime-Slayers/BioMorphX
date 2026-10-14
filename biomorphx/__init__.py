"""
BioMorphX (v1.0.0) — Unified Cross-Scale Mechano-Genomic, Biomaterial, Nanofabrication
& Life-Critical Biomedical Engineering Python SDK.
"""
from types import SimpleNamespace
from .core import (
    __version__,
    __author__,
    ClinicalBoundsError,
    LifeCriticalSafetyWarning,
    SafetyCertificate,
    verify_bounds,
    conformal_safety_interval,
    AA_MECH_TABLE,
    SANTALUCIA_NN,
    HUMAN_CODON_TABLE,
    NATIVE_TISSUE_ATLAS,
    PKPD_DRUG_ATLAS,
    SNIPPET_REGISTRY,
    analyze_mechano_protein,
    design_crispr_mechano_guide,
    optimize_mrna_construct,
    design_chrono_hydrogel,
    simulate_scaffold_degradation,
    generate_auxetic_gcode_and_stl,
    simulate_lnp_nanocarrier,
    mbsp_filter_and_quantify,
    audit_icu_hemodynamics_ventilation,
    optimize_renal_pkpd_dosing,
    list_snippets,
    get_snippet,
    run_snippet,
    self_test,
)

safety = SimpleNamespace(
    ClinicalBoundsError=ClinicalBoundsError,
    LifeCriticalSafetyWarning=LifeCriticalSafetyWarning,
    SafetyCertificate=SafetyCertificate,
    verify_bounds=verify_bounds,
    conformal_safety_interval=conformal_safety_interval,
)

seqmech = SimpleNamespace(
    AA_MECH_TABLE=AA_MECH_TABLE,
    SANTALUCIA_NN=SANTALUCIA_NN,
    HUMAN_CODON_TABLE=HUMAN_CODON_TABLE,
    analyze_mechano_protein=analyze_mechano_protein,
    design_crispr_mechano_guide=design_crispr_mechano_guide,
    optimize_mrna_construct=optimize_mrna_construct,
)

biomaterials = SimpleNamespace(
    NATIVE_TISSUE_ATLAS=NATIVE_TISSUE_ATLAS,
    design_chrono_hydrogel=design_chrono_hydrogel,
    simulate_scaffold_degradation=simulate_scaffold_degradation,
)

nanofab = SimpleNamespace(
    generate_auxetic_gcode_and_stl=generate_auxetic_gcode_and_stl,
    simulate_lnp_nanocarrier=simulate_lnp_nanocarrier,
)

imaging = SimpleNamespace(
    mbsp_filter_and_quantify=mbsp_filter_and_quantify,
)

clinical = SimpleNamespace(
    PKPD_DRUG_ATLAS=PKPD_DRUG_ATLAS,
    audit_icu_hemodynamics_ventilation=audit_icu_hemodynamics_ventilation,
    optimize_renal_pkpd_dosing=optimize_renal_pkpd_dosing,
)

snippets = SimpleNamespace(
    SNIPPET_REGISTRY=SNIPPET_REGISTRY,
    list_snippets=list_snippets,
    get_snippet=get_snippet,
    run_snippet=run_snippet,
)
