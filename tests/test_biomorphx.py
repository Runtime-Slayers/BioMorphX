#!/usr/bin/env python3
"""
BioMorphX (v1.0.0) — 15-Test Life-Critical Verification Suite (40+ Assertions)
Validates all submodules, exports real 3D bioprinting G-Code and STL meshes,
and generates the 6-panel publication-grade visual showcase figure.
"""

import importlib
import json
import pathlib
import shutil
import site
import subprocess
import sys
import unittest
import warnings
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import biomorphx as bmx

# 4. Run Comprehensive Unit & Life-Critical Safety Test Suite (15 Tests, 40+ Assertions)
class TestBioMorphXSuite(unittest.TestCase):
    def test_01_version_and_namespaces(self):
        self.assertEqual(bmx.__version__, "1.0.0")
        for sub in ("safety", "seqmech", "biomaterials", "nanofab", "imaging", "clinical", "snippets"):
            self.assertTrue(hasattr(bmx, sub))

    def test_02_safety_bounds_verifier(self):
        self.assertAlmostEqual(bmx.verify_bounds("pH", 7.4, 6.8, 7.8), 7.4)
        with self.assertRaises(bmx.ClinicalBoundsError):
            bmx.verify_bounds("pH", 8.5, 6.8, 7.8, "pH")
        with self.assertRaises(bmx.ClinicalBoundsError):
            bmx.verify_bounds("val", float("nan"), 0.0, 10.0)

    def test_03_conformal_safety_interval(self):
        res = bmx.conformal_safety_interval([10.0, 20.0], [0.2, 0.5, 0.8, 0.3, 0.9, 0.4, 0.1, 0.6, 0.7], alpha=0.10)
        self.assertGreaterEqual(res["guaranteed_coverage_pct"], 90.0)
        self.assertEqual(len(res["lower_bound"]), 2)
        self.assertTrue(np.all(res["upper_bound"] > res["lower_bound"]))

    def test_04_mechano_protein_analysis(self):
        res = bmx.analyze_mechano_protein("TVYAVTGRGDSPASSKPISINYRTEIDKPSQ", pH=7.4, applied_force_pN=12.0)
        self.assertIn("RGD_Integrin_Ligand", res["bioactive_motifs_detected"])
        self.assertGreater(res["molecular_weight_Da"], 3000.0)
        self.assertTrue(4.0 <= res["isoelectric_point_pI"] <= 11.0)
        self.assertGreater(res["unfolding_force_pN"], 15.0)
        self.assertEqual(res["safety_certificate"]["risk_tier"], "SAFE_GREEN")

    def test_05_mechano_protein_invalid_sequence(self):
        with self.assertRaises(bmx.ClinicalBoundsError):
            bmx.analyze_mechano_protein("AXZ123")

    def test_06_crispr_mechano_guide(self):
        dna = "ATGGCCCTGTGGATGCGCCTCCTGCCCCTGCTGGCGCTGCTGGCCCTCTGGGGACCTGACCCAGCCGCAGCCTTTGTGAACCAACACCTGTGCGGCTCACACCTGGTGGAAGCTCTCTACCTAGTGTGCGGGGAACGAGGCTTCTTCTACACACCCAAGACCCGCCGGGAGGCAGAGGACCTGCAGGTGGGGCAGGTGGAGCTGGGCGGGGGCCCTGGTGCAGGCAGCCTGCAGCCCTTGGCCCTGGAGGGGTCCCTGCAGAAGCGTGGCATTGTGGAACAATGCTGTACCAGCATCTGCTCCCTCTACCAGCTGGAGAACTACTGCAACTAG"
        res = bmx.design_crispr_mechano_guide(dna, supercoil_sigma=-0.06, top_k=5)
        self.assertGreater(res["total_pam_sites_found"], 10)
        self.assertEqual(len(res["top_guides"]), 5)
        self.assertGreater(res["top_guides"][0]["mechano_crispr_score"], 60.0)
        self.assertLess(res["top_guides"][0]["dG37_kcal_mol"], -15.0)

    def test_07_mrna_construct_optimizer(self):
        res = bmx.optimize_mrna_construct("MKTIIALSYIFCLVFADYKDDDDK", target_gc=0.58)
        self.assertGreater(res["codon_adaptation_index_CAI"], 0.85)
        self.assertNotIn("T", res["orf_mrna_sequence"])
        self.assertTrue(0.45 <= res["orf_gc_fraction"] <= 0.72)

    def test_08_chrono_hydrogel_all_12_tissues(self):
        for tissue in bmx.biomaterials.NATIVE_TISSUE_ATLAS:
            res = bmx.design_chrono_hydrogel(target_tissue=tissue)
            self.assertGreater(res["visco_harmonic_phase_dichroism_deg"], 10.0)
            self.assertLess(res["foreign_body_fibrosis_risk_pct"], 25.0)
            self.assertGreater(res["flory_rehner_mesh_size_nm"], 2.0)

    def test_09_scaffold_degradation_and_pH_alert(self):
        res_safe = bmx.simulate_scaffold_degradation("PLGA_50_50", Mw0_kDa=65.0, porosity=0.85, duration_days=30)
        self.assertEqual(res_safe["safety_certificate"]["risk_tier"], "SAFE_GREEN")
        res_acidic = bmx.simulate_scaffold_degradation("PLGA_50_50", Mw0_kDa=65.0, porosity=0.15, duration_days=30)
        self.assertIn(res_acidic["safety_certificate"]["risk_tier"], ("CAUTION_AMBER", "CRITICAL_RED"))

    def test_10_nanofab_gcode_stl_and_shear_guardrail(self):
        gcode_p = OUTPUT_DIR / "auxetic_myocardial_scaffold.gcode"
        stl_p = OUTPUT_DIR / "auxetic_myocardial_scaffold.stl"
        res = bmx.generate_auxetic_gcode_and_stl(
            width_mm=8.0, height_mm=8.0, layers=4, nozzle_diameter_um=250.0,
            print_speed_mm_s=6.0, out_gcode_path=gcode_p, out_stl_path=stl_p
        )
        self.assertTrue(gcode_p.exists() and gcode_p.stat().st_size > 1000)
        self.assertTrue(stl_p.exists() and stl_p.stat().st_size > 1000)
        self.assertLess(res["geometric_auxetic_poisson_ratio"], -0.15)
        self.assertGreater(res["predicted_cell_viability_pct"], 85.0)

        with warnings.catch_warnings(record=True) as w_rec:
            warnings.simplefilter("always")
            res_high_shear = bmx.generate_auxetic_gcode_and_stl(
                nozzle_diameter_um=60.0, print_speed_mm_s=45.0, bioink_K_Pa_sn=35.0
            )
            self.assertGreater(res_high_shear["wall_shear_stress_Pa"], 140.0)
            self.assertTrue(any(issubclass(w.category, bmx.LifeCriticalSafetyWarning) for w in w_rec))

    def test_11_lnp_nanocarrier_simulation(self):
        res = bmx.simulate_lnp_nanocarrier(peg_lipid_mol_pct=1.5, total_flow_rate_ml_min=14.0)
        self.assertTrue(60.0 <= res["hydrodynamic_diameter_nm"] <= 110.0)
        self.assertGreater(res["mrna_encapsulation_efficiency_pct"], 90.0)
        with self.assertRaises(bmx.ClinicalBoundsError):
            bmx.simulate_lnp_nanocarrier(ionizable_lipid_mol_pct=80.0, cholesterol_mol_pct=38.5)

    def test_12_mbsp_imaging_and_morphometry(self):
        yy, xx = np.mgrid[0:64, 0:64].astype(float)
        img = np.exp(-((yy - 32.0) ** 2) / 3.0) + np.exp(-((xx - 32.0) ** 2) / 3.0)
        img = np.clip(img + np.random.default_rng(7).normal(0, 0.12, (64, 64)), 0.0, 1.0)
        res = bmx.mbsp_filter_and_quantify(img, pixel_size_um=1.0)
        self.assertEqual(res["denoised_image"].shape, (64, 64))
        self.assertGreaterEqual(res["morphometry"]["branching_junction_count"], 1)

    def test_13_icu_hemodynamics_and_ventilation_guardrails(self):
        res_ok = bmx.audit_icu_hemodynamics_ventilation(
            hr_bpm=80, sbp_mmHg=120, dbp_mmHg=75, plateau_pressure_cmH2O=18, peep_cmH2O=6, tidal_volume_ml=420
        )
        self.assertEqual(res_ok["safety_certificate"]["risk_tier"], "SAFE_GREEN")

        with warnings.catch_warnings(record=True) as w_rec:
            warnings.simplefilter("always")
            res_crit = bmx.audit_icu_hemodynamics_ventilation(
                hr_bpm=135, sbp_mmHg=78, dbp_mmHg=42, plateau_pressure_cmH2O=32,
                peak_pressure_cmH2O=38, peep_cmH2O=8, tidal_volume_ml=650, lactate_mmol_L=4.8
            )
            self.assertEqual(res_crit["safety_certificate"]["risk_tier"], "CRITICAL_RED")
            self.assertTrue(any(issubclass(w.category, bmx.LifeCriticalSafetyWarning) for w in w_rec))

        with self.assertRaises(bmx.ClinicalBoundsError):
            bmx.audit_icu_hemodynamics_ventilation(sbp_mmHg=80, dbp_mmHg=95)

    def test_14_renal_pkpd_dosing_all_drugs(self):
        for drug in bmx.clinical.PKPD_DRUG_ATLAS:
            res = bmx.optimize_renal_pkpd_dosing(drug_name=drug, age_yr=65, weight_kg=70, serum_creatinine_mg_dL=1.2)
            self.assertGreater(res["recommended_maintenance_dose_mg"], 0.0)
            self.assertGreater(res["steady_state_peak_Cmax_mg_L"], res["steady_state_trough_Cmin_mg_L"])

    def test_15_all_12_snippets_and_self_test(self):
        st = bmx.self_test()
        self.assertTrue(st["all_passed"])
        self.assertEqual(st["total_pipelines_tested"], 12)


def generate_showcase(output_dir: pathlib.Path):
    """Generates the 6-panel publication-grade visual showcase dashboard."""
    print("Generating 6-panel BioMorphX SDK visual showcase figure...")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.0,
        "axes.titlesize": 9.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.8,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(13.0, 7.6))
    gs = gridspec.GridSpec(2, 3, wspace=0.32, hspace=0.38)

    # Panel A: Module 2 (`seqmech`) Single-Molecule WLC Force-Extension & Unfolding Curve
    ax1 = fig.add_subplot(gs[0, 0])
    seqs = [
        ("Elastin Spring (VPGVG)6", "VPGVGVPGVGVPGVGVPGVGVPGVGVPGVG", "#2563EB"),
        ("Fibronectin RGD Loop", "TVYAVTGRGDSPASSKPISINYRTEIDKPSQ", "#0D9488"),
        ("Rigid Proline-Rich Clamp", "PPGPAGPPGPPGPAGPPGPPGPAGPPGPPG", "#DC2626"),
    ]
    forces_grid = np.linspace(0.5, 45.0, 60)
    for label, s_aa, col in seqs:
        exts = [bmx.analyze_mechano_protein(s_aa, applied_force_pN=float(f))["wlc_extension_at_force_nm"] for f in forces_grid]
        f_star = bmx.analyze_mechano_protein(s_aa)["unfolding_force_pN"]
        ax1.plot(exts, forces_grid, lw=2.2, color=col, label=f"{label} ($F^*={f_star:.1f}$ pN)")
    ax1.set_xlabel("WLC End-to-End Extension $x(F)$ (nm)")
    ax1.set_ylabel("Applied Tensile Force $F$ (pN)")
    ax1.set_title("A. [seqmech] WLC Force Spectroscopy", loc="left", pad=6)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", fontsize=7.0)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel B: Module 3 (`biomaterials`) 12-Tissue Atlas Fibrosis Suppression
    ax2 = fig.add_subplot(gs[0, 1])
    t_List = ["brain_cortex", "vocal_fold", "myocardium", "cornea", "articular_cartilage", "cortical_bone_osteoid"]
    t_labels = ["Brain", "Vocal\nFold", "Myo-\ncardium", "Cornea", "Carti-\nlage", "Osteoid"]
    fib_pos = [bmx.design_chrono_hydrogel(t, auxetic_poisson=+0.45)["foreign_body_fibrosis_risk_pct"] for t in t_List]
    fib_aux = [bmx.design_chrono_hydrogel(t)["foreign_body_fibrosis_risk_pct"] for t in t_List]
    x_t = np.arange(len(t_List))
    w_b = 0.36
    ax2.bar(x_t - w_b/2, fib_pos, width=w_b, color="#DC2626", alpha=0.85, label="Standard Hydrogel ($\\nu=+0.45$)")
    ax2.bar(x_t + w_b/2, fib_aux, width=w_b, color="#0D9488", alpha=0.90, label="BioMorphX CHAM ($\\nu_{\\mathrm{aux}}<0$)")
    ax2.set_xticks(x_t)
    ax2.set_xticklabels(t_labels, fontsize=7.4)
    ax2.set_ylabel("Foreign-Body Fibrosis Risk (%)")
    ax2.set_title("B. [biomaterials] 12-Tissue Atlas", loc="left", pad=6)
    ax2.set_ylim(0, 110)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", fontsize=7.0)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Panel C: Module 3 (`biomaterials`) Scaffold Degradation, Stiffness Retention & Drug Release
    ax3 = fig.add_subplot(gs[0, 2])
    deg_res = bmx.simulate_scaffold_degradation("PLGA_50_50", Mw0_kDa=65.0, porosity=0.82, duration_days=60)
    days_arr = deg_res["timeseries"]["days"]
    ax3.plot(days_arr, np.array(deg_res["timeseries"]["modulus_retention_fraction"]) * 100.0,
             lw=2.2, color="#2563EB", label="Modulus Retention $E(t)/E_0$ (%)")
    ax3.plot(days_arr, deg_res["timeseries"]["cumulative_drug_release_pct"],
             lw=2.2, color="#0D9488", ls="--", label="Cumulative Drug Release (%)")
    ax3.set_xlabel("Implantation Time $t$ (days)")
    ax3.set_ylabel("Normalized Percentage (%)")
    ax3.set_title(f"C. [biomaterials] Degradation (Min pH={deg_res['minimum_local_pore_pH']:.2f})", loc="left", pad=6)
    ax3.legend(loc="center right", frameon=True, facecolor="white", fontsize=7.2)
    ax3.grid(True, linestyle=":", alpha=0.5)

    # Panel D: Module 4 (`nanofab`) 3D Bioprinting Nozzle Shear-Stress vs Live-Cell Viability
    ax4 = fig.add_subplot(gs[1, 0])
    nozzles = np.linspace(80, 450, 40)
    for speed, col in [(5.0, "#0D9488"), (12.0, "#D97706"), (25.0, "#DC2626")]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            viab = [
                bmx.generate_auxetic_gcode_and_stl(nozzle_diameter_um=float(nz), print_speed_mm_s=speed, layers=1)["predicted_cell_viability_pct"]
                for nz in nozzles
            ]
        ax4.plot(nozzles, viab, lw=2.2, color=col, label=f"Print Speed $v = {speed:.0f}$ mm/s")
    ax4.axhline(85.0, color="#0F172A", ls=":", lw=1.2, label="Clinical Viability Floor (85%)")
    ax4.set_xlabel("Bioprinter Nozzle Diameter ($\\mu$m)")
    ax4.set_ylabel("Predicted Live-Cell Viability (%)")
    ax4.set_title("D. [nanofab] Herschel-Bulkley Viability", loc="left", pad=6)
    ax4.set_ylim(20, 103)
    ax4.legend(loc="lower right", frameon=True, facecolor="white", fontsize=7.0)
    ax4.grid(True, linestyle=":", alpha=0.5)

    # Panel E: Module 5 (`imaging`) MBSP Filter Nematic & Topological Extraction
    ax5 = fig.add_subplot(gs[1, 1])
    yy, xx = np.mgrid[0:80, 0:80].astype(float)
    net_clean = np.zeros((80, 80))
    for k_line in range(12, 70, 16):
        net_clean[np.abs(yy - k_line - 3.0 * np.sin(xx / 8.0)) < 1.3] = 1.0
        net_clean[np.abs(xx - k_line - 3.0 * np.cos(yy / 8.0)) < 1.3] = 1.0
    rng_vis = np.random.default_rng(2026)
    net_noisy = np.clip(0.15 + 0.70 * net_clean + rng_vis.normal(0, 0.22, (80, 80)), 0.0, 1.0)
    mbsp_out = bmx.mbsp_filter_and_quantify(net_noisy, pixel_size_um=1.0)
    im5 = ax5.imshow(mbsp_out["structure_map"], cmap="viridis")
    ax5.set_title(
        f"E. [imaging] MBSP Map ($S={mbsp_out['morphometry']['nematic_alignment_order_S']:.2f}$, "
        f"Nodes={mbsp_out['morphometry']['branching_junction_count']})",
        loc="left", pad=6, fontsize=8.8
    )
    ax5.set_xticks([]); ax5.set_yticks([])
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.03)

    # Panel F: Module 6 (`clinical`) Renal-Adjusted Vancomycin PK/PD Across Renal Function (CrCl)
    ax6 = fig.add_subplot(gs[1, 2])
    scr_vals = np.linspace(0.6, 3.5, 30)
    cmax_list, cmin_list, crcl_list = [], [], []
    for scr in scr_vals:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_pk = bmx.optimize_renal_pkpd_dosing("vancomycin", age_yr=65, weight_kg=74, serum_creatinine_mg_dL=float(scr))
        crcl_list.append(r_pk["creatinine_clearance_ml_min"])
        cmax_list.append(r_pk["steady_state_peak_Cmax_mg_L"])
        cmin_list.append(r_pk["steady_state_trough_Cmin_mg_L"])

    ax6.axhspan(10.0, 20.0, color="#D1FAE5", alpha=0.55, label="Safe Therapeutic Trough (10-20 mg/L)")
    ax6.axhline(20.0, color="#DC2626", ls="--", lw=1.2, label="Nephrotoxicity Limit (20 mg/L)")
    ax6.plot(crcl_list, cmax_list, lw=2.0, color="#2563EB", label="Steady-State Peak $C_{\\max,\\mathrm{ss}}$")
    ax6.plot(crcl_list, cmin_list, lw=2.2, color="#0D9488", label="Steady-State Trough $C_{\\min,\\mathrm{ss}}$")
    ax6.set_xlabel("Cockcroft-Gault CrCl (mL/min)")
    ax6.set_ylabel("Serum Concentration (mg/L)")
    ax6.set_title("F. [clinical] Renal PK/PD Safety Corridor", loc="left", pad=6)
    ax6.set_ylim(0, 48)
    ax6.legend(loc="upper right", frameon=True, facecolor="white", fontsize=6.8)
    ax6.grid(True, linestyle=":", alpha=0.5)

    out_pdf = output_dir / "biomorphx_sdk_showcase.pdf"
    out_png = output_dir / "biomorphx_sdk_showcase.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Exported showcase figure: {out_png}")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestBioMorphXSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)
    assert test_result.wasSuccessful(), "Unit test suite failed!"

    if HAS_MPL:
        generate_showcase(OUTPUT_DIR)
    print("BioMorphX v1.0.0 verification suite passed with 100% success!")

