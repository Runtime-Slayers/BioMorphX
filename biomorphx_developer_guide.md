# `biomorphx` (v1.0.0) — Developer Guide, API Reference & Life-Critical Safety Architecture

**Paradigm:** *Omni-Scale Morpho-Mechanical Transduction (OMMT)*  
**Distribution Formats:**
1. **`pip`-Installable Python Wheel:** `biomorphx-1.0.0-py3-none-any.whl`
2. **Source Distribution Archive:** `biomorphx-1.0.0.tar.gz`
3. **Zero-Dependency Drop-In Single-File SDK:** `biomorphx.py`

## 1. Why `biomorphx` Exists: Solving the Fragmentation of Biological & Biomedical Computing

Existing Python libraries operate in isolated silos without physical or mechanical coupling:
* **Sequence Libraries (`Biopython`)** treat DNA, RNA, and proteins as 1D text strings, unaware of how a peptide sequence behaves under piconewton mechanical tension (Worm-Like Chain persistence length, Bell-Evans unfolding force $F^*$, or force-gated cryptic motif exposure).
* **Image Processing Libraries (`scikit-image`, `OpenCV`)** diffuse pixel intensities via scalar Fickian heat conduction or second-order Hessian eigenvalues, severing coherent speckle dropouts and erasing 3-way/4-way branching junctions.
* **Biomaterials, 3D Bioprinting & Clinical ICU Tools** lack a unified Python engine that connects **Polymer/Hydrogel Constitutive Rheology $\rightarrow$ 3D Auxetic Bioprinting G-Code & STL Generation with Nozzle Shear-Lysis Viability Auditing $\rightarrow$ Deterministic ICU Hemodynamic & Renal PK/PD Safety Guardrails**.

`biomorphx` compresses workflows that previously required **150 to 400 lines of custom multi-physics code** into **single-line, unit-verified, cryptographically hash-stamped function calls**.

## 2. Installation & CLI Usage

### Install from the Wheel or Source Distribution
```bash
pip install biomorphx-1.0.0-py3-none-any.whl
# OR
pip install biomorphx-1.0.0.tar.gz
```

### Command-Line Interface (`biomorphx`)
```bash
biomorphx --version
biomorphx self-test
biomorphx list-snippets
biomorphx show-snippet 01_protein_mechano_unfolding
biomorphx run-snippet 09_icu_ventilator_power_audit
```

## 3. Complete Submodule & One-Liner API Reference

| Submodule | Primary One-Liner Function | Replaces (~Lines) | Core Mathematical / Physical Engine |
| :--- | :--- | :---: | :--- |
| **`biomorphx.safety`** | `verify_bounds()`, `conformal_safety_interval()` | ~120 lines | Hard physiological survival bounds (`ClinicalBoundsError`), SHA-256 `SafetyCertificate`, finite-sample Split Conformal Prediction intervals |
| **`biomorphx.seqmech`** | `analyze_mechano_protein(seq, ...)` | ~250 lines | Henderson-Hasselbalch $\text{pI}$, Kyte-Doolittle GRAVY, Bustamante-Marko-Siggia Worm-Like Chain (WLC) extension $x(F)$, Bell-Evans unfolding force $F^*\text{ (pN)}$, catch-bond lifetime $\tau(F)$, cryptic motif scanner (`RGD`, `PHSRN`, `IKVAV`, `GPQGIAGQ`) |
| **`biomorphx.seqmech`** | `design_crispr_mechano_guide(dna, ...)` | ~220 lines | SantaLucia (1998) nearest-neighbor $\Delta H^\circ, \Delta S^\circ, \Delta G^\circ_{37}, T_m$ coupled with sequence-dependent DNA persistence length $L_p$ and supercoiling torque R-loop invasion efficiency |
| **`biomorphx.seqmech`** | `optimize_mrna_construct(prot, ...)` | ~180 lines | Human Codon Adaptation Index ($\text{CAI}$) optimizer with homopolymer/hairpin avoidance and $5'/3'\text{ UTR}$ + poly(A) assembly |
| **`biomorphx.biomaterials`** | `design_chrono_hydrogel(target_tissue, ...)` | ~300 lines | 12-tissue native atlas, Fractional Zener + 2nd-order Auxetic Metamaterial rheology, Flory-Rehner mesh size $\xi\text{ (nm)}$, Visco-Harmonic Phase Dichroism $\Delta\Phi_{\text{VH}}$, YAP/TAZ & Piezo1 fibrosis predictor |
| **`biomorphx.biomaterials`** | `simulate_scaffold_degradation(polymer, ...)` | ~240 lines | Autocatalytic ester/MMP scission ODE, percolation modulus decay $E(t)/E_0$, local acidic pore $\text{pH}(t)$ necrosis guardrail, biphasic Korsmeyer-Peppas drug release |
| **`biomorphx.nanofab`** | `generate_auxetic_gcode_and_stl(...)` | ~380 lines | Re-entrant bow-tie auxetic 3D Bioprinter `.gcode` + watertight 3D `.stl` mesh generator + Herschel-Bulkley non-Newtonian nozzle wall shear stress $\tau_w\text{ (Pa)}$ & live-cell viability (%) auditor |
| **`biomorphx.nanofab`** | `simulate_lnp_nanocarrier(...)` | ~190 lines | Staggered herringbone microfluidic mixing time $\tau_{\text{mix}}$, Damköhler nanoprecipitation size $D_h\text{ (nm)}$, PDI, apparent $\text{pK}_a$, endosomal escape potency |
| **`biomorphx.imaging`** | `mbsp_filter_and_quantify(image_2d, ...)` | ~320 lines | 4-Stage Mechano-Beltrami Stress-Percolation (MBSP) Filter + automated nematic alignment order $S$, branching junction density ($\text{nodes/mm}^2$), EDT pore diameter, and Euler characteristic $\chi = \beta_0 - \beta_1$ |
| **`biomorphx.clinical`** | `audit_icu_hemodynamics_ventilation(...)` | ~260 lines | MAP, Shock Index, Windkessel compliance, Berlin ARDS $\text{PaO}_2/\text{FiO}_2$ staging, Driving Pressure $\Delta P$, Gattinoni Mechanical Power of Ventilation ($\text{J/min}$) with VILI barotrauma guardrails |
| **`biomorphx.clinical`** | `optimize_renal_pkpd_dosing(drug_name, ...)` | ~240 lines | Cockcroft-Gault $\text{CrCl}$, Sawchuk-Zaske intermittent IV infusion dosing, steady-state $C_{\max,\text{ss}}, C_{\min,\text{ss}}$, $\text{AUC}_{24}/\text{MIC}$, and nephrotoxicity trough guardrails |

## 4. Built-In One-Liner Code Snippets (`biomorphx.snippets`)

You can list, inspect, or execute any of the 12 built-in translational workflows directly in Python:

```python
import biomorphx as bmx

# List all 12 built-in snippets
for item in bmx.list_snippets():
    print(f"{item['name']} ({item['domain']}): {item['description']}")

# View source code of any snippet
print(bmx.get_snippet("06_bioprinter_gcode_and_stl"))

# Execute any snippet directly and get the full dictionary + SHA-256 SafetyCertificate
result = bmx.run_snippet("09_icu_ventilator_power_audit")
print(result["safety_certificate"])
```
