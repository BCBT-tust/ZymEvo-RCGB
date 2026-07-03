#!/usr/bin/env python3
"""
ZymEvo adaptive docking box builder (deterministic).

This replaces the previous pocket-scoring + Bayesian-optimization box search.
The box is now a pure function of two structure-side priors:

    center = catalytic center                          (from known residues)
    edge   = max_ligand_diameter + 2 * margin          (isotropic cube)

No docking score enters box construction, so the geometry is blind to any
mutant outcome. This removes the "result decides geometry" problem and lets
the box be validated by WT self-recovery independently of the mutant tested.
The chosen edge and its inputs are written into the config header, so the box
size is auditable (this is what replaces ad-hoc size choices such as 20 vs 25).

Confirmed design:
  - isotropic cube box
  - margin default 6 A (working range 4-8)
  - substrate + product SHARE ONE box, sized to the LARGER ligand, both placed
    in the same catalytic-center frame so the catalytic cycle is comparable
  - center resolved by a three-layer fallback; layer 1 (user residues) is the
    default and the only path used for the XynC application case

Removed vs previous version (all only served score-driven box search):
  ActiveSiteDetector, BayesianOptimizer, AutoPocketOptimizer,
  MockDockingScorer, VinaDockingScorer, VinaInstaller.
Kept:
  PDBParser, PDBQTValidator (ligand sanity / auto-fix).
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import pdist

warnings.filterwarnings('ignore')


# Functional side-chain atoms used to place the catalytic center.
# For each catalytic residue we average these atoms; missing atoms fall back to CA.
FUNCTIONAL_ATOMS: Dict[str, List[str]] = {
    'GLU': ['OE1', 'OE2', 'CD'],
    'ASP': ['OD1', 'OD2', 'CG'],
    'HIS': ['NE2', 'ND1'],
    'SER': ['OG'],
    'THR': ['OG1'],
    'CYS': ['SG'],
    'TYR': ['OH'],
    'LYS': ['NZ'],
    'ARG': ['NH1', 'NH2', 'NE'],
    'ASN': ['OD1', 'ND2'],
    'GLN': ['OE1', 'NE2'],
}


@dataclass
class AtomData:
    """Protein structure data"""
    coordinates: np.ndarray
    elements: List[str]
    atom_names: List[str]
    residues: List[str]
    residue_ids: List[int]
    chains: Optional[List[str]] = None


@dataclass
class BoxParameters:
    """Deterministic docking box (no score fields by design)."""
    center_x: float
    center_y: float
    center_z: float
    edge: float                     # cube edge (size_x = size_y = size_z)
    n_atoms: int
    box_volume: float
    max_ligand_diameter: float
    margin: float
    clipped: bool
    center_provenance: str
    per_ligand: List[Tuple[str, float]] = field(default_factory=list)


# ============================================================================
# PDBQT validation (kept from previous version)
# ============================================================================

class PDBQTValidator:
    """PDBQT file validator and auto-fixer"""

    @staticmethod
    def validate_and_fix_pdbqt(pdbqt_file: str, verbose: bool = True) -> Tuple[bool, str, Optional[str]]:
        if not os.path.exists(pdbqt_file):
            return False, "File not found", None
        try:
            with open(pdbqt_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            return False, f"Cannot read file: {e}", None

        has_atoms = False
        has_torsdof = False
        torsdof_value = 0
        n_atoms = 0
        needs_fix = False
        carbon_issues = []

        for i, line in enumerate(lines):
            if line.startswith('ATOM') or line.startswith('HETATM'):
                has_atoms = True
                n_atoms += 1
                if len(line) >= 79:
                    atom_name = line[12:16].strip()
                    atom_type = line[77:79].strip()
                    if atom_name.startswith('C') and atom_type == "C":
                        needs_fix = True
                        carbon_issues.append(i)
            elif line.startswith('TORSDOF'):
                has_torsdof = True
                try:
                    torsdof_value = int(line.split()[1])
                except Exception:
                    pass

        if not has_atoms:
            return False, "No ATOM records found", None
        if not has_torsdof:
            return False, "Missing TORSDOF record", None
        if torsdof_value == 0 and verbose:
            print("  TORSDOF=0 (rigid molecule) - may cause Vina issues")

        if needs_fix:
            if verbose:
                print(f"  Fixing {len(carbon_issues)} carbon atoms with wrong type...")
            fixed_file = pdbqt_file.replace('.pdbqt', '_fixed.pdbqt')
            fixed_lines = []
            for i, line in enumerate(lines):
                if i in carbon_issues:
                    fixed_line = line[:77] + " A" + (line[79:] if len(line) > 79 else "\n")
                    fixed_lines.append(fixed_line)
                else:
                    fixed_lines.append(line)
            try:
                with open(fixed_file, 'w') as f:
                    f.writelines(fixed_lines)
                return True, f"Auto-fixed {len(carbon_issues)} carbon atoms", fixed_file
            except Exception as e:
                return False, f"Fix failed: {e}", None

        return True, f"File OK ({n_atoms} atoms, TORSDOF={torsdof_value})", pdbqt_file


# ============================================================================
# Structure reading
# ============================================================================

def _element_from_name(atom_name: str) -> str:
    letters = ''.join(c for c in atom_name if c.isalpha()).upper()
    if letters[:2] in ('CL', 'BR', 'FE', 'ZN', 'MG', 'CA', 'NA', 'SE', 'MN', 'CU'):
        return letters[:2]
    return letters[:1] if letters else 'C'


class PDBParser:

    ATOMIC_MASSES = {
        'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
        'S': 32.06, 'P': 30.974, 'F': 18.998, 'CL': 35.45,
        'BR': 79.904, 'I': 126.90, 'SE': 78.971, 'FE': 55.845,
        'ZN': 65.38, 'MG': 24.305, 'CA': 40.078, 'NA': 22.990,
        'K': 39.098, 'MN': 54.938, 'CU': 63.546, 'CO': 58.933
    }

    @staticmethod
    def parse_pdb(pdb_file: str) -> Optional[AtomData]:
        if not os.path.exists(pdb_file):
            return None

        coordinates, elements, atom_names, residues, residue_ids, chains = [], [], [], [], [], []
        try:
            with open(pdb_file, 'r') as f:
                for line in f:
                    if not line.startswith('ATOM'):
                        continue
                    try:
                        atom_name = line[12:16].strip()
                        residue = line[17:20].strip()
                        chain = line[21:22].strip()
                        res_id = int(line[22:26].strip())
                        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                        element = line[76:78].strip().upper() if len(line) >= 78 else ''
                        if abs(x) > 9999 or abs(y) > 9999 or abs(z) > 9999:
                            continue
                        if not element:
                            element = _element_from_name(atom_name)
                        coordinates.append([x, y, z])
                        elements.append(element)
                        atom_names.append(atom_name)
                        residues.append(residue)
                        residue_ids.append(res_id)
                        chains.append(chain)
                    except Exception:
                        continue
        except Exception:
            return None

        if not coordinates:
            return None

        return AtomData(
            coordinates=np.array(coordinates, dtype=np.float64),
            elements=elements, atom_names=atom_names,
            residues=residues, residue_ids=residue_ids, chains=chains
        )

    @staticmethod
    def read_ligand_heavy(ligand_file: str) -> np.ndarray:
        """Heavy-atom coordinates of a ligand (PDB or PDBQT; ATOM + HETATM)."""
        if not os.path.exists(ligand_file):
            raise FileNotFoundError(ligand_file)
        coords = []
        with open(ligand_file, 'r') as f:
            for line in f:
                if not (line.startswith('ATOM') or line.startswith('HETATM')):
                    continue
                try:
                    atom_name = line[12:16].strip()
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    continue
                element = line[76:78].strip().upper() if len(line) >= 78 else ''
                if not element:
                    element = _element_from_name(atom_name)
                if element == 'H':
                    continue
                coords.append([x, y, z])
        if not coords:
            raise ValueError(f"No heavy atoms parsed from {ligand_file}")
        return np.asarray(coords, dtype=np.float64)


# ============================================================================
# Layer 1 / 2: catalytic center from known residues
# ============================================================================

class CatalyticCenterResolver:

    @staticmethod
    def center_from_residues(atom_data: AtomData, residue_ids: List[int],
                             chain: Optional[str] = None) -> np.ndarray:
        """Center = centroid of functional side-chain atoms of the given residues.
        Residues with no functional-atom hit fall back to their CA."""
        wanted = set(residue_ids)
        seen = set()
        func_pts: List[np.ndarray] = []
        residues_with_func = set()
        ca_pts: Dict[int, np.ndarray] = {}

        chains = atom_data.chains or [''] * len(atom_data.residue_ids)
        for i, rid in enumerate(atom_data.residue_ids):
            if rid not in wanted:
                continue
            if chain and chains[i] != chain:
                continue
            seen.add(rid)
            resname = atom_data.residues[i]
            aname = atom_data.atom_names[i]
            if aname in FUNCTIONAL_ATOMS.get(resname, []):
                func_pts.append(atom_data.coordinates[i])
                residues_with_func.add(rid)
            if aname == 'CA':
                ca_pts[rid] = atom_data.coordinates[i]

        if not seen:
            raise ValueError(
                f"None of residues {sorted(wanted)} found"
                + (f" in chain {chain}" if chain else "")
            )

        for rid in seen - residues_with_func:
            if rid in ca_pts:
                func_pts.append(ca_pts[rid])

        if not func_pts:
            raise ValueError("Found residues but no usable functional/CA atoms.")

        return np.mean(np.asarray(func_pts, dtype=np.float64), axis=0)

    @staticmethod
    def resolve(atom_data: AtomData,
                residues: Optional[List[int]] = None,
                chain: Optional[str] = None,
                annotation: Optional[List[int]] = None,
                predictor=None) -> Tuple[np.ndarray, str]:
        """Three-layer fallback. Returns (center, provenance).
        Layer 1: user `residues` (default). Layer 2: pre-fetched `annotation`
        ids (M-CSA / UniProt / CAZy-family). Layer 3: `predictor(atom_data)`
        -> residue ids, flagged PREDICTED and requiring WT self-recovery check.
        The center never uses a docking score at any layer."""
        if residues:
            c = CatalyticCenterResolver.center_from_residues(atom_data, residues, chain)
            return c, f"user_residues:{sorted(residues)}"
        if annotation:
            c = CatalyticCenterResolver.center_from_residues(atom_data, annotation, chain)
            return c, f"annotation:{sorted(annotation)}"
        if predictor is not None:
            predicted = predictor(atom_data)
            if not predicted:
                raise ValueError("predictor returned no residues.")
            c = CatalyticCenterResolver.center_from_residues(atom_data, predicted, chain)
            return c, f"PREDICTED(needs WT-self-recovery):{sorted(predicted)}"
        raise ValueError(
            "No catalytic center source. Provide residues (layer 1), "
            "annotation (layer 2), or a predictor (layer 3)."
        )


# ============================================================================
# Adaptive box construction
# ============================================================================

class AdaptiveBoxBuilder:

    @staticmethod
    def ligand_diameter(coords: np.ndarray) -> float:
        """Max internal heavy-atom distance: the dimension the cube must fit
        under free rotation (this is why the box is isotropic)."""
        if len(coords) < 2:
            return 0.0
        return float(pdist(coords).max())

    @staticmethod
    def build(center: np.ndarray, ligand_files: List[str],
              margin: float = 6.0, min_edge: float = 15.0,
              max_edge: float = 30.0, provenance: str = "",
              n_atoms: int = 0) -> BoxParameters:
        if not (4.0 <= margin <= 8.0):
            print(f"  [warn] margin={margin} A is outside the 4-8 A working range.")

        per_ligand = []
        for lf in ligand_files:
            d = AdaptiveBoxBuilder.ligand_diameter(PDBParser.read_ligand_heavy(lf))
            per_ligand.append((os.path.basename(lf), d))

        max_d = max(d for _, d in per_ligand)
        raw_edge = max_d + 2.0 * margin
        edge = float(np.clip(raw_edge, min_edge, max_edge))

        if edge < max_d:
            print(f"  [warn] box edge {edge:.1f} A < largest ligand diameter "
                  f"{max_d:.1f} A. Raise --max_edge.")

        return BoxParameters(
            center_x=float(center[0]), center_y=float(center[1]), center_z=float(center[2]),
            edge=edge, n_atoms=n_atoms, box_volume=float(edge ** 3),
            max_ligand_diameter=max_d, margin=margin,
            clipped=abs(raw_edge - edge) > 1e-6,
            center_provenance=provenance, per_ligand=per_ligand,
        )

    @staticmethod
    def sanity_check(box: BoxParameters, atom_data: AtomData) -> None:
        """Non-gating warnings only. Never changes the box (the box must not be
        tuned to pass any check)."""
        center = np.array([box.center_x, box.center_y, box.center_z])
        min_dist = float(np.min(np.linalg.norm(atom_data.coordinates - center, axis=1)))
        if min_dist > 8.0:
            print(f"  [warn] catalytic center is {min_dist:.1f} A from nearest "
                  f"protein atom; check residue ids / chain.")
        if box.edge > 30.0:
            print(f"  [warn] edge {box.edge:.1f} A exceeds the usual Vina "
                  f"efficiency ceiling (~30 A).")


# ============================================================================
# Output
# ============================================================================

class ParameterWriter:

    @staticmethod
    def write_vina_config(box: BoxParameters, output_file: str, protein_name: str) -> bool:
        try:
            with open(output_file, 'w') as f:
                f.write("# AutoDock Vina configuration - ZymEvo adaptive box\n")
                f.write(f"# Protein: {protein_name}\n")
                f.write("# Deterministic box: center from catalytic prior, edge from ligand size.\n")
                f.write("# No docking score used in construction (blind to mutant outcome).\n")
                f.write(f"# center_provenance = {box.center_provenance}\n")
                f.write(f"# margin = {box.margin:.1f} A\n")
                f.write(f"# max_ligand_diameter = {box.max_ligand_diameter:.2f} A\n")
                for name, d in box.per_ligand:
                    f.write(f"#   ligand {name}: diameter {d:.2f} A\n")
                f.write(f"# applied_edge = {box.edge:.2f} A"
                        + (" (clipped)\n" if box.clipped else "\n"))
                f.write(f"# box_volume = {box.box_volume:.1f} A^3\n")
                f.write("#\n# Reference: Trott & Olson (2010) J Comput Chem 31:455-461\n\n")

                f.write("# Docking box center (A)\n")
                f.write(f"center_x = {box.center_x:.3f}\n")
                f.write(f"center_y = {box.center_y:.3f}\n")
                f.write(f"center_z = {box.center_z:.3f}\n\n")

                f.write("# Docking box size (A) - isotropic cube\n")
                f.write(f"size_x = {box.edge:.3f}\n")
                f.write(f"size_y = {box.edge:.3f}\n")
                f.write(f"size_z = {box.edge:.3f}\n\n")

                f.write("# Recommended Vina parameters\n")
                f.write("exhaustiveness = 32\n")
                f.write("num_modes = 20\n")
                f.write("energy_range = 4\n")
            return True
        except Exception as e:
            print(f"  [error] write config failed: {e}")
            return False

    @staticmethod
    def write_csv_summary(results: List[Tuple[str, BoxParameters]], output_file: str) -> bool:
        try:
            with open(output_file, 'w') as f:
                f.write("Protein,Center_X,Center_Y,Center_Z,Edge,Box_Volume,"
                        "Max_Ligand_Diameter,Margin,Clipped,Center_Provenance\n")
                for name, b in results:
                    prov = b.center_provenance.replace(',', ';')
                    f.write(f"{name},{b.center_x:.3f},{b.center_y:.3f},{b.center_z:.3f},"
                            f"{b.edge:.3f},{b.box_volume:.2f},{b.max_ligand_diameter:.2f},"
                            f"{b.margin:.1f},{int(b.clipped)},{prov}\n")
            return True
        except Exception as e:
            print(f"  [error] write CSV failed: {e}")
            return False


# ============================================================================
# Drivers
# ============================================================================

def build_for_protein(pdb_file: str, ligand_files: List[str], output_dir: str,
                      residues: Optional[List[int]] = None,
                      chain: Optional[str] = None,
                      annotation: Optional[List[int]] = None,
                      margin: float = 6.0, min_edge: float = 15.0,
                      max_edge: float = 30.0, validate_ligands: bool = False
                      ) -> Optional[Tuple[str, BoxParameters]]:
    protein_name = Path(pdb_file).stem

    if validate_ligands:
        checked = []
        for lf in ligand_files:
            if lf.endswith('.pdbqt'):
                ok, msg, fixed = PDBQTValidator.validate_and_fix_pdbqt(lf)
                print(f"  Ligand {os.path.basename(lf)}: {msg}")
                if not ok:
                    print("  Cannot proceed with invalid ligand.")
                    return None
                checked.append(fixed)
            else:
                checked.append(lf)
        ligand_files = checked

    atom_data = PDBParser.parse_pdb(pdb_file)
    if atom_data is None:
        print(f"  Failed to parse: {protein_name}")
        return None

    print(f"\nProcessing: {protein_name}")

    center, provenance = CatalyticCenterResolver.resolve(
        atom_data, residues=residues, chain=chain, annotation=annotation
    )
    box = AdaptiveBoxBuilder.build(
        center, ligand_files, margin=margin, min_edge=min_edge, max_edge=max_edge,
        provenance=provenance, n_atoms=len(atom_data.coordinates)
    )
    AdaptiveBoxBuilder.sanity_check(box, atom_data)

    print(f"  center provenance: {provenance}")
    print(f"  center (A): ({box.center_x:.2f}, {box.center_y:.2f}, {box.center_z:.2f})")
    for name, d in box.per_ligand:
        print(f"    {name}: diameter {d:.2f} A")
    print(f"  edge: {box.edge:.2f} A" + (" (clipped)" if box.clipped else ""))

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{protein_name}_docking_params.txt")
    if ParameterWriter.write_vina_config(box, out_file, protein_name):
        print(f"  config: {out_file}")
        return (protein_name, box)
    return None


def run_manifest(manifest_file: str, output_dir: str, margin: float,
                 min_edge: float, max_edge: float) -> List[Tuple[str, BoxParameters]]:
    """Batch mode. manifest is a JSON list of entries, each:
        {"receptor": "...", "ligands": ["...", "..."],
         "residues": [.., ..], "chain": "A" (optional),
         "annotation": [..] (optional), "margin": 6 (optional)}
    Each enzyme carries its own catalytic residues, as required."""
    with open(manifest_file) as f:
        entries = json.load(f)

    results = []
    for e in entries:
        res = build_for_protein(
            e["receptor"], e["ligands"], output_dir,
            residues=e.get("residues"), chain=e.get("chain"),
            annotation=e.get("annotation"),
            margin=e.get("margin", margin), min_edge=min_edge, max_edge=max_edge,
        )
        if res:
            results.append(res)
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ZymEvo deterministic adaptive docking box builder'
    )
    # Single-receptor mode
    parser.add_argument('--receptor', help='Receptor PDB file')
    parser.add_argument('--ligands', nargs='+',
                        help='Ligand files (substrate and product share one box)')
    parser.add_argument('--residues', nargs='+', type=int,
                        help='Catalytic residue ids (layer 1, default path)')
    parser.add_argument('--chain', default=None, help='Restrict residues to a chain')
    parser.add_argument('--annotation', nargs='+', type=int,
                        help='Pre-fetched annotation residue ids (layer 2)')
    # Batch mode
    parser.add_argument('--manifest', help='JSON manifest for batch mode')
    # Shared
    parser.add_argument('--output', default='docking_params', help='Output directory')
    parser.add_argument('--margin', type=float, default=6.0, help='Margin in A (4-8, default 6)')
    parser.add_argument('--min_edge', type=float, default=15.0)
    parser.add_argument('--max_edge', type=float, default=30.0)
    parser.add_argument('--validate_ligands', action='store_true',
                        help='Run PDBQT validation/auto-fix on ligands first')

    args = parser.parse_args()

    print("=" * 70)
    print("ZymEvo Adaptive Docking Box Builder (deterministic)")
    print("=" * 70)

    if args.manifest:
        results = run_manifest(args.manifest, args.output, args.margin,
                               args.min_edge, args.max_edge)
    else:
        if not (args.receptor and args.ligands):
            parser.error("single mode needs --receptor and --ligands "
                         "(or use --manifest for batch).")
        if not (args.residues or args.annotation):
            parser.error("provide --residues (layer 1) or --annotation (layer 2).")
        res = build_for_protein(
            args.receptor, args.ligands, args.output,
            residues=args.residues, chain=args.chain, annotation=args.annotation,
            margin=args.margin, min_edge=args.min_edge, max_edge=args.max_edge,
            validate_ligands=args.validate_ligands,
        )
        results = [res] if res else []

    if results:
        summary = os.path.join(args.output, "docking_box_summary.csv")
        ParameterWriter.write_csv_summary(results, summary)
        print(f"\nsummary: {summary}")

    print("=" * 70)
    print(f"Done: {len(results)} box(es) built")
    print("=" * 70)


if __name__ == "__main__":
    main()
