#!/usr/bin/env python3

from __future__ import annotations

import os
import re
import glob
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Constants -- mirror FuncSite-ML so the contact definition stays identical
# =============================================================================

CHARGED_POS = {"LYS", "ARG", "HIS"}
CHARGED_NEG = {"ASP", "GLU"}
CHARGED_ALL = CHARGED_POS | CHARGED_NEG

HYDROPHOBIC_RES = {
    "ALA", "VAL", "LEU", "ILE", "MET",
    "PHE", "TRP", "PRO", "TYR", "CYS",
}

HALOGENS = {"F", "CL", "BR", "I"}

POLAR_RECEPTOR = set("NOS")
POLAR_LIGAND = set("NOFS")
HYDROPHOBIC_RECEPTOR = set("CS")
HYDROPHOBIC_LIGAND = {"C", "S", "F", "CL", "BR", "I"}

HBOND_CUTOFF = 3.5         # polar / H-bond
IONIC_CUTOFF = 4.0         # salt bridge
HYDROPHOBIC_CUTOFF = 4.5   # hydrophobic stacking
HALOGEN_CUTOFF = 4.0       # halogen bond
CLASH_CUTOFF = 2.0         # steric clash (same threshold FuncSite-ML penalises)

CONTACT_TYPES = ("ionic", "polar", "halogen", "hydrophobic")
DEFAULT_REFERENCE_POSE_ENSEMBLE_N = 5

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA1TO3 = {v: k for k, v in AA3TO1.items()}
STANDARD_AA1 = list("ACDEFGHIKLMNPQRSTVWY")


# =============================================================================
# Low-level parsing / geometry (pure Python)
# =============================================================================

TWO_CHAR_ELEM = {"CL", "BR", "MG", "ZN", "CA", "FE", "MN",
                 "NA", "CU", "NI", "CO", "SE", "SI"}


def get_element(atom_name: str) -> str:
    """Element symbol from a PDB/PDBQT atom name (mirrors FuncSite-ML)."""
    name = re.sub(r"\d", "", atom_name.strip()).strip().upper()
    if not name:
        return "X"
    if len(name) >= 2 and name[:2] in TWO_CHAR_ELEM:
        return name[:2]
    return name[0]


def standardize_pdb_for_receptor_prep(inp_pdb: str, out_pdb: str) -> str:
    """Rewrite loose FoldX PDB records into strict PDB columns.

    FoldX can emit readable but non-column-strict PDB lines, e.g.
    "ATOM 0 N LEU A 15 x y z ...". AutoDockTools prepare_receptor4 is more
    reliable when the PDB is first normalized to fixed-width PDB columns.
    """
    serial = 1
    wrote_atom = False
    with open(inp_pdb) as fin, open(out_pdb, "w") as fout:
        for raw in fin:
            if raw.startswith(("ATOM", "HETATM")):
                rec = raw[:6].strip() or "ATOM"
                parts = raw.split()
                try:
                    if len(parts) >= 11 and parts[0] in ("ATOM", "HETATM"):
                        atom_name = parts[2]
                        res_name = parts[3]
                        chain = parts[4] or "A"
                        res_num = int(float(parts[5]))
                        x, y, z = map(float, parts[6:9])
                        occ = float(parts[9])
                        bfac = float(parts[10])
                    else:
                        atom_name = raw[12:16].strip()
                        res_name = raw[17:20].strip()
                        chain = raw[21:22].strip() or "A"
                        res_num = int(raw[22:26])
                        x = float(raw[30:38])
                        y = float(raw[38:46])
                        z = float(raw[46:54])
                        occ = float(raw[54:60] or 1.0)
                        bfac = float(raw[60:66] or 0.0)
                except (IndexError, ValueError):
                    continue
                elem = get_element(atom_name).title()
                fout.write(
                    f"{rec:<6}{serial:5d} {atom_name:<4} {res_name:>3} "
                    f"{chain:1}{res_num:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{occ:6.2f}{bfac:6.2f}          {elem:>2}\n"
                )
                serial += 1
                wrote_atom = True
            elif raw.startswith(("TER", "END")):
                fout.write(raw if raw.endswith("\n") else raw + "\n")
        if wrote_atom:
            fout.write("END\n")
    if not wrote_atom:
        raise RuntimeError(f"standardized PDB has no ATOM/HETATM records: {inp_pdb}")
    return out_pdb


def residue_id(res_name: str, res_num: str, chain: str) -> str:
    """FuncSite-ML residue_id convention: e.g. 'SER139A'."""
    return f"{res_name}{res_num}{chain or 'A'}"


def parse_receptor_pdbqt(content: str) -> Dict[str, dict]:
    """Parse a receptor PDB/PDBQT into {residue_id: {res_name,res_num,chain,atoms}}."""
    residues: Dict[str, dict] = {}
    for line in content.split("\n"):
        if line.startswith(("ATOM", "HETATM")):
            try:
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                res_num = line[22:26].strip()
                chain = line[21:22].strip() or "A"
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except (IndexError, ValueError):
                continue
            rid = residue_id(res_name, res_num, chain)
            res = residues.setdefault(
                rid, {"res_name": res_name, "res_num": res_num,
                      "chain": chain, "atoms": {}})
            res["atoms"][atom_name] = [x, y, z]
    return residues


def validate_receptor_structure(path: str, label: str = "receptor") -> Dict[str, dict]:
    """Read and validate a receptor PDB/PDBQT before scoring or docking."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    with open(path) as f:
        content = f.read()
    has_atom_line = any(ln.startswith(("ATOM", "HETATM"))
                        for ln in content.splitlines())
    if not has_atom_line:
        raise ValueError(f"{label} has no ATOM/HETATM lines: {path}")
    residues = parse_receptor_pdbqt(content)
    if not residues:
        raise ValueError(
            f"{label} contains ATOM/HETATM lines but no parseable residues: {path}"
        )
    return residues


def parse_ligand_poses(content: str) -> List[dict]:
    """Parse a docked-ligand PDBQT (multi-MODEL) into a list of poses.

    Each pose: {'model': int, 'score': float, 'atoms': [(name, elem, [x,y,z])]}.
    Atom order is preserved, so two poses of the SAME ligand are index-aligned.
    """
    poses: List[dict] = []
    cur = None
    for line in content.split("\n"):
        if line.startswith("MODEL"):
            try:
                cur = {"model": int(line.split()[1]), "score": None, "atoms": []}
            except (IndexError, ValueError):
                cur = {"model": None, "score": None, "atoms": []}
        elif line.startswith("REMARK VINA RESULT:") and cur is not None:
            try:
                cur["score"] = float(line.split()[3])
            except (IndexError, ValueError):
                pass
        elif line.startswith(("ATOM", "HETATM")):
            if cur is None:
                cur = {"model": 1, "score": None, "atoms": []}
            try:
                name = line[12:16].strip()
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                raw_type = line[77:79].strip() if len(line) >= 79 else ""
                elem = get_element(raw_type if raw_type else name)
                cur["atoms"].append((name, elem, [x, y, z]))
            except (IndexError, ValueError):
                continue
        elif line.startswith("ENDMDL") and cur is not None:
            poses.append(cur)
            cur = None
    # single-model files without MODEL/ENDMDL
    if cur is not None and cur["atoms"]:
        poses.append(cur)
    return poses


def choose_reference_ligand_pose(poses: Sequence[dict],
                                 user_supplied: bool = False) -> dict:
    """Choose the WT reference ligand pose.

    If the user provides a WT reference pose file, the first MODEL is used
    intentionally. This lets a manually checked pose remain the reference even
    if the file contains multiple docking models with different Vina scores.

    If the pose file was generated internally by this pipeline, the lowest
    Vina score remains the default reference.
    """
    if not poses:
        raise ValueError("no WT reference pose parsed")
    if user_supplied:
        return poses[0]
    return min((p for p in poses if p.get("score") is not None),
               key=lambda p: p["score"], default=poses[0])


def choose_reference_ligand_pose_ensemble(
    poses: Sequence[dict],
    user_supplied: bool = False,
    n: int = DEFAULT_REFERENCE_POSE_ENSEMBLE_N,
) -> List[dict]:
    """Choose WT reference poses for RMSD comparison.

    User-supplied WT pose files are treated as a checked pose ensemble: the
    first n parsed MODELs are accepted as equivalent WT-like references for
    pose RMSD, while MODEL 1 remains the contact/score baseline.

    Internally generated WT docking keeps the stricter single lowest-score
    reference, preserving the original behavior when no external WT pose file
    is provided.
    """
    if not poses:
        raise ValueError("no WT reference pose parsed")
    if user_supplied:
        return list(poses[:max(1, int(n))])
    return [choose_reference_ligand_pose(poses, user_supplied=False)]


def parse_vina_box(config_text: str) -> dict:
    """Parse center_x/y/z and size_x/y/z (and seed/exhaustiveness if present)."""
    box: Dict[str, float] = {}
    for line in config_text.split("\n"):
        line = line.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        k, v = (s.strip() for s in line.split("=", 1))
        try:
            box[k] = float(v)
        except ValueError:
            pass
    required = ["center_x", "center_y", "center_z", "size_x", "size_y", "size_z"]
    missing = [r for r in required if r not in box]
    if missing:
        raise ValueError(f"vina box missing fields: {missing}")
    return box


def best_score(poses: Sequence[dict]) -> Optional[float]:
    s = [p["score"] for p in poses if p.get("score") is not None]
    return min(s) if s else None


def _heavy_atoms(pose: dict) -> List[Tuple[str, str, np.ndarray]]:
    """Heavy atoms as (name, element, coord) tuples."""
    out = []
    for name, elem, xyz in pose["atoms"]:
        if elem.upper() == "H":
            continue
        out.append((name, elem.upper(), np.asarray(xyz, float)))
    return out


def _rmsd_symmetry_safe(a: List[Tuple[str, str, np.ndarray]],
                        b: List[Tuple[str, str, np.ndarray]]) -> float:
    """Element-wise optimal-assignment RMSD (handles symmetric atoms).

    Within each element, match atoms of pose A to pose B by minimising total
    squared distance (Hungarian via scipy if available, else greedy nearest).
    This corrects the common symmetry problem (e.g. equivalent ring atoms or
    two equivalent oxygens) without a full graph-isomorphism treatment.
    """
    from collections import defaultdict
    ga, gb = defaultdict(list), defaultdict(list)
    for _, e, x in a:
        ga[e].append(x)
    for _, e, x in b:
        gb[e].append(x)
    if sorted((e, len(v)) for e, v in ga.items()) != \
       sorted((e, len(v)) for e, v in gb.items()):
        return float("nan")  # composition mismatch

    total_sq, n = 0.0, 0
    try:
        from scipy.optimize import linear_sum_assignment
        have_scipy = True
    except Exception:
        have_scipy = False

    for e, xa in ga.items():
        xb = gb[e]
        XA = np.asarray(xa); XB = np.asarray(xb)
        cost = ((XA[:, None, :] - XB[None, :, :]) ** 2).sum(axis=2)
        if have_scipy:
            ri, ci = linear_sum_assignment(cost)
            total_sq += cost[ri, ci].sum()
        else:  # greedy fallback
            used = set()
            for i in range(cost.shape[0]):
                order = np.argsort(cost[i])
                for j in order:
                    if j not in used:
                        used.add(j); total_sq += cost[i, j]; break
        n += len(xa)
    return float(np.sqrt(total_sq / n)) if n else float("nan")


def ligand_rmsd(pose_a: dict, pose_b: dict,
                symmetry_safe: bool = False) -> float:
    """Frame-locked heavy-atom RMSD between two poses of the SAME ligand.

    No superposition: both receptors must share one coordinate frame
    (FoldX RepairPDB / BuildModel do not recenter) and the docking box is in
    absolute coordinates, so poses are directly comparable.

    Matching strategy:
      * symmetry_safe=True  -> element-wise optimal assignment (robust to
        symmetric atoms and to atom-order changes).
      * symmetry_safe=False -> match by atom name (robust to order changes);
        falls back to index order, then to symmetry-safe, if names don't map.
    """
    a = _heavy_atoms(pose_a)
    b = _heavy_atoms(pose_b)
    if not a or not b:
        return float("nan")

    if symmetry_safe:
        return _rmsd_symmetry_safe(a, b)

    # name-based matching (handles changed atom order, exact atom identity)
    names_a = [n for n, _, _ in a]
    bmap: Dict[str, List[np.ndarray]] = {}
    for n, _, x in b:
        bmap.setdefault(n, []).append(x)
    if (len(set(names_a)) == len(names_a)
            and all(n in bmap and len(bmap[n]) == 1 for n in names_a)):
        diff = np.asarray([x - bmap[n][0] for n, _, x in a])
        return float(np.sqrt((diff * diff).sum() / len(a)))

    # index fallback when counts match
    if len(a) == len(b):
        ca = np.asarray([x for _, _, x in a])
        cb = np.asarray([x for _, _, x in b])
        diff = ca - cb
        return float(np.sqrt((diff * diff).sum() / len(a)))

    # last resort: symmetry-safe assignment
    return _rmsd_symmetry_safe(a, b)


# =============================================================================
# Contact fingerprint (IFP) + clash -- reuse FuncSite-ML classification
# =============================================================================

def compute_contacts(receptor_residues: Dict[str, dict],
                     ligand_atoms: Sequence[Tuple[str, str, List[float]]]
                     ) -> Tuple[Dict[str, dict], int]:
    """Per-residue contact fingerprint and global clash count for one pose.

    Returns (ifp, n_clash) where ifp[residue_id] = {
        'ionic','polar','halogen','hydrophobic': counts,
        'min_dist': float, 'dominant': str or None}.
    Pair classification is mutually exclusive with priority
    ionic > polar > halogen > hydrophobic, identical to FuncSite-ML.
    """
    ifp: Dict[str, dict] = {}
    n_clash = 0

    lig = [(elem.upper(), np.asarray(xyz, float)) for _, elem, xyz in ligand_atoms]

    for rid, res in receptor_residues.items():
        res_name = res["res_name"].upper()
        is_charged = res_name in CHARGED_ALL
        rec = {t: 0 for t in CONTACT_TYPES}
        rec["min_dist"] = float("inf")

        for ratom_name, rxyz in res["atoms"].items():
            elem_r = get_element(ratom_name)
            rxyz = np.asarray(rxyz, float)
            for elem_l, lxyz in lig:
                dist = float(np.linalg.norm(rxyz - lxyz))
                if dist < rec["min_dist"]:
                    rec["min_dist"] = dist
                if dist < CLASH_CUTOFF:
                    n_clash += 1

                classified = False
                # ionic
                if not classified and dist <= IONIC_CUTOFF and is_charged:
                    if (res_name in CHARGED_POS and elem_l in {"O", "S"}) or \
                       (res_name in CHARGED_NEG and elem_l == "N"):
                        rec["ionic"] += 1; classified = True
                # polar / H-bond
                if not classified and dist <= HBOND_CUTOFF:
                    if elem_r in POLAR_RECEPTOR and elem_l in POLAR_LIGAND:
                        rec["polar"] += 1; classified = True
                # halogen
                if not classified and dist <= HALOGEN_CUTOFF:
                    if elem_l in HALOGENS and elem_r in POLAR_RECEPTOR:
                        rec["halogen"] += 1; classified = True
                # hydrophobic
                if not classified and dist <= HYDROPHOBIC_CUTOFF:
                    if (res_name in HYDROPHOBIC_RES
                            and elem_r in HYDROPHOBIC_RECEPTOR
                            and elem_l in HYDROPHOBIC_LIGAND):
                        rec["hydrophobic"] += 1; classified = True

        total = sum(rec[t] for t in CONTACT_TYPES)
        if total > 0:
            rec["dominant"] = max(CONTACT_TYPES, key=lambda t: rec[t])
            if rec[rec["dominant"]] == 0:
                rec["dominant"] = None
        else:
            rec["dominant"] = None
        if rec["min_dist"] == float("inf"):
            rec["min_dist"] = 999.0
        ifp[rid] = rec

    return ifp, n_clash


def residue_contact_weight(rec: dict) -> float:
    """Weight a residue's WT contact by type (catalytic-relevant > hydrophobic)."""
    w = {"ionic": 1.5, "polar": 1.5, "halogen": 1.0, "hydrophobic": 0.5}
    return sum(w[t] * rec[t] for t in CONTACT_TYPES)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EnvelopeConfig:
    # --- locked docking parameters (identical for WT and every mutant) ---
    exhaustiveness: int = 16
    num_modes: int = 10
    seed: int = 42

    # --- hard exclusion gates ---
    clash_tol: int = 0               # max NEW clashes allowed
    retention_floor: float = 0.70    # min weighted key-contact retention
    rmsd_cap: float = 2.0            # max substrate pose RMSD vs WT (A)
    score_band: float = 1.5          # max docking-score worsening (kcal/mol)
    ddg_guard_line: float = 2.5      # exclude if global ddG > this (unfolding)
    fail_on_missing_ddg: bool = True # NaN ddG -> exclude (do not silently pass)
    symmetry_safe_rmsd: bool = False # element-wise assignment for symmetric ligands

    # --- key-contact set definition on WT ---
    key_contact_min_weight: float = 1.0   # WT residue counts as "key" if >= this
    exclude_mutated_site: bool = True     # mutated residue may change freely

    # --- envelope preservation score E (NO stability term) ---
    w_contact: float = 0.5
    w_rmsd: float = 0.3
    w_score: float = 0.2

    # --- saturation set ---
    saturation_aa: Sequence[str] = field(default_factory=lambda: list(STANDARD_AA1))

    # --- frame-consistency sanity ---
    frame_centroid_tol: float = 1.0  # CA centroid drift WT-vs-mutant (A)


# =============================================================================
# Reference envelope (built once from the WT complex)
# =============================================================================

class ReferenceEnvelope:
    """The fixed WT substrate envelope: pose + key-contact set + clash baseline."""

    def __init__(self, receptor_residues: Dict[str, dict], ligand_pose: dict,
                 config: EnvelopeConfig,
                 site_weights: Optional[Dict[str, float]] = None,
                 ligand_pose_ensemble: Optional[Sequence[dict]] = None):
        self.config = config
        self.receptor_residues = receptor_residues
        self.ligand_pose = ligand_pose
        self.ligand_pose_ensemble = (
            list(ligand_pose_ensemble) if ligand_pose_ensemble else [ligand_pose]
        )
        self.site_weights = site_weights or {}

        self.ifp, self.clash_baseline = compute_contacts(
            receptor_residues, ligand_pose["atoms"])
        self.ca_centroid = self._ca_centroid(receptor_residues)
        self.best_score = ligand_pose.get("score")

        # key-contact set: weighted WT contacts, optionally boosted by an
        # external per-residue weight map (e.g. FuncSite-ML catalytic_prob)
        self.key_weights: Dict[str, float] = {}
        for rid, rec in self.ifp.items():
            base = residue_contact_weight(rec)
            if base <= 0:
                continue
            ext = 1.0 + float(self.site_weights.get(rid, 0.0))
            w = base * ext
            if w >= config.key_contact_min_weight:
                self.key_weights[rid] = w

    @staticmethod
    def _ca_centroid(residues: Dict[str, dict]) -> np.ndarray:
        cas = [r["atoms"]["CA"] for r in residues.values() if "CA" in r["atoms"]]
        return np.mean(np.asarray(cas, float), axis=0) if cas else np.zeros(3)

    def key_set(self, mutated_residue_id: Optional[str]) -> Dict[str, float]:
        if self.config.exclude_mutated_site and mutated_residue_id:
            return {k: v for k, v in self.key_weights.items()
                    if k != mutated_residue_id}
        return dict(self.key_weights)


# =============================================================================
# Envelope scorer (pure Python; the scientific core of this layer)
# =============================================================================

@dataclass
class EnvelopeResult:
    site: str
    target_aa: str
    survived: bool
    E: float
    contact_retention: float
    pose_rmsd: float
    n_new_clash: int
    dscore: float
    ddg_guard: float
    fail_reasons: List[str]


class EnvelopeScorer:
    def __init__(self, reference: ReferenceEnvelope, config: EnvelopeConfig):
        self.ref = reference
        self.config = config

    def score(self, site: str, target_aa: str,
              mutant_receptor: Dict[str, dict],
              mutant_poses: Sequence[dict],
              ddg_guard: float) -> EnvelopeResult:
        cfg = self.config
        reasons: List[str] = []

        # frame consistency (FoldX must not have recentred the structure)
        mut_centroid = ReferenceEnvelope._ca_centroid(mutant_receptor)
        if np.linalg.norm(mut_centroid - self.ref.ca_centroid) > cfg.frame_centroid_tol:
            reasons.append("frame_shift")  # RMSD comparison would be invalid

        if not mutant_poses:
            return EnvelopeResult(site, target_aa, False, 0.0, 0.0,
                                  float("nan"), 999, float("inf"),
                                  ddg_guard, ["no_pose"])

        # use the best-scoring mutant pose as the representative
        mut_pose = min(
            (p for p in mutant_poses if p.get("score") is not None),
            key=lambda p: p["score"], default=mutant_poses[0])

        # --- pose RMSD vs WT reference ensemble (frame-locked) ---
        rmsd_values = [
            ligand_rmsd(ref_pose, mut_pose,
                        symmetry_safe=cfg.symmetry_safe_rmsd)
            for ref_pose in self.ref.ligand_pose_ensemble
        ]
        rmsd_values = [v for v in rmsd_values if not np.isnan(v)]
        rmsd = min(rmsd_values) if rmsd_values else float("nan")

        # --- contacts on the mutant pose ---
        mut_ifp, mut_clash = compute_contacts(mutant_receptor, mut_pose["atoms"])

        # --- key-contact retention (rest-of-pocket must keep gripping) ---
        key = self.ref.key_set(site)
        if key:
            retained = 0.0
            for rid, w in key.items():
                wt_dom = self.ref.ifp[rid]["dominant"]
                m = mut_ifp.get(rid)
                ok = (m is not None
                      and sum(m[t] for t in CONTACT_TYPES) > 0
                      and (wt_dom is None or m[t_present := wt_dom] > 0))
                if ok:
                    retained += w
            contact_retention = retained / sum(key.values())
        else:
            contact_retention = 1.0  # no key contacts to lose

        # --- new clashes (mutant minus WT baseline) ---
        n_new_clash = max(0, mut_clash - self.ref.clash_baseline)

        # --- docking-score band (don't reward improvement; only punish loss) ---
        mut_best = best_score(mutant_poses)
        if mut_best is not None and self.ref.best_score is not None:
            dscore = mut_best - self.ref.best_score  # >0 means worse
        else:
            dscore = float("nan")

        # --- hard gates ---
        if n_new_clash > cfg.clash_tol:
            reasons.append("new_clash")
        if contact_retention < cfg.retention_floor:
            reasons.append("contact_loss")
        if not np.isnan(rmsd) and rmsd > cfg.rmsd_cap:
            reasons.append("pose_drift")
        if not np.isnan(dscore) and dscore > cfg.score_band:
            reasons.append("score_drop")
        if np.isnan(ddg_guard):
            if cfg.fail_on_missing_ddg:
                reasons.append("ddg_missing")  # do NOT let NaN slip past the guard
        elif ddg_guard > cfg.ddg_guard_line:
            reasons.append("unfolding")

        survived = len(reasons) == 0

        # --- envelope preservation score E (objective-agnostic) ---
        rmsd_term = 1.0 - min((0.0 if np.isnan(rmsd) else rmsd) / cfg.rmsd_cap, 1.0)
        score_term = 1.0 - min(max(0.0, 0.0 if np.isnan(dscore) else dscore)
                               / max(cfg.score_band, 1e-6), 1.0)
        E = (cfg.w_contact * contact_retention
             + cfg.w_rmsd * rmsd_term
             + cfg.w_score * score_term)

        return EnvelopeResult(
            site=site, target_aa=target_aa, survived=survived, E=round(E, 4),
            contact_retention=round(contact_retention, 4),
            pose_rmsd=(float("nan") if np.isnan(rmsd) else round(rmsd, 3)),
            n_new_clash=n_new_clash,
            dscore=(float("nan") if np.isnan(dscore) else round(dscore, 3)),
            ddg_guard=round(ddg_guard, 3), fail_reasons=reasons)


# =============================================================================
# External-tool wrappers (require installed binaries; injected for testability)
# =============================================================================

def _run(cmd, cwd: Optional[str], timeout: int, what: str,
         log_dir: Optional[str] = None):
    """Run an external command, capture output, fail loudly on error.

    Raises RuntimeError with the stderr tail when the command returns non-zero
    or times out. Writes full stdout/stderr to <log_dir>/<what>.log if given.
    """
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout,
                              shell=isinstance(cmd, str))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{what}: timeout after {timeout}s") from exc

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", what)
        with open(os.path.join(log_dir, f"{safe}.log"), "w") as f:
            f.write(f"$ {cmd}\n\n[returncode] {proc.returncode}\n")
            f.write(f"\n[stdout]\n{proc.stdout}\n\n[stderr]\n{proc.stderr}\n")

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        raise RuntimeError(f"{what}: exit {proc.returncode}: {' | '.join(tail)}")
    return proc

class FoldXMutator:
    """RepairPDB once, then BuildModel per single-point mutation (out-pdb=true).

    Reuses the proven RepairPDB -> BuildModel flow; ddG is taken as a GUARD.
    """

    def __init__(self, foldx_bin: str, repaired_pdb: str, work_root: str,
                 number_of_runs: int = 3, timeout: int = 600):
        self.foldx_bin = foldx_bin
        self.repaired_pdb = repaired_pdb
        self.repaired_name = os.path.basename(repaired_pdb)
        self.work_root = work_root
        self.number_of_runs = number_of_runs
        self.timeout = timeout
        os.makedirs(work_root, exist_ok=True)

    @staticmethod
    def repair(foldx_bin: str, pdb_path: str, work_root: str,
               timeout: int = 600) -> str:
        """Run RepairPDB and return the repaired PDB path."""
        os.makedirs(work_root, exist_ok=True)
        name = os.path.basename(pdb_path)
        shutil.copy2(pdb_path, os.path.join(work_root, name))
        with open(os.path.join(work_root, "repair.cfg"), "w") as f:
            f.write(f"command=RepairPDB\npdb={name}\n")
        _run([foldx_bin, "-f", "repair.cfg"], cwd=work_root, timeout=timeout,
             what="foldx_repair", log_dir=os.path.join(work_root, "logs"))
        cands = glob.glob(os.path.join(work_root, "*Repair*.pdb"))
        if not cands:
            raise RuntimeError("RepairPDB returned 0 but produced no *Repair*.pdb")
        return cands[0]

    @staticmethod
    def mutation_string(site_residue_id: str, target_aa1: str) -> str:
        """FuncSite-ML residue_id 'SER139A' + 'W' -> FoldX 'SA139W;'."""
        m = re.match(r"([A-Z]{3})(\d+)([A-Za-z])$", site_residue_id)
        if not m:
            raise ValueError(f"cannot parse residue_id: {site_residue_id}")
        res3, num, chain = m.group(1), m.group(2), m.group(3)
        wt1 = AA3TO1.get(res3.upper(), "X")
        return f"{wt1}{chain}{num}{target_aa1};"

    def build(self, site_residue_id: str, target_aa1: str) -> Tuple[str, float]:
        """Build one mutant; return (mutant_pdb_path, ddG)."""
        tag = f"{site_residue_id}_{target_aa1}"
        mut_dir = os.path.join(self.work_root, f"mut_{tag}")
        os.makedirs(mut_dir, exist_ok=True)
        shutil.copy2(self.repaired_pdb, os.path.join(mut_dir, self.repaired_name))

        with open(os.path.join(mut_dir, "individual_list.txt"), "w") as f:
            f.write(self.mutation_string(site_residue_id, target_aa1) + "\n")
        with open(os.path.join(mut_dir, "build.cfg"), "w") as f:
            f.write("command=BuildModel\n")
            f.write(f"pdb={self.repaired_name}\n")
            f.write("mutant-file=individual_list.txt\n")
            f.write(f"numberOfRuns={self.number_of_runs}\n")
            f.write("out-pdb=true\n")          # <-- need the structure downstream

        _run([self.foldx_bin, "-f", "build.cfg"], cwd=mut_dir,
             timeout=self.timeout, what=f"foldx_build_{tag}",
             log_dir=os.path.join(mut_dir, "logs"))

        ddg = self._parse_ddg(mut_dir)
        mut_pdb = self._pick_mutant_pdb(mut_dir)
        if mut_pdb is None:
            raise RuntimeError(f"BuildModel produced no mutant PDB for {tag}")
        return mut_pdb, ddg

    @staticmethod
    def _parse_ddg(mut_dir: str) -> float:
        dif = glob.glob(os.path.join(mut_dir, "Dif_*.fxout"))
        if not dif:
            return float("nan")
        with open(dif[0]) as f:
            for line in f:
                if line.startswith("Pdb") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        continue
        return float("nan")

    @staticmethod
    def _pick_mutant_pdb(mut_dir: str) -> Optional[str]:
        """Pick the mutant PDB produced by FoldX BuildModel.

        FoldX can write mutant files such as WT_Repair_1_0.pdb when the input
        structure is WT_Repair.pdb. WT control files are usually named
        WT_WT_Repair_1_0.pdb, so we exclude only those WT controls and keep
        repaired-output mutant names.
        """
        preferred, fallback = [], []
        for p in sorted(glob.glob(os.path.join(mut_dir, "*.pdb"))):
            name = os.path.basename(p)
            lower = name.lower()
            if lower.startswith("wt_wt_") or "_wt_" in lower:
                continue
            if re.search(r"_\d+(?:_\d+)?\.pdb$", name):
                preferred.append(p)
            else:
                fallback.append(p)
        if preferred:
            return sorted(preferred)[0]
        return sorted(fallback)[0] if fallback else None


class PDBToPDBQTConverter:
    """Mutant PDB -> receptor PDBQT, then strip TORSDOF (rigid receptor).

    Injection point for AutoPrep-Dock. Provide either a Python callable
    (preferred: reuse AutoPrep-Dock's receptor branch) or a shell command
    template containing {inp} and {out} placeholders.
    """

    def __init__(self, fn: Optional[Callable[[str, str], None]] = None,
                 command_template: Optional[str] = None):
        if fn is None and command_template is None:
            raise ValueError("supply AutoPrep-Dock callable or command_template")
        self.fn = fn
        self.command_template = command_template

    def convert(self, pdb_path: str, out_pdbqt: str) -> str:
        if self.fn is not None:
            self.fn(pdb_path, out_pdbqt)
        else:
            cmd = self.command_template.format(inp=pdb_path, out=out_pdbqt)
            _run(cmd, cwd=None, timeout=300, what="pdb_to_pdbqt")
        if not os.path.exists(out_pdbqt):
            raise RuntimeError(f"converter produced no output: {out_pdbqt}")
        self._strip_torsdof(out_pdbqt)
        validate_receptor_structure(out_pdbqt, "converted receptor PDBQT")
        return out_pdbqt

    @staticmethod
    def _strip_torsdof(pdbqt_path: str) -> None:
        with open(pdbqt_path) as f:
            lines = [ln for ln in f if not ln.startswith("TORSDOF")]
        with open(pdbqt_path, "w") as f:
            f.writelines(lines)


class LockedBoxDocker:
    """Re-dock the SAME ligand into a mutant receptor using the WT box.

    Box is in absolute coordinates and identical for every mutant, with a
    fixed seed/exhaustiveness, so the resulting poses are directly comparable
    to the WT reference pose without superposition.
    """

    def __init__(self, vina_bin: str, box: dict, config: EnvelopeConfig,
                 timeout: int = 600, log_dir: Optional[str] = None):
        self.vina_bin = vina_bin
        self.box = box
        self.config = config
        self.timeout = timeout
        self.log_dir = log_dir

    def dock(self, receptor_pdbqt: str, ligand_pdbqt: str,
             out_pdbqt: str) -> List[dict]:
        cfg = self.config
        cmd = [
            self.vina_bin,
            "--receptor", receptor_pdbqt,
            "--ligand", ligand_pdbqt,
            "--center_x", str(self.box["center_x"]),
            "--center_y", str(self.box["center_y"]),
            "--center_z", str(self.box["center_z"]),
            "--size_x", str(self.box["size_x"]),
            "--size_y", str(self.box["size_y"]),
            "--size_z", str(self.box["size_z"]),
            "--exhaustiveness", str(cfg.exhaustiveness),
            "--num_modes", str(cfg.num_modes),
            "--seed", str(cfg.seed),
            "--out", out_pdbqt,
        ]
        tag = f"vina_{Path(receptor_pdbqt).stem}"
        _run(cmd, cwd=None, timeout=self.timeout, what=tag, log_dir=self.log_dir)
        if not os.path.exists(out_pdbqt):
            raise RuntimeError(f"vina returned 0 but produced no out: {out_pdbqt}")
        with open(out_pdbqt) as f:
            return parse_ligand_poses(f.read())


# =============================================================================
# Orchestrator
# =============================================================================

class EnvelopeFilter:
    """Drive single-point envelope screening over a set of FuncSite-ML sites."""

    def __init__(self, reference: ReferenceEnvelope, config: EnvelopeConfig,
                 mutator: FoldXMutator, converter: PDBToPDBQTConverter,
                 docker: LockedBoxDocker, ligand_pdbqt: str, work_dir: str):
        self.ref = reference
        self.config = config
        self.mutator = mutator
        self.converter = converter
        self.docker = docker
        self.ligand_pdbqt = ligand_pdbqt
        self.work_dir = work_dir
        self.scorer = EnvelopeScorer(reference, config)
        os.makedirs(work_dir, exist_ok=True)

    def calibrate_pose_noise(self, repaired_wt_pdbqt: str,
                             seeds: Sequence[int] = (1, 7, 13)) -> dict:
        """Re-dock WT under several seeds to estimate the pose-RMSD noise floor.

        Use this to choose rmsd_cap honestly: the cap should sit comfortably
        above WT-vs-WT pose scatter, otherwise the protocol rejects on noise.
        """
        ref_pose = self.ref.ligand_pose
        rmsds = []
        for s in seeds:
            cfg = self.config
            saved = cfg.seed
            cfg.seed = s
            out = os.path.join(self.work_dir, f"wt_calib_seed{s}.pdbqt")
            poses = self.docker.dock(repaired_wt_pdbqt, self.ligand_pdbqt, out)
            cfg.seed = saved
            if poses:
                mp = min((p for p in poses if p.get("score") is not None),
                         key=lambda p: p["score"], default=poses[0])
                rmsds.append(ligand_rmsd(ref_pose, mp))
        rmsds = [r for r in rmsds if not np.isnan(r)]
        return {
            "wt_pose_rmsd_mean": float(np.mean(rmsds)) if rmsds else float("nan"),
            "wt_pose_rmsd_max": float(np.max(rmsds)) if rmsds else float("nan"),
            "suggested_rmsd_cap": (float(np.max(rmsds)) * 1.5
                                   if rmsds else self.config.rmsd_cap),
            "n_seeds": len(rmsds),
        }

    def screen_site(self, site_residue_id: str,
                    target_aas: Optional[Sequence[str]] = None
                    ) -> List[EnvelopeResult]:
        """Saturation-scan one site; return one EnvelopeResult per residue."""
        wt3 = re.match(r"([A-Z]{3})", site_residue_id)
        wt1 = AA3TO1.get(wt3.group(1).upper(), "X") if wt3 else "X"
        aas = list(target_aas or self.config.saturation_aa)
        aas = [a for a in aas if a != wt1]  # 19, skip the wild type

        results: List[EnvelopeResult] = []
        for aa in aas:
            try:
                mut_pdb, ddg = self.mutator.build(site_residue_id, aa)
                out_pdbqt = os.path.join(
                    self.work_dir, f"recv_{site_residue_id}_{aa}.pdbqt")
                self.converter.convert(mut_pdb, out_pdbqt)
                mut_receptor = validate_receptor_structure(
                    mut_pdb, "mutant receptor PDB")
                dock_out = os.path.join(
                    self.work_dir, f"dock_{site_residue_id}_{aa}.pdbqt")
                poses = self.docker.dock(out_pdbqt, self.ligand_pdbqt, dock_out)
                results.append(
                    self.scorer.score(site_residue_id, aa,
                                      mut_receptor, poses, ddg))
            except Exception as exc:  # isolate per-mutation failures
                results.append(EnvelopeResult(
                    site_residue_id, aa, False, 0.0, 0.0, float("nan"),
                    999, float("inf"), float("nan"), [f"error:{type(exc).__name__}"]))
        return results

    def screen(self, sites: Sequence[str]) -> pd.DataFrame:
        rows = []
        for site in sites:
            for r in self.screen_site(site):
                rows.append({
                    "site": r.site, "target_aa": r.target_aa,
                    "survived": r.survived, "E": r.E,
                    "contact_retention": r.contact_retention,
                    "pose_rmsd": r.pose_rmsd, "n_new_clash": r.n_new_clash,
                    "dscore": r.dscore, "ddg_guard": r.ddg_guard,
                    "fail_reasons": ";".join(r.fail_reasons),
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # survivors first, then by envelope-preservation score
        return df.sort_values(["survived", "E"], ascending=[False, False]
                              ).reset_index(drop=True)


# =============================================================================
# FuncSite-ML glue: site selection + combined pipeline (sites + mutations)
# =============================================================================

def select_candidate_sites(residue_features: pd.DataFrame,
                           top_k: int = 8,
                           min_function: float = 0.5,
                           catalytic_exclude_prob: float = 0.90,
                           cat_col: str = "catalytic_prob",
                           spec_col: str = "specificity_prob",
                           id_col: str = "residue_id"
                           ) -> Tuple[List[str], Dict[str, float]]:
    """Pick pocket-edge, non-direct-catalytic candidate sites from FuncSite-ML.

    Intent (matches the Phase 3 -> Phase 4 handoff): keep residues that are
    functionally important and near the substrate, but exclude the residues
    that look like the direct catalytic machinery (you do not want to mutate
    the catalytic acid/base). Dual-function and specificity-leaning residues
    -- the kind S139 is -- are exactly what survives here.

    Returns (site_ids, weight_map). weight_map (catalytic_prob per residue)
    is passed to ReferenceEnvelope so catalytic-neighbouring contacts weigh
    more in the retention score.
    """
    df = residue_features.copy()
    for col in (cat_col, spec_col, id_col):
        if col not in df.columns:
            raise ValueError(f"residue_features missing column: {col}")

    # functional importance and a "directly catalytic" guard
    df["_function"] = df[[cat_col, spec_col]].max(axis=1)
    is_direct_catalytic = (df[cat_col] >= catalytic_exclude_prob) & \
                          (df[spec_col] < df[cat_col])

    cand = df[(df["_function"] >= min_function) & (~is_direct_catalytic)]
    cand = cand.sort_values("_function", ascending=False).head(top_k)

    site_ids = cand[id_col].astype(str).tolist()
    weight_map = {str(r[id_col]): float(r[cat_col]) for _, r in cand.iterrows()}
    return site_ids, weight_map


def make_prepare_receptor4_converter(prepare_receptor4: str,
                                     pythonsh: Optional[str] = None
                                     ) -> "PDBToPDBQTConverter":
    """PDB -> receptor PDBQT via AutoDockTools prepare_receptor4.py.

    This is the intended receptor-preparation path for Vina. The input PDB is
    first normalized to strict PDB columns so FoldX BuildModel outputs can be
    consumed reliably by AutoDockTools.
    """
    if not prepare_receptor4:
        raise ValueError("prepare_receptor4 path is required")

    def _convert(inp: str, out: str) -> None:
        clean = str(Path(out).with_suffix(".prepare_receptor4_clean.pdb"))
        standardize_pdb_for_receptor_prep(inp, clean)
        if pythonsh:
            cmd = [pythonsh, prepare_receptor4, "-r", clean, "-o", out,
                   "-A", "hydrogens"]
        elif os.access(prepare_receptor4, os.X_OK):
            cmd = [prepare_receptor4, "-r", clean, "-o", out,
                   "-A", "hydrogens"]
        else:
            raise RuntimeError(
                "prepare_receptor4.py is not executable and pythonsh was not "
                "provided; pass pythonsh=/path/to/pythonsh"
            )
        _run(cmd, cwd=None, timeout=600, what="prepare_receptor4")

    return PDBToPDBQTConverter(fn=_convert)


def run_pipeline(residue_features: pd.DataFrame,
                 ligand_pdbqt: str,
                 box: dict,
                 wt_pdb_for_foldx: str,
                 foldx_bin: str,
                 vina_bin: str,
                 converter: "PDBToPDBQTConverter",
                 work_dir: str,
                 wt_ligand_pose_pdbqt: Optional[str] = None,
                 config: Optional[EnvelopeConfig] = None,
                 sites: Optional[Sequence[str]] = None,
                 select_kwargs: Optional[dict] = None,
                 wt_reference_structure: Optional[str] = None) -> pd.DataFrame:
    """One call: FuncSite-ML scores in, (sites + mutation screen) table out.

    residue_features : FuncSite-ML output (residue_id, catalytic_prob,
                       specificity_prob, ...).
    ligand_pdbqt     : substrate ligand to re-dock into every mutant.
    box              : locked WT MultiOpt box (center_/size_).
    wt_pdb_for_foldx : WT structure in PDB for FoldX RepairPDB/BuildModel.
    wt_ligand_pose_pdbqt: optional WT docked substrate pose PDBQT. If provided,
                       the first parsed pose is used as the contact/score
                       reference and the first 5 poses are used as an RMSD
                       reference ensemble. If omitted, it is generated by
                       docking ligand_pdbqt into the WT receptor prepared from
                       FoldX-repaired WT PDB, and the lowest-score pose is used.
    wt_reference_structure: optional clean WT PDB/PDBQT used only for
                       reference contacts. Defaults to FoldX-repaired WT PDB.
    """
    config = config or EnvelopeConfig()
    os.makedirs(work_dir, exist_ok=True)

    # 1) reference envelope (+ FuncSite-ML weights)
    sel_sites, weight_map = select_candidate_sites(
        residue_features, **(select_kwargs or {}))
    sites = list(sites) if sites is not None else sel_sites
    if not sites:
        raise ValueError("no candidate sites selected; relax select_kwargs")

    # 2) external tools and WT receptor preparation
    repaired = FoldXMutator.repair(foldx_bin, wt_pdb_for_foldx, work_dir)
    wt_receptor_pdbqt = converter.convert(
        repaired, os.path.join(work_dir, "wt_receptor.pdbqt"))

    docker = LockedBoxDocker(vina_bin, box, config,
                             log_dir=os.path.join(work_dir, "vina_logs"))

    user_supplied_wt_pose = wt_ligand_pose_pdbqt is not None
    if wt_ligand_pose_pdbqt is None:
        wt_ligand_pose_pdbqt = os.path.join(work_dir, "wt_reference_pose.pdbqt")
        poses = docker.dock(wt_receptor_pdbqt, ligand_pdbqt, wt_ligand_pose_pdbqt)
        if not poses:
            raise ValueError("WT reference docking produced no pose")

    reference_structure = wt_reference_structure or repaired
    wt_receptor = validate_receptor_structure(
        reference_structure, "WT reference receptor structure")
    with open(wt_ligand_pose_pdbqt) as f:
        wt_poses = parse_ligand_poses(f.read())
    wt_ref_pose = choose_reference_ligand_pose(
        wt_poses, user_supplied=user_supplied_wt_pose)
    wt_ref_pose_ensemble = choose_reference_ligand_pose_ensemble(
        wt_poses, user_supplied=user_supplied_wt_pose)
    reference = ReferenceEnvelope(
        wt_receptor, wt_ref_pose, config, weight_map,
        ligand_pose_ensemble=wt_ref_pose_ensemble)

    mutator = FoldXMutator(foldx_bin, repaired, os.path.join(work_dir, "foldx"))

    # 3) screen
    flt = EnvelopeFilter(reference, config, mutator, converter, docker,
                         ligand_pdbqt, os.path.join(work_dir, "screen"))
    result = flt.screen(sites)

    # 4) merge FuncSite-ML per-site scores onto the screen table
    if not result.empty:
        keep = [c for c in ("residue_id", "catalytic_prob", "specificity_prob")
                if c in residue_features.columns]
        fs = residue_features[keep].rename(columns={"residue_id": "site"})
        result = result.merge(fs, on="site", how="left")
    return result


# =============================================================================
# Self-test: pure-Python geometry / IFP / scoring (no FoldX / Vina needed)
# =============================================================================

def _atom_line(rec: str, serial: int, aname: str, rname: str, chain: str,
               rseq: int, x: float, y: float, z: float) -> str:
    """Column-correct PDB ATOM/HETATM line (element inferred from name)."""
    return (f"{rec:<6}{serial:>5} {aname:<4}{'':1}{rname:>3} "
            f"{chain:1}{rseq:>4}    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}")


def _mini_receptor_pdbqt() -> str:
    # SER139 (polar via OG), ASP140 (ionic via OD1), PHE200 (hydrophobic via CZ)
    return "\n".join([
        _atom_line("ATOM", 1, "CA", "SER", "A", 139, 10.0, 14.5, 10.0),
        _atom_line("ATOM", 2, "OG", "SER", "A", 139, 10.0, 13.0, 10.0),
        _atom_line("ATOM", 3, "CA", "ASP", "A", 140, 14.5, 10.0, 10.0),
        _atom_line("ATOM", 4, "OD1", "ASP", "A", 140, 13.0, 10.0, 10.0),
        _atom_line("ATOM", 5, "CA", "PHE", "A", 200, 16.5, 16.5, 10.0),
        _atom_line("ATOM", 6, "CZ", "PHE", "A", 200, 15.0, 15.0, 10.0),
    ])


def _mini_ligand_pose(score: float, dx: float = 0.0, dy: float = 0.0,
                      dz: float = 0.0, model: int = 1) -> str:
    # N near SER-OG (polar) and ASP-OD1 (ionic); C near PHE-CZ (hydrophobic)
    return "\n".join([
        f"MODEL {model}",
        f"REMARK VINA RESULT:    {score:.1f}      0.000      0.000",
        _atom_line("HETATM", 1, "N", "LIG", "A", 1,
                   10.0 + dx, 10.0 + dy, 10.0 + dz),
        _atom_line("HETATM", 2, "C", "LIG", "A", 1,
                   13.0 + dx, 13.0 + dy, 10.0 + dz),
        "ENDMDL",
    ])


def _selftest() -> None:
    cfg = EnvelopeConfig(rmsd_cap=2.0, retention_floor=0.5, score_band=1.5)

    receptor = parse_receptor_pdbqt(_mini_receptor_pdbqt())
    assert "SER139A" in receptor and "ASP140A" in receptor

    wt_pose = parse_ligand_poses(_mini_ligand_pose(-7.0))[0]
    ref = ReferenceEnvelope(receptor, wt_pose, cfg)
    assert ref.key_weights, "WT should have key contacts"
    assert ref.clash_baseline == 0

    scorer = EnvelopeScorer(ref, cfg)

    # (1) benign mutation: ligand barely moves, contacts kept, mild ddG
    good_pose = parse_ligand_poses(_mini_ligand_pose(-6.8, dx=0.1))[0]
    r_good = scorer.score("SER139A", "W", receptor, [good_pose], ddg_guard=0.8)
    assert r_good.survived, r_good.fail_reasons
    assert r_good.contact_retention == 1.0

    # (2) pose drift: ligand jumps far -> pose_drift + contact loss
    bad_pose = parse_ligand_poses(_mini_ligand_pose(-6.5, dx=6.0, dz=6.0))[0]
    r_bad = scorer.score("SER139A", "K", receptor, [bad_pose], ddg_guard=0.5)
    assert not r_bad.survived
    assert "pose_drift" in r_bad.fail_reasons

    # (3) unfolding guard: fine envelope but ddG huge -> excluded
    r_unf = scorer.score("SER139A", "G", receptor, [good_pose], ddg_guard=5.0)
    assert not r_unf.survived and "unfolding" in r_unf.fail_reasons

    # (4) clash: drop a ligand atom on top of ASP OD1 (13,10,10)
    clash = parse_ligand_poses(_mini_ligand_pose(-7.0))[0]
    clash["atoms"].append(("N2", "N", [13.0, 10.0, 10.0]))
    r_clash = scorer.score("SER139A", "W", receptor, [clash], ddg_guard=0.5)
    assert r_clash.n_new_clash >= 1 and "new_clash" in r_clash.fail_reasons

    # (5) RMSD frame-locked + name-based (robust to atom reorder)
    assert abs(ligand_rmsd(wt_pose, wt_pose)) < 1e-9
    assert ligand_rmsd(wt_pose, bad_pose) > 2.0
    reordered = {"model": 1, "score": -7.0,
                 "atoms": list(reversed(wt_pose["atoms"]))}
    assert abs(ligand_rmsd(wt_pose, reordered)) < 1e-9  # name match beats order
    assert abs(ligand_rmsd(wt_pose, reordered, symmetry_safe=True)) < 1e-9

    # (6) ddG missing must NOT pass silently
    r_nan = scorer.score("SER139A", "W", receptor, [good_pose],
                         ddg_guard=float("nan"))
    assert not r_nan.survived and "ddg_missing" in r_nan.fail_reasons

    # (7) mutation-string mapping
    assert FoldXMutator.mutation_string("SER139A", "W") == "SA139W;"

    # (8) FuncSite-ML site selection (glue)
    rf = pd.DataFrame({
        "residue_id": ["SER139A", "ASP140A", "GLU200A", "ALA10A"],
        "catalytic_prob": [0.62, 0.95, 0.30, 0.05],   # ASP140 looks catalytic
        "specificity_prob": [0.81, 0.20, 0.10, 0.04],
    })
    sites, wmap = select_candidate_sites(rf, top_k=3, min_function=0.5)
    assert "SER139A" in sites          # dual/spec-leaning -> kept
    assert "ASP140A" not in sites      # direct-catalytic -> excluded
    assert wmap["SER139A"] == 0.62

    print("envelope_filter self-test: ALL PASS")
    print(f"  key residues (WT): {sorted(ref.key_weights)}")
    print(f"  benign  E={r_good.E}  retention={r_good.contact_retention}")
    print(f"  drift   reasons={r_bad.fail_reasons}")
    print(f"  unfold  reasons={r_unf.fail_reasons}")
    print(f"  clash   new_clash={r_clash.n_new_clash}")
    print(f"  ddg-nan reasons={r_nan.fail_reasons}")
    print(f"  selected sites={sites}")


def _build_cli():
    import argparse
    p = argparse.ArgumentParser(
        description="Envelope-preservation single-point mutation screen "
                    "(runs after FuncSite-ML).")
    p.add_argument("--selftest", action="store_true",
                   help="run the pure-Python self-test and exit")
    p.add_argument("--funcsite-csv",
                   help="FuncSite-ML residue_features CSV (residue_id, "
                        "catalytic_prob, specificity_prob)")
    p.add_argument("--wt-reference-structure",
                   help="clean WT PDB/PDBQT used only for reference contacts; "
                        "defaults to FoldX-repaired WT PDB")
    p.add_argument("--wt-ligand-pose",
                   help="optional WT docked substrate pose PDBQT; MODEL 1 is "
                        "used for contacts/score and MODEL 1-5 for RMSD. If "
                        "omitted, it is generated by docking --ligand into "
                        "prepared WT")
    p.add_argument("--ligand", help="substrate ligand PDBQT to re-dock")
    p.add_argument("--wt-pdb", help="WT structure PDB for FoldX")
    p.add_argument("--box", help="Vina box config file (center_/size_)")
    p.add_argument("--foldx-bin", default="foldx")
    p.add_argument("--vina-bin", default="vina")
    p.add_argument("--converter-cmd",
                   help='PDB->PDBQT command template with {inp} {out}; '
                        'advanced override; prefer --prepare-receptor4')
    p.add_argument("--prepare-receptor4",
                   help="AutoDockTools prepare_receptor4.py path")
    p.add_argument("--pythonsh",
                   help="MGLTools pythonsh path used to run prepare_receptor4.py")
    p.add_argument("--sites", help="comma-separated residue_ids (override "
                                   "auto-selection), or a .txt file")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--out", default="envelope_screen.csv")
    p.add_argument("--work-dir", default="envelope_work")
    # threshold overrides
    p.add_argument("--rmsd-cap", type=float)
    p.add_argument("--retention-floor", type=float)
    p.add_argument("--score-band", type=float)
    p.add_argument("--ddg-guard-line", type=float)
    p.add_argument("--exhaustiveness", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--symmetry-safe-rmsd", action="store_true")
    p.add_argument("--allow-missing-ddg", action="store_true",
                   help="do NOT exclude on missing ddG (not recommended)")
    return p


def _cli_main():
    args = _build_cli().parse_args()
    if args.selftest or args.funcsite_csv is None:
        _selftest()
        return

    required = ["ligand", "wt_pdb", "box"]
    missing = [r for r in required if getattr(args, r) is None]
    if missing:
        raise SystemExit(f"missing required args: {missing}")
    if args.converter_cmd is None and args.prepare_receptor4 is None:
        raise SystemExit(
            "missing required arg: --prepare-receptor4 "
            "(or provide --converter-cmd as an advanced override)"
        )

    cfg = EnvelopeConfig()
    for attr in ("rmsd_cap", "retention_floor", "score_band",
                 "ddg_guard_line", "exhaustiveness", "seed"):
        v = getattr(args, attr)
        if v is not None:
            setattr(cfg, attr, v)
    if args.symmetry_safe_rmsd:
        cfg.symmetry_safe_rmsd = True
    if args.allow_missing_ddg:
        cfg.fail_on_missing_ddg = False

    residue_features = pd.read_csv(args.funcsite_csv)
    with open(args.box) as f:
        box = parse_vina_box(f.read())

    if args.converter_cmd:
        converter = PDBToPDBQTConverter(command_template=args.converter_cmd)
    else:
        converter = make_prepare_receptor4_converter(
            args.prepare_receptor4, pythonsh=args.pythonsh)

    sites = None
    if args.sites:
        if os.path.exists(args.sites):
            with open(args.sites) as f:
                sites = [s.strip() for s in f.read().split() if s.strip()]
        else:
            sites = [s.strip() for s in args.sites.split(",") if s.strip()]

    df = run_pipeline(
        residue_features=residue_features,
        ligand_pdbqt=args.ligand,
        box=box,
        wt_pdb_for_foldx=args.wt_pdb,
        foldx_bin=args.foldx_bin,
        vina_bin=args.vina_bin,
        converter=converter,
        work_dir=args.work_dir,
        wt_ligand_pose_pdbqt=args.wt_ligand_pose,
        config=cfg,
        sites=sites,
        select_kwargs={"top_k": args.top_k},
        wt_reference_structure=args.wt_reference_structure,
    )
    df.to_csv(args.out, index=False)
    n_keep = int(df["survived"].sum()) if not df.empty else 0
    print(f"wrote {args.out}: {len(df)} mutations, {n_keep} survived")


if __name__ == "__main__":
    _cli_main()
