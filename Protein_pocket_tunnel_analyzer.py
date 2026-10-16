#!/usr/bin/env python3

import os
import re
import time
import json
import zipfile
import subprocess
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Union
import shutil

# Active-site specification accepted by analyze()/run_fpocket()/run_caver():
#   "84,153,224"                 same residue numbers for every structure (any chain)
#   "A:84,A:153,B:30A"           chain-qualified, insertion codes allowed
#   {"WT": "84,153", "MUT1": ...} per structure (key = file stem); key "*" = default
#   "/path/to/active_sites.json" a JSON file holding such a dict
ActiveSiteSpec = Union[str, Dict[str, Union[str, List]], None]


class ProteinAnalyzer:
    def __init__(self, work_dir: str = "/content/protein_analysis"):

        self.work_dir = Path(work_dir)
        self.input_dir = self.work_dir / "input"
        self.output_dir = self.work_dir / "output"
        self.p2rank_output = self.output_dir / "p2rank"
        self.caver_output = self.output_dir / "caver"
        self.fpocket_output = self.output_dir / "fpocket"
        self.integrated_output = self.output_dir / "integrated_summary"

        self.p2rank_path = self.work_dir / "p2rank_2.4.2"
        self.caver_path = self.work_dir

        # fpocket is compiled lazily on first use and cached here
        self.fpocket_src = Path("/content/fpocket_src")
        self.fpocket_bin: Optional[str] = None

        # sanitized stem -> original upload stem (so per-structure active sites
        # can be keyed by either name)
        self._stem_alias: Dict[str, str] = {}
        # input stem -> CAVER group name (identical unless frames are grouped)
        self._caver_group_of: Dict[str, str] = {}

        self._create_directories()
        
    def _create_directories(self):
        for directory in [self.input_dir, self.output_dir,
                          self.p2rank_output, self.caver_output,
                          self.fpocket_output, self.integrated_output]:
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Input handling: sanitize file names so spaces / parentheses in
    # uploaded names (e.g. "WT (2).pdb") never break P2Rank / CAVER.
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_stem(path) -> str:
        """Filesystem/CLI-safe stem: only [A-Za-z0-9_.-], no leading/trailing junk."""
        stem = Path(path).stem
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.")
        return stem or "structure"

    @staticmethod
    def _pdb_has_atoms(pdb_path) -> bool:
        try:
            with open(pdb_path, "r", errors="ignore") as f:
                for line in f:
                    if line.startswith("ATOM"):
                        return True
        except Exception:
            pass
        return False

    @staticmethod
    def _split_models(pdb_path) -> List[List[str]]:
        """Coordinate lines (ATOM/HETATM/TER) per MODEL. A file without MODEL
        records is one model."""
        models: List[List[str]] = []
        cur: Optional[List[str]] = None
        with open(pdb_path, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("MODEL"):
                    cur = []
                    models.append(cur)
                elif line.startswith("ENDMDL"):
                    cur = None
                elif line.startswith(("ATOM", "HETATM", "TER")):
                    if cur is None:            # coordinates outside any MODEL block
                        cur = []
                        models.append(cur)
                    cur.append(line if line.endswith("\n") else line + "\n")
        return [m for m in models if any(l.startswith("ATOM") for l in m)]

    def prepare_inputs(self, uploaded_paths, clear_existing: bool = True,
                       multi_model: str = "first") -> List[Path]:
        """
        Copy uploaded structures into input_dir with sanitized names.
        Drops files with no ATOM records and avoids duplicate-name collisions.
        Returns the clean list of PDB paths the pipeline will analyze.

        Multi-MODEL files (docking poses, NMR ensembles) are NOT passed through
        as-is: written flat, every model's atoms end up superposed in one
        structure and fpocket analyses N overlapping copies of the protein.
          multi_model="first": keep model 1 only (default)
          multi_model="split": one file per model, <stem>_m001, <stem>_m002, ...
        Every output file holds exactly one model.
        """
        if multi_model not in ("first", "split"):
            raise ValueError("multi_model must be 'first' or 'split'")
        if clear_existing:
            for old in self.input_dir.glob("*.pdb"):
                old.unlink(missing_ok=True)
            self._stem_alias = {}

        clean_files: List[Path] = []
        used = set()

        def _unique(stem):
            candidate, i = stem, 2
            while candidate in used:           # de-duplicate cleanly (no "(2)")
                candidate = f"{stem}_{i}"
                i += 1
            used.add(candidate)
            return candidate

        for src in uploaded_paths:
            src = Path(src)
            if not self._pdb_has_atoms(src):
                print(f"   ⚠️ Skipping {src.name}: no ATOM records")
                continue

            models = self._split_models(src)
            stem = self._safe_stem(src)
            if len(models) > 1:
                print(f"   ⚠️ {src.name}: {len(models)} MODELs -> "
                      + ("keeping model 1 only" if multi_model == "first" else "splitting into one file per model"))
            todo = [(stem, models[0])] if (multi_model == "first" or len(models) == 1) else \
                   [(f"{stem}_m{k:03d}", m) for k, m in enumerate(models, 1)]

            for out_stem, lines in todo:
                candidate = _unique(out_stem)
                dst = self.input_dir / f"{candidate}.pdb"
                with open(dst, "w") as f:
                    f.writelines(lines)
                    f.write("END\n")
                clean_files.append(dst)
                self._stem_alias[candidate] = src.stem
                label = src.name if src.name == dst.name else f"{src.name} → {dst.name}"
                print(f"   • {label}")

        return sorted(clean_files)

    # ------------------------------------------------------------------
    # Active-site specification (shared by fpocket hit counting, P2Rank
    # pocket lookup and the CAVER starting point)
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_residue_tokens(spec) -> Set[Tuple[Optional[str], str]]:
        """'84, A:153, B:30A' or [84, 'A:153'] -> {(None,'84'), ('A','153'), ('B','30A')}."""
        if spec is None:
            return set()
        tokens = spec if isinstance(spec, (list, tuple, set)) else str(spec).replace(";", ",").split(",")
        out = set()
        for t in tokens:
            t = str(t).strip()
            if not t:
                continue
            chain, num = (t.split(":", 1) if ":" in t else (None, t))
            chain = chain.strip() if chain else None
            num = num.strip().upper()
            if not re.fullmatch(r"-?\d+[A-Z]?", num):
                raise ValueError(f"Unrecognised active-site residue '{t}' (use 84, A:84 or A:84B)")
            out.add((chain or None, num))
        return out

    @staticmethod
    def load_active_site_file(path) -> Dict[str, object]:
        """Read a per-structure active-site file.
        .json : {"stem": "84,153"} or {"stem": ["A:84", 153]} or
                {"stem": {"resids": [...], ...}} (anchors.json style)
        .csv  : columns structure,residues   e.g.  WT,"84,153,224"
        """
        path = Path(path)
        if path.suffix.lower() == ".json":
            with open(path) as f:
                return json.load(f)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, dtype=str)
            cols = {c.lower().strip(): c for c in df.columns}
            s, r = cols.get("structure"), cols.get("residues")
            if not s or not r:
                raise ValueError(f"{path.name}: needs columns 'structure' and 'residues'")
            return {str(a).strip(): str(b) for a, b in zip(df[s], df[r]) if str(b).strip() not in ("", "nan")}
        raise ValueError(f"Unsupported active-site file: {path.name} (use .json or .csv)")

    def _active_site_for(self, active_site_resids: ActiveSiteSpec, stem: str) -> Set[Tuple[Optional[str], str]]:
        """Active-site residues for one structure, from a global string, a per-stem
        dict, or a JSON/CSV file of such a dict. 'auto' and '' mean "none given"."""
        spec = active_site_resids
        if isinstance(spec, (str, Path)) and str(spec).lower().endswith((".json", ".csv")) and Path(spec).exists():
            spec = self.load_active_site_file(spec)
        if isinstance(spec, dict):
            for key in (stem, self._stem_alias.get(stem), "*"):
                if key is not None and key in spec:
                    val = spec[key]
                    if isinstance(val, dict):
                        val = val.get("resids", val.get("residues", ""))
                    return self._parse_residue_tokens(val)
            return set()
        if isinstance(spec, str) and spec.strip().lower() == "auto":
            return set()
        return self._parse_residue_tokens(spec)

    @staticmethod
    def _residue_matches(chain: str, resid: str, targets) -> bool:
        """resid includes insertion code ('84', '84A'); match is exact."""
        resid = resid.strip().upper()
        return any(resid == num and (c is None or c == chain) for c, num in targets)

    # ==================================================================
    # Active-site auto-detection (used when no residues are supplied)
    # ==================================================================
    # Evidence is tried in order of reliability; every candidate is kept so the
    # choice can be reviewed (active_sites_auto.json):
    #   ligand      residues within 4.5 A of a bound ligand (holo / docked input)
    #   metal       protein donors coordinating a metal ion
    #   motif       catalytic constellations found geometrically or by sequence
    #               (MOTIF_PRIORITY below; extend it for other enzyme families)
    #   (none)      engine falls back to P2Rank's top pocket, flagged as warning
    # Within a tier, candidates lying in a better-ranked P2Rank pocket and with
    # higher model confidence (pLDDT in the B-factor column) are preferred.
    WATERS = {"HOH", "WAT", "DOD", "H2O", "SOL", "TIP", "TIP3"}
    METAL_ELEMENTS = {"ZN", "MG", "CA", "MN", "FE", "CO", "NI", "CU", "CD", "NA", "K"}
    # crystallisation additives / buffers that are not substrates
    ADDITIVES = {"SO4", "PO4", "GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "ACT", "ACY", "MPD",
                 "DMS", "FMT", "TRS", "BME", "IOD", "CL", "BR", "NO3", "EPE", "MES", "CIT",
                 "IPA", "EOH", "MOH", "NH4", "SCN", "AZI", "IMD", "TLA", "BOG", "LDA"}
    # donor atoms that can coordinate a metal (Asn ND2 / Gln NE2 are amide N: excluded)
    METAL_DONORS = {("HIS", "ND1"), ("HIS", "NE2"), ("ASP", "OD1"), ("ASP", "OD2"),
                    ("GLU", "OE1"), ("GLU", "OE2"), ("CYS", "SG"), ("ASN", "OD1"),
                    ("GLN", "OE1"), ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"), ("MET", "SD")}
    MOTIF_PRIORITY = {
        # name                      base score   what it marks
        "Ser/Cys-His-Asp/Glu triad": 70,        # serine/cysteine hydrolases, proteases, lipases
        "HKD x2":                    68,        # PLD superfamily, cardiolipin synthases, nucleases
        "HKD x1":                    50,
        "GxSxG nucleophile + acid":  45,        # patatin-like / alpha-beta hydrolase without His in range
        "Ser-Asp/Glu H-bonded dyad": 40,        # patatin / SGNH-type dyads; common by chance, so only
                                                # accepted inside a top-3 P2Rank pocket
    }
    DYAD_MAX_POCKET_RANK = 3
    TIER_SCORE = {"ligand": 90, "metal": 60}

    def _read_atoms(self, pdb_file) -> List[Dict]:
        atoms = []
        with open(pdb_file, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("ENDMDL"):
                    break
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                name = line[12:16].strip()
                el = (line[76:78].strip() or re.sub(r"[^A-Za-z]", "", name)[:1]).upper()
                if el in ("H", "D"):
                    continue
                try:
                    b = float(line[60:66])
                except ValueError:
                    b = float("nan")
                atoms.append(dict(het=line.startswith("HETATM"), name=name, resn=line[17:20].strip(),
                                  chain=line[21].strip(),
                                  resid=(line[22:26].strip() + line[26].strip()).upper(),
                                  el=el, b=b,
                                  xyz=(float(line[30:38]), float(line[38:46]), float(line[46:54]))))
        return atoms

    @staticmethod
    def _dist(a, b) -> float:
        return sum((a[k] - b[k]) ** 2 for k in range(3)) ** 0.5

    @staticmethod
    def _looks_like_plddt(atoms) -> bool:
        bs = [a["b"] for a in atoms if not a["het"] and a["b"] == a["b"]]
        return bool(bs) and min(bs) >= 0 and max(bs) <= 100 and sum(b > 50 for b in bs) > 0.5 * len(bs)

    def _detect_ligand_sites(self, atoms) -> List[Dict]:
        groups: Dict[Tuple, List[Dict]] = {}
        for a in atoms:
            if a["het"] and a["resn"] not in self.WATERS and a["resn"] not in self.ADDITIVES \
                    and a["el"] not in self.METAL_ELEMENTS:
                groups.setdefault((a["chain"], a["resid"], a["resn"]), []).append(a)
        prot = [a for a in atoms if not a["het"]]
        out = []
        for (ch, ri, rn), lig in groups.items():
            if len(lig) < 6:                      # too small to define a binding site
                continue
            near = {(p["chain"], p["resid"]) for p in prot
                    if min(self._dist(p["xyz"], l["xyz"]) for l in lig) <= 4.5}
            if near:
                out.append(dict(source="ligand", name=f"ligand {rn} {ch}:{ri} ({len(lig)} heavy atoms)",
                                residues=sorted(near), score=self.TIER_SCORE["ligand"] + min(len(lig), 60) / 10))
        return out

    def _detect_metal_sites(self, atoms) -> List[Dict]:
        out = []
        donors = [a for a in atoms if not a["het"] and (a["resn"], a["name"]) in self.METAL_DONORS]
        for m in atoms:
            if not (m["het"] and m["el"] in self.METAL_ELEMENTS and m["el"] not in ("NA", "K")):
                continue
            cut = 3.0 if m["el"] in ("CA", "MG") else 2.8
            lig = {(d["chain"], d["resid"]): d["resn"] for d in donors if self._dist(d["xyz"], m["xyz"]) <= cut}
            if len(lig) >= 2:
                out.append(dict(source="metal", name=f"{m['el']} {m['chain']}:{m['resid']} site, {len(lig)} protein donors",
                                residues=sorted(lig), score=self.TIER_SCORE["metal"] + len(lig)))
        return out

    @staticmethod
    def _chain_sequences(atoms) -> Dict[str, List[Tuple[str, str]]]:
        """chain -> [(resid, one-letter)] in file order, from CA atoms."""
        three = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
                 "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
                 "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
        seqs: Dict[str, List[Tuple[str, str]]] = {}
        for a in atoms:
            if not a["het"] and a["name"] == "CA":
                seqs.setdefault(a["chain"], []).append((a["resid"], three.get(a["resn"], "X")))
        return seqs

    def _detect_motifs(self, atoms) -> List[Dict]:
        prot = [a for a in atoms if not a["het"]]
        by_res: Dict[Tuple[str, str], Dict[str, Dict]] = {}
        for a in prot:
            by_res.setdefault((a["chain"], a["resid"]), {})[a["name"]] = a
        resn = {k: next(iter(v.values()))["resn"] for k, v in by_res.items()}
        his_n = [(k, v[n]) for k, v in by_res.items() if resn[k] == "HIS" for n in ("ND1", "NE2") if n in v]
        acid_o = [(k, v[n]) for k, v in by_res.items() for n in ("OD1", "OD2", "OE1", "OE2")
                  if n in v and resn[k] in ("ASP", "GLU")]
        out = []

        # 1) Ser/Cys - His - Asp/Glu triad (geometry only, sequence-order free)
        for k, v in by_res.items():
            nu = v.get("OG") if resn[k] == "SER" else v.get("SG") if resn[k] == "CYS" else None
            if nu is None:
                continue
            cut1 = 4.0 if resn[k] == "SER" else 4.3
            for hk, hn in his_n:
                d1 = self._dist(nu["xyz"], hn["xyz"])
                if d1 > cut1:
                    continue
                other = by_res[hk].get("ND1" if hn["name"] == "NE2" else "NE2")
                if other is None:
                    continue
                best = min(((self._dist(other["xyz"], o["xyz"]), ak) for ak, o in acid_o), default=None)
                if best and best[0] <= 3.6:
                    d2, ak = best
                    out.append(dict(source="motif", name="Ser/Cys-His-Asp/Glu triad",
                                    residues=sorted({k, hk, ak}),
                                    detail=f"{resn[k]}{k[1]}-{hn['name']}:{d1:.2f} A, HIS{hk[1]}-{resn[ak]}{ak[1]}:{d2:.2f} A",
                                    score=self.MOTIF_PRIORITY["Ser/Cys-His-Asp/Glu triad"] - (d1 + d2)))

        seqs = self._chain_sequences(atoms)
        triad_ser = {r for c in out for r in c["residues"] if resn.get(r) in ("SER", "CYS")}
        for ch, seq in seqs.items():
            s = "".join(x for _, x in seq)
            # 2) HKD motif(s): H-x-K-x(4)-D
            hkd = [m.start() for m in re.finditer(r"(?=H.K....D)", s)]
            if hkd:
                his = [(ch, seq[i][0]) for i in hkd]
                hits = [(ch, seq[i + j][0]) for i in hkd for j in (0, 2, 7)]
                name = "HKD x2" if len(hkd) >= 2 else "HKD x1"
                detail = "HxK(x4)D at " + ", ".join(f"H{seq[i][0]}" for i in hkd)
                if len(hkd) >= 2:
                    ne = [by_res[h].get("NE2") for h in his[:2]]
                    if all(ne):
                        detail += f"; His-His NE2 {self._dist(ne[0]['xyz'], ne[1]['xyz']):.1f} A"
                out.append(dict(source="motif", name=name, residues=sorted(set(hits)), detail=detail,
                                score=self.MOTIF_PRIORITY[name]))
            # 3) GxSxG nucleophile elbow whose Ser has no His partner: nearest acid
            for m in re.finditer(r"(?=G.S.G)", s):
                sk = (ch, seq[m.start() + 2][0])
                og = by_res.get(sk, {}).get("OG")
                if og is None or sk in triad_ser:
                    continue
                best = min(((self._dist(og["xyz"], o["xyz"]), ak) for ak, o in acid_o), default=None)
                if best and best[0] <= 7.0:
                    out.append(dict(source="motif", name="GxSxG nucleophile + acid",
                                    residues=sorted({sk, best[1]}),
                                    detail=f"GxSxG Ser{sk[1]}, nearest {resn[best[1]]}{best[1][1]} carboxylate {best[0]:.2f} A",
                                    score=self.MOTIF_PRIORITY["GxSxG nucleophile + acid"] - best[0] / 2))

        # 4) Ser OG directly H-bonded to an Asp/Glu carboxylate (<= 3.2 A). Frequent by
        #    chance, so detect_active_sites() keeps it only inside a top-ranked P2Rank pocket.
        claimed = {r for c in out for r in c["residues"]}
        for k, v in by_res.items():
            og = v.get("OG") if resn[k] == "SER" else None
            if og is None or k in claimed:
                continue
            best = min(((self._dist(og["xyz"], o["xyz"]), ak) for ak, o in acid_o), default=None)
            if best and best[0] <= 3.2:
                out.append(dict(source="motif", name="Ser-Asp/Glu H-bonded dyad", requires_pocket=True,
                                residues=sorted({k, best[1]}),
                                detail=f"SER{k[1]} OG - {resn[best[1]]}{best[1][1]} {best[0]:.2f} A",
                                score=self.MOTIF_PRIORITY["Ser-Asp/Glu H-bonded dyad"] - best[0]))
        return out

    def detect_active_sites(self, pdb_files, p2rank_results: Optional[Dict[str, Dict]] = None,
                            given: ActiveSiteSpec = "") -> Dict[str, Dict]:
        """
        For every structure without user-supplied residues, collect candidate
        active sites and pick one. Returns stem -> record with keys
          resids      "A:125,A:181,A:214"   (empty if nothing found)
          source      user | ligand | metal | motif | none
          name, detail, confidence (high/medium/low), plddt_mean,
          p2rank_rank (best P2Rank pocket containing a candidate residue),
          alternatives  the other candidates, best first
        """
        p2rank_results = p2rank_results or {}
        report: Dict[str, Dict] = {}
        for pdb in pdb_files:
            pdb = Path(pdb)
            stem = pdb.stem
            user = self._active_site_for(given, stem)
            if user:
                report[stem] = dict(resids=",".join(f"{c + ':' if c else ''}{n}" for c, n in sorted(
                    user, key=lambda t: (t[0] or "", t[1]))), source="user", name="supplied",
                    confidence="user", detail="", alternatives=[])
                continue
            atoms = self._read_atoms(pdb)
            plddt = self._looks_like_plddt(atoms)
            bres: Dict[Tuple[str, str], float] = {}
            for a in atoms:
                if not a["het"]:
                    bres.setdefault((a["chain"], a["resid"]), a["b"])
            cands = self._detect_ligand_sites(atoms) + self._detect_metal_sites(atoms) + self._detect_motifs(atoms)
            pockets = p2rank_results.get(stem, {}).get("pockets", [])
            for c in cands:
                tset = {(ch, r) for ch, r in c["residues"]}
                rank = next((p["rank"] for p in pockets
                             if any(self._residue_matches(ch, r, tset) for ch, r in self._p2rank_pocket_residues(p))), None)
                c["p2rank_rank"] = rank
                c["plddt_mean"] = round(sum(bres.get(r, 0) for r in c["residues"]) / len(c["residues"]), 1) if plddt else None
                c["rank_score"] = c["score"] + (10.0 / rank if rank else -5.0) \
                    - (15.0 if (c["plddt_mean"] is not None and c["plddt_mean"] < 70) else 0.0)
            cands = [c for c in cands if not c.get("requires_pocket")
                     or (c["p2rank_rank"] and c["p2rank_rank"] <= self.DYAD_MAX_POCKET_RANK)]
            cands.sort(key=lambda c: -c["rank_score"])
            fmt = lambda c: ",".join(f"{ch}:{r}" if ch else r for ch, r in c["residues"])
            if cands:
                best = cands[0]
                conf = "high" if best["source"] == "ligand" or (best["source"] == "motif" and best["p2rank_rank"] == 1) \
                    else "medium" if best["p2rank_rank"] else "low"
                if best.get("requires_pocket"):
                    conf = "low"                      # chance-prone motif: never more than low
                if len(cands) > 1 and cands[1]["source"] == best["source"] and \
                        cands[1]["rank_score"] > best["rank_score"] - 2 and set(cands[1]["residues"]) != set(best["residues"]):
                    conf = "low"                      # near-tie between different sites
                report[stem] = dict(resids=fmt(best), source=best["source"], name=best["name"],
                                    detail=best.get("detail", ""), confidence=conf,
                                    plddt_mean=best["plddt_mean"], p2rank_rank=best["p2rank_rank"],
                                    alternatives=[dict(resids=fmt(c), source=c["source"], name=c["name"],
                                                       detail=c.get("detail", ""), p2rank_rank=c["p2rank_rank"],
                                                       plddt_mean=c["plddt_mean"], score=round(c["rank_score"], 2))
                                                  for c in cands[1:8]])
            else:
                report[stem] = dict(resids="", source="none", name="no candidate", confidence="none",
                                    detail="no ligand, metal site or known catalytic motif; "
                                           "CAVER will start at P2Rank's top pocket", alternatives=[])
            r = report[stem]
            print(f"   • {stem}: [{r['source']}/{r['confidence']}] {r['name']} -> {r['resids'] or '-'}")
        return report

    def setup_environment(self):
        print("=" * 70)
        print("🔧 Setting up environment (P2Rank 2.4.2 + CAVER 3.0.2)...")
        print("=" * 70)
        
        self._install_java()
        self._download_p2rank()
        self._download_caver()
        
        print("\n✅ Environment setup completed!")
        print("=" * 70)
    
    def _install_java(self):
        print("\n☕ Installing Java...")
        try:
            result = subprocess.run(["java", "-version"],
                                    capture_output=True,
                                    text=True)
            if result.returncode == 0:
                print("✅ Java already installed")
                return
        except FileNotFoundError:
            pass
        
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "openjdk-11-jdk"], check=True)
        print("✅ Java installed successfully")

    def _download_p2rank(self):
        print("\n📦 Downloading P2Rank v2.4.2")

        if self.p2rank_path.exists():
            print("✅ P2Rank already exists")
            return

        url = "https://github.com/rdk/p2rank/releases/download/2.4.2/p2rank_2.4.2.tar.gz"
        tar_path = self.work_dir / "p2rank.tar.gz"

        print(f"⏳ Downloading P2Rank from official GitHub")
        try:
            subprocess.run(
                ["wget", "-O", str(tar_path), url],
                check=True,
                timeout=600
            )
        except Exception as e:
            raise RuntimeError(f"❌ Failed to download P2Rank: {e}")

        if not tar_path.exists() or os.path.getsize(tar_path) < 50_000_000:
            raise RuntimeError("❌ Downloaded P2Rank file is too small, possibly failed.")

        print("📦 Extracting P2Rank...")
        try:
            subprocess.run(
                ["tar", "-xzf", str(tar_path), "-C", str(self.work_dir)],
                check=True
            )
        except Exception as e:
            raise RuntimeError(f"❌ Failed to extract P2Rank: {e}")

        tar_path.unlink(missing_ok=True)
        print("✅ P2Rank downloaded and extracted successfully")

    def _download_caver(self):
        print("\n📦 Downloading CAVER 3.0.2 (ZIP version)...")

        existing = list(self.work_dir.rglob("caver.jar"))
        if existing:
            self.caver_path = existing[0].parent
            print(f"✅ CAVER already exists at: {self.caver_path}")
            return

        caver_url = "https://www.caver.cz/fil/download/caver30/302/caver_3.0.2.zip"
        zip_path = self.work_dir / "caver.zip"

        try:
            subprocess.run(
                ["wget", "-q", "-O", str(zip_path), caver_url],
                check=True,
                timeout=300
            )
        except Exception as e:
            raise RuntimeError(f"❌ Failed to download CAVER 3.0.2: {e}")

        if os.path.getsize(zip_path) < 50_000:
            raise RuntimeError("❌ CAVER zip too small, download likely failed.")

        print("📦 Extracting CAVER...")
        subprocess.run(["unzip", "-o", str(zip_path), "-d", str(self.work_dir)],
                       check=True)
        zip_path.unlink(missing_ok=True)

        jar_files = list(self.work_dir.rglob("caver.jar"))
        if not jar_files:
            raise FileNotFoundError("❌ caver.jar not found after extraction")

        self.caver_path = jar_files[0].parent
        print(f"   ✓ Found caver.jar at: {self.caver_path}")
        print("✅ CAVER 3.0.2 installed successfully")

    def run_p2rank(self, pdb_files: List[str] = None,
                   min_score: float = 0.0,
                   threads: int = 2) -> Dict[str, Dict]:

        print("\n" + "=" * 70)
        print("🔍 Running P2Rank Analysis...")
        print("=" * 70)
        
        if pdb_files is None:
            pdb_files = list(self.input_dir.glob("*.pdb"))
        else:
            pdb_files = [Path(f) for f in pdb_files]
        
        if not pdb_files:
            print("⚠️  No PDB files found!")
            return {}
        
        results: Dict[str, Dict] = {}
        
        for i, pdb_file in enumerate(pdb_files, 1):
            print(f"\n[{i}/{len(pdb_files)}] 📊 Analyzing {pdb_file.name}...")

            prank_script = self.p2rank_path / "prank"
            output_subdir = self.p2rank_output / pdb_file.stem
            
            cmd = [
                str(prank_script),
                "predict",
                "-f", str(pdb_file),
                "-o", str(output_subdir),
                "-threads", str(threads)
            ]

            # NOTE: P2Rank's `predict` has NO -min_score parameter (passing it
            # raises "Invalid parameter name: min_score"). Score filtering is
            # therefore applied as POST-PROCESSING on the parsed pockets below.
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                
                if result.returncode == 0:
                    print("   ✅ P2Rank completed")
                    pocket_info = self._parse_p2rank_results(
                        output_subdir, pdb_file.stem, min_score=min_score)
                    results[pdb_file.stem] = pocket_info
                else:
                    print("   ❌ P2Rank failed")
                    print(result.stderr[:400])
                    results[pdb_file.stem] = {"error": result.stderr}
                    
            except subprocess.TimeoutExpired:
                print("   ⏱️ Timeout (>600s)")
                results[pdb_file.stem] = {"error": "Timeout"}
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[pdb_file.stem] = {"error": str(e)}
        
        print("\n" + "=" * 70)
        print("✅ P2Rank analysis completed!")
        print("=" * 70)
        
        return results

    def _parse_p2rank_results(self, output_dir: Path, pdb_name: str,
                              min_score: float = 0.0) -> Dict:
        pocket_info: Dict[str, object] = {
            "pdb_name": pdb_name,
            "pockets": [],
            "summary": {}
        }
        
        csv_files = list(output_dir.glob("*predictions.csv")) or list(output_dir.glob("*.csv"))
        if not csv_files:
            print(f"      ⚠️  No CSV file found in {output_dir}")
            return pocket_info
        
        try:
            df = pd.read_csv(csv_files[0], skipinitialspace=True)

            all_pockets = []
            for idx, row in df.iterrows():
                pocket = {
                    "name": str(row.get("name", f"pocket{idx+1}")),
                    "rank": int(row.get("rank", idx + 1)),
                    "score": float(row.get("score", 0.0)),
                    "probability": float(row.get("probability", 0.0)),
                    "center_x": float(row.get("center_x", 0.0)),
                    "center_y": float(row.get("center_y", 0.0)),
                    "center_z": float(row.get("center_z", 0.0)),
                    "residue_ids": str(row.get("residue_ids", "")).strip()
                }
                all_pockets.append(pocket)

            # Post-processing score filter (replaces the invalid -min_score CLI flag)
            kept = [p for p in all_pockets if p["score"] >= min_score]
            pocket_info["pockets"] = kept
            pocket_info["summary"] = {
                "total_pockets": len(kept),
                "pockets_before_filter": len(all_pockets),
                "min_score": min_score,
                "output_file": str(csv_files[0])
            }

            if min_score > 0:
                print(f"      ✓ Score filter ≥ {min_score}: kept {len(kept)}/{len(all_pockets)} pockets")

            if kept:
                top = kept[0]
                print(
                    f"      ✓ Top pocket: score={top['score']:.2f}, "
                    f"prob={top['probability']:.3f}, "
                    f"center=({top['center_x']:.1f}, {top['center_y']:.1f}, {top['center_z']:.1f})"
                )

        except Exception as e:
            pocket_info["parse_error"] = str(e)
            print(f"      ❌ Failed to parse P2Rank CSV: {e}")
        
        return pocket_info


    @staticmethod
    def _p2rank_pocket_residues(pocket: Dict) -> List[Tuple[str, str]]:
        """P2Rank residue_ids 'A_84 A_85B' -> [('A','84'), ('A','85B')]."""
        out = []
        for item in str(pocket.get("residue_ids", "")).split():
            if "_" in item:
                chain, num = item.split("_", 1)
                out.append((chain, num.upper()))
        return out

    def _p2rank_active_site_pocket(self, p2rank_result: Dict, targets) -> Optional[Dict]:
        """Best-ranked P2Rank pocket that contains at least one active-site residue."""
        if not targets:
            return None
        for p in (p2rank_result or {}).get("pockets", []):
            if any(self._residue_matches(c, n, targets) for c, n in self._p2rank_pocket_residues(p)):
                return p
        return None

    def _caver_start_point(self, pdb_file: Path, targets, p2rank_result: Dict) -> Dict:
        """
        Where CAVER starts, in order of preference:
          1. active site given  -> centroid of the side-chain heavy atoms of those
                                   residues (CA for Gly)
          2. no active site     -> centre of P2Rank's top pocket (flagged as a warning:
                                   the top pocket is often NOT the catalytic site)
          3. neither            -> no start point; CAVER is skipped for this structure.
        There is deliberately no "atom 1" fallback: atom 1 is usually the N-terminus,
        and tunnels from it look like valid output while meaning nothing.
        """
        if targets:
            xyz, found = [], set()
            with open(pdb_file, "r", errors="ignore") as f:
                for line in f:
                    if not line.startswith(("ATOM", "HETATM")):
                        continue
                    chain, resid = line[21].strip(), (line[22:26].strip() + line[26].strip()).upper()
                    if not self._residue_matches(chain, resid, targets):
                        continue
                    name, resn = line[12:16].strip(), line[17:20].strip()
                    el = (line[76:78].strip() or name[:1]).upper()
                    if el == "H":
                        continue
                    if name in ("N", "C", "O", "CA") and not (resn == "GLY" and name == "CA"):
                        continue
                    xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                    found.add((chain, resid))
            missing = [f"{c + ':' if c else ''}{n}" for c, n in targets
                       if not any(n == r and (c is None or c == fc) for fc, r in found)]
            if not xyz:
                return {"source": "none", "xyz": None,
                        "detail": f"active-site residues not found in structure: {sorted(missing)}"}
            cen = tuple(sum(v[k] for v in xyz) / len(xyz) for k in range(3))
            detail = f"{len(found)} active-site residues, {len(xyz)} side-chain atoms"
            if missing:
                detail += f"; NOT FOUND: {sorted(missing)}"
            return {"source": "active_site", "xyz": cen, "detail": detail}

        pockets = (p2rank_result or {}).get("pockets", [])
        if pockets:
            top = pockets[0]
            return {"source": "p2rank_top_pocket",
                    "xyz": (top["center_x"], top["center_y"], top["center_z"]),
                    "detail": "no active site given; P2Rank rank-1 pocket used - may not be the catalytic site"}
        return {"source": "none", "xyz": None, "detail": "no active site given and no P2Rank pocket"}

    def run_caver(self, pdb_files: List[str] = None,
                  probe_radius: float = 0.9,
                  active_site_resids: ActiveSiteSpec = "",
                  p2rank_results: Optional[Dict[str, Dict]] = None,
                  group_frames: bool = False) -> Dict[str, Dict]:
        """
        group_frames=False (default): every file is its own CAVER run.
        group_frames=True: files whose stems differ only by a trailing number
        (traj_1.pdb, traj_2.pdb, ...) are treated as frames of ONE trajectory.
        Only use this for MD snapshots - it would also merge WT1/WT2 or rep1/rep2.
        """
        p2rank_results = p2rank_results or {}

        print("\n" + "=" * 70)
        print("🌀 Running CAVER 3.0.2 Analysis...")
        print("=" * 70)

        if pdb_files is None:
            pdb_files = list(self.input_dir.glob("*.pdb"))
        else:
            pdb_files = [Path(f) for f in pdb_files]

        if not pdb_files:
            print("⚠️ No PDB files found for CAVER")
            return {}

        caver_jar = self._find_caver_jar()
        if not caver_jar:
            print("❌ caver.jar not found")
            return {}

        pdb_groups: Dict[str, List[Path]] = {}
        self._caver_group_of = {}
        if group_frames:
            for pdb in sorted(pdb_files):
                m = re.match(r"^(.*?)[_-]?(\d+)$", pdb.stem)
                prefix = m.group(1) if m and m.group(1) else pdb.stem
                pdb_groups.setdefault(prefix, []).append(pdb)
            # a "group" of one file keeps its full name, so it still matches P2Rank/fpocket
            pdb_groups = {(k if len(v) > 1 else v[0].stem): v for k, v in pdb_groups.items()}
        else:
            pdb_groups = {pdb.stem: [pdb] for pdb in pdb_files}
        for name, group in pdb_groups.items():
            for pdb in group:
                self._caver_group_of[pdb.stem] = name

        print(f"\n📊 {len(pdb_groups)} CAVER run(s)"
              + (" (trajectory frames grouped)" if group_frames else " (one per structure)"))
        for prefix, files in pdb_groups.items():
            if len(files) > 1:
                print(f"   • {prefix}: {len(files)} frame(s)")

        results: Dict[str, Dict] = {}

        for group_idx, (protein_name, group_pdbs) in enumerate(pdb_groups.items(), 1):
            print(f"\n[{group_idx}/{len(pdb_groups)}] 🌀 CAVER for {protein_name} ({len(group_pdbs)} frame(s))")

            output_subdir = self.caver_output / protein_name
            output_subdir.mkdir(parents=True, exist_ok=True)

            representative_pdb = group_pdbs[0]
            targets = self._active_site_for(active_site_resids, representative_pdb.stem)
            start = self._caver_start_point(representative_pdb, targets,
                                            p2rank_results.get(representative_pdb.stem, {}))
            if start["xyz"] is None:
                print(f"   ❌ No CAVER starting point: {start['detail']} - skipped")
                results[protein_name] = {"error": f"no start point: {start['detail']}",
                                         "start_point": start}
                continue
            flag = "✓" if start["source"] == "active_site" else "⚠️"
            print(f"   {flag} Start [{start['source']}] at "
                  f"({start['xyz'][0]:.2f}, {start['xyz'][1]:.2f}, {start['xyz'][2]:.2f}) - {start['detail']}")

            temp_pdb_dir = output_subdir / "temp_input"
            temp_pdb_dir.mkdir(exist_ok=True)
            for pdb in group_pdbs:
                shutil.copy(str(pdb), str(temp_pdb_dir / pdb.name))
            print(f"   ✓ Prepared {len(group_pdbs)} PDB file(s) in: {temp_pdb_dir}")

            config_file = self._create_full_caver_config(
                output_subdir,
                start_xyz=start["xyz"],
                probe_radius=probe_radius,
                num_frames=len(group_pdbs)
            )
            print(f"   📝 CAVER config: {config_file}")

            cmd = [
                "java", "-jar", str(caver_jar),
                "-home", str(caver_jar.parent),
                "-pdb", str(temp_pdb_dir),  # ✅ Directory containing all frames
                "-conf", str(config_file),
                "-out", str(output_subdir),
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                if result.returncode == 0:
                    print("   ✓ CAVER finished")
                    results[protein_name] = self._parse_caver_results(output_subdir, protein_name)
                    tc = results[protein_name]["summary"]["tunnel_count"]
                    print(f"   ✓ Tunnels found: {tc}")
                else:
                    print("   ❌ CAVER error")
                    print(result.stderr[:200])
                    results[protein_name] = {"error": result.stderr}

            except subprocess.TimeoutExpired:
                print("   ⏱️ CAVER timeout")
                results[protein_name] = {"error": "Timeout"}
            except Exception as e:
                print(f"   ❌ CAVER exception: {e}")
                results[protein_name] = {"error": str(e)}
            finally:
                results.setdefault(protein_name, {})["start_point"] = start
                # Clean up temporary PDB directory
                if temp_pdb_dir.exists():
                    shutil.rmtree(temp_pdb_dir, ignore_errors=True)

        print("\n" + "=" * 70)
        print("✅ CAVER analysis completed!")
        print("=" * 70)

        return results

    def _create_full_caver_config(self, output_dir: Path,
                                  start_xyz: Tuple[float, float, float],
                                  probe_radius: float = 0.9,
                                  num_frames: int = 1) -> Path:
        config_file = output_dir / "config.txt"

        txt = """#*****************************
# CALCULATION SETUP
#*****************************
load_tunnels no
load_cluster_tree no

#*****************************
# INPUT DATA
#*****************************
time_sparsity 1
first_frame 1 #如果有多个MD轨迹文件pdb文件，可修改为10
"""
        # Set last_frame based on number of PDB files
        txt += f"last_frame {num_frames}\n"
        
        txt += """
#*****************************
# TUNNEL CALCULATION
#*****************************
"""

        # CAVER optimises the start within max_distance of this point (default 3 A)
        txt += "starting_point_coordinates {:.3f} {:.3f} {:.3f}\n".format(*start_xyz)

        txt += f"""
probe_radius {probe_radius}
shell_radius 3
shell_depth 4

#*****************************
# TUNNEL CLUSTERING
#*****************************
clustering average_link
weighting_coefficient 1
clustering_threshold 3.5

#*****************************
# GENERATION OF OUTPUTS
#*****************************
one_tunnel_in_snapshot cheapest
save_dynamics_visualization yes

generate_summary yes
generate_tunnel_characteristics yes
generate_tunnel_profiles yes

generate_histograms yes
bottleneck_histogram 0.0 2.0 20
throughput_histogram 0 1.0 10

generate_bottleneck_heat_map yes
bottleneck_heat_map_range 1.0 2.0
bottleneck_heat_map_element_size 10 20

generate_profile_heat_map yes
profile_heat_map_resolution 0.5
profile_heat_map_range 1.0 2.0
profile_heat_map_element_size 20 10

compute_tunnel_residues yes
residue_contact_distance 3.0

compute_bottleneck_residues yes
bottleneck_contact_distance 3.0

#*****************************
# ADVANCED SETTINGS
#*****************************
number_of_approximating_balls 12

compute_errors no
save_error_profiles no

path_to_vmd ""
generate_trajectory yes

#-----------------------------
# Others
#-----------------------------
swap no
seed 1
"""

        with open(config_file, "w", encoding="utf-8") as f:
            f.write(txt)

        return config_file

    def _find_caver_jar(self) -> Optional[Path]:
        jars = list(self.work_dir.rglob("caver.jar"))
        return jars[0] if jars else None

    def _parse_caver_results(self, output_dir: Path, pdb_name: str) -> Dict:
        result = {
            "pdb_name": pdb_name,
            "summary": {"tunnel_count": 0},
            "tunnels": []
        }
        
        csv_path = output_dir / "analysis" / "tunnel_characteristics.csv"
        
        if not csv_path.exists():
            tunnel_files = []
            for pattern in ["tunnel_*.pdb", "tunnel*.pdb"]:
                tunnel_files.extend(list(output_dir.glob(pattern)))
                data_dir = output_dir / "data"
                if data_dir.exists():
                    tunnel_files.extend(list(data_dir.glob(pattern)))
            
            tunnel_files = list(set(tunnel_files))
            result["summary"]["tunnel_count"] = len(tunnel_files)
            result["tunnels"] = [{"id": f.stem, "file": str(f)} for f in tunnel_files]
            return result
        
        try:
            # CAVER may use comma or tab separator
            # Try both with proper space handling
            df = None
            for sep in [',', '\t']:
                try:
                    df = pd.read_csv(csv_path, sep=sep, skipinitialspace=True)
                    # Validate: should have multiple columns
                    if len(df.columns) > 5:
                        break
                except:
                    continue
            
            if df is None or len(df) == 0:
                print(f"      ⚠️ Could not parse CSV with comma or tab separator")
                return result
            
            df.columns = df.columns.str.strip()
            
            result["summary"]["tunnel_count"] = len(df)
            result["summary"]["csv_file"] = str(csv_path)

            for idx, row in df.iterrows():
                try:
                    tunnel_data = {
                        "cluster": int(row["Tunnel cluster"]) if "Tunnel cluster" in df.columns else idx+1,
                        "tunnel_id": int(row["Tunnel"]) if "Tunnel" in df.columns else idx+1,
                        "throughput": float(row["Throughput"]) if "Throughput" in df.columns else 0.0,
                        "bottleneck_radius": float(row["Bottleneck radius"]) if "Bottleneck radius" in df.columns else 0.0,
                        "length": float(row["Length"]) if "Length" in df.columns else 0.0,
                        "curvature": float(row["Curvature"]) if "Curvature" in df.columns else 0.0,
                        "cost": float(row["Cost"]) if "Cost" in df.columns else 0.0
                    }
                    result["tunnels"].append(tunnel_data)
                except (KeyError, ValueError) as e:
                    print(f"      ⚠️ Skipping row {idx}: {e}")
                    continue
            
            if len(df) > 0 and len(result["tunnels"]) > 0:
                try:
                    if "Throughput" in df.columns:
                        result["summary"]["avg_throughput"] = float(df["Throughput"].mean())
                    if "Bottleneck radius" in df.columns:
                        result["summary"]["avg_bottleneck"] = float(df["Bottleneck radius"].mean())
                    if "Length" in df.columns:
                        result["summary"]["avg_length"] = float(df["Length"].mean())
                    if "Curvature" in df.columns:
                        result["summary"]["avg_curvature"] = float(df["Curvature"].mean())
                except Exception as stat_err:
                    print(f"      ⚠️ Statistics calculation failed: {stat_err}")
                
        except Exception as e:
            print(f"      ⚠️ Failed to parse tunnel_characteristics.csv: {e}")
            import traceback
            print(f"      → Traceback: {traceback.format_exc()}")
            result["parse_error"] = str(e)
        
        return result

    def generate_summary_report(self, p2rank_results: Dict, caver_results: Dict,
                                fpocket_selected_df: Optional[pd.DataFrame] = None,
                                active_site_resids: ActiveSiteSpec = "",
                                site_report: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
        """
        One row per input structure. Rows are keyed by the input file stem; a
        structure that was CAVER-analysed as part of a frame group gets that
        group's CAVER values (column CAVER_Group). Values that would be
        misleading are left empty (NaN), never filled with 0: e.g. the
        functional-pocket volume when no pocket contains the active site.
        """
        print("\n📋 Generating Summary Report...")
        nan = float("nan")

        fp_lookup: Dict[str, Dict] = {}
        if isinstance(fpocket_selected_df, pd.DataFrame) and not fpocket_selected_df.empty:
            for _, r in fpocket_selected_df.iterrows():
                fp_lookup[str(r.get("protein", ""))] = r.to_dict()

        report_rows = []
        proteins = sorted(set(p2rank_results) | set(self._caver_group_of) | set(fp_lookup)
                          | (set(caver_results) - set(self._caver_group_of.values())))

        for p in proteins:
            r1 = p2rank_results.get(p, {})
            group = self._caver_group_of.get(p, p)
            r2 = caver_results.get(group, {})
            fp = fp_lookup.get(p, {})
            targets = self._active_site_for(active_site_resids, p)

            pockets = r1.get("pockets", [])
            top = pockets[0] if pockets else {}
            asp = self._p2rank_active_site_pocket(r1, targets)
            start = r2.get("start_point", {})

            fp_on_site = fp.get("selected_by") == "active_site"
            if "error" in r1:
                p2_status = "Failed"
            elif targets and asp is None:
                p2_status = "Warning: no pocket contains the active site"
            else:
                p2_status = "Success"
            if "error" in r2:
                cv_status = "Failed: " + str(r2["error"]).splitlines()[0][:80]
            elif not r2:
                cv_status = "Not run"
            elif start.get("source") != "active_site":
                cv_status = "Warning: start from P2Rank top pocket, not the active site"
            else:
                cv_status = "Success"

            xyz = start.get("xyz")
            site = (site_report or {}).get(p, {})
            row = {
                "Protein": p,
                "Active_Site_Residues": " ".join(f"{c + ':' if c else ''}{n}" for c, n in sorted(
                    targets, key=lambda t: (t[0] or "", t[1]))),
                "Active_Site_Source": site.get("source", "user" if targets else "none"),
                "Active_Site_Basis": site.get("name", ""),
                "Active_Site_Confidence": site.get("confidence", "user" if targets else "none"),
                "Total_Pockets": r1.get("summary", {}).get("total_pockets", 0),
                "Top_Pocket_Score": round(top.get("score", nan), 2),
                "Top_Pocket_Probability": round(top.get("probability", nan), 3),
                "ActiveSite_Pocket_Rank": asp["rank"] if asp else nan,
                "ActiveSite_Pocket_Probability": round(asp["probability"], 3) if asp else nan,
                "CAVER_Group": group,
                "Caver_Start_Source": start.get("source", ""),
                "Caver_Start_XYZ": " ".join(f"{v:.2f}" for v in xyz) if xyz else "",
                "Tunnel_Count": r2.get("summary", {}).get("tunnel_count", 0),
                "Avg_Bottleneck_Radius": round(r2.get("summary", {}).get("avg_bottleneck", nan), 3),
                "Avg_Length": round(r2.get("summary", {}).get("avg_length", nan), 2),
                "Func_Pocket_Selected_By": fp.get("selected_by", ""),
                "Func_Pocket_ID": int(fp["fpocket_pocket_id"]) if fp_on_site else nan,
                "Func_Pocket_Volume_A3": round(float(fp.get("fpocket_volume_A3", nan)), 1) if fp_on_site else nan,
                "Active_Site_Hits": int(fp.get("active_site_hit_count", 0)) if fp else 0,
                "P2Rank_Status": p2_status,
                "CAVER_Status": cv_status,
            }
            report_rows.append(row)

        df = pd.DataFrame(report_rows)
        report_path = self.output_dir / "summary_report.csv"
        df.to_csv(report_path, index=False)
        print(f"✅ Summary saved to: {report_path}")
        return df

    def save_detailed_results(self, p2rank_results: Dict, caver_results: Dict):
        data = {"p2rank": p2rank_results, "caver": caver_results}
        json_path = self.output_dir / "detailed_results.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"📄 Detailed results saved to: {json_path}")
        
        self._save_tunnel_details_csv(caver_results)
    
    def _save_tunnel_details_csv(self, caver_results: Dict):
        tunnel_rows = []
        
        for protein_name, result in caver_results.items():
            if "error" in result or not result.get("tunnels"):
                continue
            
            for tunnel in result["tunnels"]:
                row = {
                    "Protein": protein_name,
                    "Cluster": tunnel.get("cluster", ""),
                    "Tunnel_ID": tunnel.get("tunnel_id", ""),
                    "Throughput": tunnel.get("throughput", 0.0),
                    "Bottleneck_Radius": tunnel.get("bottleneck_radius", 0.0),
                    "Length": tunnel.get("length", 0.0),
                    "Curvature": tunnel.get("curvature", 0.0),
                    "Cost": tunnel.get("cost", 0.0)
                }
                tunnel_rows.append(row)
        
        if tunnel_rows:
            df = pd.DataFrame(tunnel_rows)
            csv_path = self.output_dir / "tunnel_details.csv"
            df.to_csv(csv_path, index=False)
            print(f"📊 Tunnel details saved to: {csv_path}")

    # ==================================================================
    # fpocket: pocket-volume / druggability analysis (moved into engine)
    # ==================================================================
    @staticmethod
    def _test_fpocket_binary(exe) -> bool:
        try:
            subprocess.run([str(exe), "-h"], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, timeout=20)
            return True
        except Exception:
            return False

    def _setup_fpocket(self) -> str:
        """Install + compile fpocket once, cache the binary for later reruns."""
        if self.fpocket_bin and self._test_fpocket_binary(self.fpocket_bin):
            return self.fpocket_bin

        print("\n🧬 Setting up fpocket...")
        src_dir = self.fpocket_src

        if src_dir.exists():
            for exe in src_dir.rglob("fpocket"):
                if exe.is_file() and os.access(exe, os.X_OK) and self._test_fpocket_binary(exe):
                    print(f"   ✅ Using cached fpocket: {exe}")
                    self.fpocket_bin = str(exe)
                    return self.fpocket_bin

        subprocess.run(["apt-get", "update", "-qq"], check=True, timeout=180)
        subprocess.run(["apt-get", "install", "-y", "-qq", "git", "build-essential"],
                       check=True, timeout=240)

        if not src_dir.exists():
            print("   • Cloning fpocket source...")
            subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/Discngine/fpocket.git", str(src_dir)],
                           check=True, timeout=240)

        print("   • Compiling fpocket...")
        subprocess.run(["make"], cwd=str(src_dir), check=True, timeout=420)

        for exe in src_dir.rglob("fpocket"):
            if not exe.is_file():
                continue
            try:
                exe.chmod(0o755)
            except Exception:
                pass
            if self._test_fpocket_binary(exe):
                print(f"   ✅ fpocket ready: {exe}")
                self.fpocket_bin = str(exe)
                return self.fpocket_bin

        raise FileNotFoundError("fpocket binary not found / not executable after compilation")

    @staticmethod
    def _make_protein_only_pdb(input_pdb, output_pdb):
        """Keep ATOM + TER of the FIRST model only; drop all HETATM (ligands,
        cofactors, ions, waters). Stopping at the first ENDMDL matters: writing
        every model's ATOM lines without MODEL records superposes N copies."""
        kept = 0
        with open(input_pdb, "r", errors="ignore") as fin, open(output_pdb, "w") as fout:
            for line in fin:
                if line.startswith("ENDMDL"):
                    break
                if line.startswith("ATOM"):
                    fout.write(line)
                    kept += 1
                elif line.startswith("TER"):
                    fout.write(line)
            fout.write("END\n")
        if kept == 0:
            raise ValueError(f"No ATOM records found in {input_pdb}")
        return output_pdb

    @staticmethod
    def _parse_fpocket_info(info_file, protein_name) -> List[Dict]:
        rows, current = [], None
        key_map = {
            "score": "fpocket_score",
            "druggability score": "fpocket_druggability_score",
            "drug score": "fpocket_druggability_score",
            "number of alpha spheres": "num_alpha_spheres",
            "total sasa": "total_sasa",
            "polar sasa": "polar_sasa",
            "apolar sasa": "apolar_sasa",
            "volume": "fpocket_volume_A3",
            "mean local hydrophobic density": "mean_local_hydrophobic_density",
            "mean alpha sphere radius": "mean_alpha_sphere_radius",
            "mean alp. sph. radius": "mean_alpha_sphere_radius",
            "mean alp. sph. solvent access": "mean_alpha_sphere_solvent_access",
            "apolar alpha sphere proportion": "apolar_alpha_sphere_proportion",
            "hydrophobicity score": "hydrophobicity_score",
            "polarity score": "polarity_score",
            "charge score": "charge_score",
        }
        info_file = Path(info_file)
        if not info_file.exists():
            return rows
        with open(info_file, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                m = re.match(r"Pocket\s+(\d+)\s*:", line, flags=re.IGNORECASE)
                if m:
                    if current:
                        rows.append(current)
                    current = {"protein": protein_name, "fpocket_pocket_id": int(m.group(1))}
                    continue
                if current and ":" in line:
                    raw_key, raw_value = line.split(":", 1)
                    key = raw_key.strip().lower()
                    num = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_value.strip())
                    if num:
                        out_key = key_map.get(key, key.replace(" ", "_"))
                        try:
                            current[out_key] = float(num.group(0))
                        except ValueError:
                            pass
        if current:
            rows.append(current)
        return rows

    @staticmethod
    def _read_fpocket_pocket_residues(pocket_atm_file) -> List[str]:
        """Residue identifiers as Chain_ResidueNumber_ResidueName, e.g. A_84_GLN."""
        residues = set()
        pocket_atm_file = Path(pocket_atm_file)
        if not pocket_atm_file.exists():
            return []
        with open(pocket_atm_file, "r", errors="ignore") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    resn = line[17:20].strip()
                    chain = line[21].strip() or "X"
                    resi = line[22:26].strip()
                    icode = line[26].strip()
                    if resi:
                        residues.add(f"{chain}_{resi}{icode}_{resn}")
        return sorted(residues)

    @classmethod
    def _annotate_fpocket_residue_hits(cls, row, pocket_residues, targets):
        """targets: {(chain or None, '84'/'84A')}. Exact residue match, insertion
        code included (84A is not 84); chain checked when given."""
        hit_residues = []
        for r in pocket_residues:
            parts = r.split("_")
            if len(parts) >= 2 and cls._residue_matches(parts[0], parts[1], targets):
                hit_residues.append(r)
        row["pocket_residues"] = " ".join(pocket_residues)
        row["active_site_given"] = bool(targets)
        row["active_site_hit_count"] = len(hit_residues)
        row["active_site_hit_residues"] = " ".join(hit_residues)
        return row

    @staticmethod
    def _select_active_site_fpocket_pockets(fpocket_df) -> pd.DataFrame:
        """Per protein: most active-site hits → higher fpocket_score → larger volume.
        Column selected_by says how the pocket was chosen:
          active_site       >= 1 active-site residue in the pocket
          score_no_hit      active site given, but no pocket contains it
          score_no_site     no active site given
        Only 'active_site' rows are functional-pocket estimates."""
        if fpocket_df is None or fpocket_df.empty:
            return pd.DataFrame()
        df = fpocket_df.copy()
        for col in ["active_site_hit_count", "fpocket_score", "fpocket_volume_A3"]:
            if col not in df.columns:
                df[col] = 0
        if "active_site_given" not in df.columns:
            df["active_site_given"] = False
        df = df.sort_values(
            by=["protein", "active_site_hit_count", "fpocket_score", "fpocket_volume_A3"],
            ascending=[True, False, False, False])
        sel = df.groupby("protein", as_index=False).head(1).reset_index(drop=True)
        sel["selected_by"] = [
            "active_site" if h > 0 else ("score_no_hit" if g else "score_no_site")
            for h, g in zip(sel.active_site_hit_count, sel.active_site_given)]
        return sel

    def run_fpocket(self, pdb_files: List[str] = None,
                    active_site_resids: ActiveSiteSpec = "",
                    timeout: int = 300,
                    protein_only: bool = True):
        """Run fpocket on all PDBs; return (all_pockets_df, selected_active_site_df)."""
        print("\n" + "=" * 70)
        print("🧬 Running fpocket Pocket-Volume Analysis...")
        print("=" * 70)

        if pdb_files is None:
            pdb_files = list(self.input_dir.glob("*.pdb"))
        else:
            pdb_files = [Path(f) for f in pdb_files]

        if not pdb_files:
            print("⚠️ No PDB files found for fpocket")
            return pd.DataFrame(), pd.DataFrame()

        fpocket_bin = self._setup_fpocket()
        all_rows = []

        for pdb_path in pdb_files:
            pdb_path = Path(pdb_path)
            safe_stem = self._safe_stem(pdb_path)
            target_resids = self._active_site_for(active_site_resids, pdb_path.stem)
            work_dir = self.fpocket_output / safe_stem
            work_dir.mkdir(parents=True, exist_ok=True)

            raw_pdb = work_dir / f"{safe_stem}_raw.pdb"
            work_pdb = work_dir / f"{safe_stem}.pdb"
            shutil.copy2(pdb_path, raw_pdb)

            try:
                if protein_only:
                    self._make_protein_only_pdb(raw_pdb, work_pdb)
                    mode = "protein-only"
                else:
                    shutil.copy2(raw_pdb, work_pdb)
                    mode = "original"
            except Exception as e:
                print(f"   ⚠️ Protein-only conversion failed for {pdb_path.name}: {e}; using original")
                shutil.copy2(raw_pdb, work_pdb)
                mode = "original-fallback"

            print(f"\n   ▶ fpocket: {pdb_path.name}  (mode: {mode})")
            try:
                subprocess.run([fpocket_bin, "-f", str(work_pdb)],
                               cwd=str(work_dir), check=True, timeout=timeout,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                fpocket_out_dir = work_dir / f"{safe_stem}_out"
                info_file = fpocket_out_dir / f"{safe_stem}_info.txt"
                if not info_file.exists():
                    cands = list(fpocket_out_dir.glob("*_info.txt"))
                    if cands:
                        info_file = cands[0]

                rows = self._parse_fpocket_info(info_file, safe_stem)
                for row in rows:
                    pocket_id = int(row["fpocket_pocket_id"])
                    pocket_atm = fpocket_out_dir / "pockets" / f"pocket{pocket_id}_atm.pdb"
                    pocket_residues = self._read_fpocket_pocket_residues(pocket_atm)
                    row["original_pdb"] = pdb_path.name
                    row["fpocket_mode"] = mode
                    self._annotate_fpocket_residue_hits(row, pocket_residues, target_resids)
                all_rows.extend(rows)
                print(f"      ✅ {len(rows)} pockets parsed")

            except subprocess.TimeoutExpired:
                print(f"      ❌ fpocket timeout for {pdb_path.name}")
            except subprocess.CalledProcessError as e:
                print(f"      ❌ fpocket failed for {pdb_path.name}: {(e.stderr or '')[:300]}")
            except Exception as e:
                print(f"      ❌ fpocket failed for {pdb_path.name}: {e}")

        fpocket_df = pd.DataFrame(all_rows)
        if fpocket_df.empty:
            print("⚠️ No fpocket pockets parsed.")
            return fpocket_df, pd.DataFrame()

        out_csv = self.fpocket_output / "fpocket_pocket_volumes.csv"
        fpocket_df.to_csv(out_csv, index=False)
        print(f"\n📄 fpocket all-pocket result: {out_csv}")

        selected_df = self._select_active_site_fpocket_pockets(fpocket_df)
        selected_csv = self.fpocket_output / "fpocket_active_site_pocket_candidates.csv"
        selected_df.to_csv(selected_csv, index=False)
        print(f"📄 fpocket active-site candidates: {selected_csv}")

        return fpocket_df, selected_df

    # ==================================================================
    # High-level orchestration (so Colab stays a thin frontend)
    # ==================================================================
    def analyze(self, pdb_files: List[str] = None,
                p2rank_threads: int = 2, p2rank_min_score: float = 0.0,
                caver_probe_radius: float = 0.9,
                run_fpocket: bool = True, fpocket_protein_only: bool = True,
                active_site_resids: ActiveSiteSpec = "", fpocket_timeout: int = 300,
                caver_group_frames: bool = False, auto_active_site: bool = True) -> Dict:
        """Run P2Rank + CAVER + fpocket, build the merged report, return a result bundle.

        active_site_resids: see ActiveSiteSpec at the top of this file. It drives
        fpocket pocket selection, the P2Rank active-site pocket columns, and the
        CAVER starting point.
        auto_active_site: for every structure the spec gives no residues for,
        detect one (ligand > metal > catalytic motif, see detect_active_sites).
        The decision and all alternatives are written to
        integrated_summary/active_sites_auto.json for review. Structures with no
        candidate fall back to P2Rank's top pocket, flagged as a warning.
        """
        if pdb_files is None:
            pdb_files = sorted(self.input_dir.glob("*.pdb"))
        pdb_files = [Path(p) for p in pdb_files]

        multi = [p.name for p in pdb_files if len(self._split_models(p)) > 1]
        if multi:
            raise ValueError(f"Multi-MODEL input {multi[:3]}{'...' if len(multi) > 3 else ''}: "
                             "pass the files through prepare_inputs() first")

        t0 = time.time()

        p2rank_results = self.run_p2rank(pdb_files=pdb_files,
                                         min_score=p2rank_min_score,
                                         threads=p2rank_threads)

        site_report: Dict[str, Dict] = {}
        if auto_active_site:
            print("\n" + "=" * 70)
            print("🎯 Active sites (user-supplied or auto-detected)")
            print("=" * 70)
            site_report = self.detect_active_sites(pdb_files, p2rank_results, given=active_site_resids)
            with open(self.integrated_output / "active_sites_auto.json", "w") as f:
                json.dump(site_report, f, indent=1)
            # from here on every step uses the resolved per-structure residues
            active_site_resids = {s: r["resids"] for s, r in site_report.items() if r["resids"]}

        try:
            caver_results = self.run_caver(pdb_files=pdb_files,
                                           probe_radius=caver_probe_radius,
                                           active_site_resids=active_site_resids,
                                           p2rank_results=p2rank_results,
                                           group_frames=caver_group_frames)
        except Exception as e:
            print(f"⚠️ CAVER step failed, continuing: {e}")
            caver_results = {}

        fpocket_all_df, fpocket_selected_df = pd.DataFrame(), pd.DataFrame()
        if run_fpocket:
            try:
                fpocket_all_df, fpocket_selected_df = self.run_fpocket(
                    pdb_files=pdb_files,
                    active_site_resids=active_site_resids,
                    timeout=fpocket_timeout,
                    protein_only=fpocket_protein_only)
            except Exception as e:
                print(f"⚠️ fpocket step failed, continuing: {e}")

        summary_df = self.generate_summary_report(
            p2rank_results, caver_results, fpocket_selected_df,
            active_site_resids=active_site_resids, site_report=site_report)
        self.save_detailed_results(p2rank_results, caver_results)

        # Consolidated copies for convenience
        if not summary_df.empty:
            summary_df.to_csv(self.integrated_output / "summary_report.csv", index=False)
        if not fpocket_all_df.empty:
            fpocket_all_df.to_csv(self.integrated_output / "fpocket_all_pockets.csv", index=False)
        if not fpocket_selected_df.empty:
            fpocket_selected_df.to_csv(
                self.integrated_output / "fpocket_selected_active_site_pockets.csv", index=False)

        counts = {
            "proteins": len(pdb_files),
            "p2rank_pockets": sum(len(v.get("pockets", [])) for v in p2rank_results.values()),
            "caver_tunnels": sum(len(v.get("tunnels", [])) for v in caver_results.values()),
            "fpocket_pockets": int(len(fpocket_all_df)),
            "elapsed_s": round(time.time() - t0, 1),
        }

        return {
            "summary_df": summary_df,
            "fpocket_all_df": fpocket_all_df,
            "fpocket_selected_df": fpocket_selected_df,
            "p2rank_results": p2rank_results,
            "caver_results": caver_results,
            "active_sites": site_report,
            "counts": counts,
        }

    def package_results(self, zip_path: str = "/content/analysis_results.zip") -> str:
        """Zip the whole output directory for download (pure stdlib)."""
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files_list in os.walk(self.output_dir):
                for f in files_list:
                    full = os.path.join(root, f)
                    zf.write(full, os.path.relpath(full, self.output_dir))
        return zip_path

def main():
    analyzer = ProteinAnalyzer()
    analyzer.setup_environment()

    pdb_files = sorted(analyzer.input_dir.glob("*.pdb"))
    result = analyzer.analyze(pdb_files=pdb_files)

    print("\n📊 Summary:")
    print(result["summary_df"].to_string(index=False))
    return analyzer, result


if __name__ == "__main__":
    analyzer, result = main()
