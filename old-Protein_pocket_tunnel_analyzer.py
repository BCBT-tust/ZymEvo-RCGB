#!/usr/bin/env python3

import os
import re
import time
import json
import zipfile
import subprocess
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import shutil


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

    def prepare_inputs(self, uploaded_paths, clear_existing: bool = True) -> List[Path]:
        """
        Copy uploaded structures into input_dir with sanitized names.
        Drops files with no ATOM records and avoids duplicate-name collisions.
        Returns the clean list of PDB paths the pipeline will analyze.
        """
        if clear_existing:
            for old in self.input_dir.glob("*.pdb"):
                old.unlink(missing_ok=True)

        clean_files: List[Path] = []
        used = set()
        for src in uploaded_paths:
            src = Path(src)
            if not self._pdb_has_atoms(src):
                print(f"   ⚠️ Skipping {src.name}: no ATOM records")
                continue

            stem = self._safe_stem(src)
            candidate = stem
            i = 2
            while candidate in used:           # de-duplicate cleanly (no "(2)")
                candidate = f"{stem}_{i}"
                i += 1
            used.add(candidate)

            dst = self.input_dir / f"{candidate}.pdb"
            shutil.copy2(src, dst)
            clean_files.append(dst)
            label = src.name if src.name == dst.name else f"{src.name} → {dst.name}"
            print(f"   • {label}")

        return sorted(clean_files)

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


    def _get_atoms_from_residue_ids(self, pdb_file: Path, pdb_name: str) -> Optional[List[int]]:

        p2rank_dir = self.p2rank_output / pdb_name
        csv_files = list(p2rank_dir.glob("*predictions.csv")) or list(p2rank_dir.glob("*.csv"))
        if not csv_files:
            print("   ⚠️ No P2Rank CSV found for residue_ids")
            return None

        try:
            df = pd.read_csv(csv_files[0], skipinitialspace=True)
            if df.empty:
                print("   ⚠️ P2Rank CSV empty")
                return None
            
            residues_str = str(df.iloc[0].get("residue_ids", "")).strip()
            if not residues_str:
                print("   ⚠️ No residue_ids in P2Rank output")
                return None

            residues = []
            for item in residues_str.split():
                if "_" in item:
                    chain, num = item.split("_")
                    try:
                        residues.append((chain, int(num)))
                    except:
                        pass

            if not residues:
                print("   ⚠️ Failed to parse residue_ids field")
                return None

            found_atoms: List[int] = []
            with open(pdb_file, "r") as f:
                for line in f:
                    if not (line.startswith("ATOM") or line.startswith("HETATM")):
                        continue

                    try:
                        chain = line[21].strip()
                        resid = int(line[22:26])
                        atom_id = int(line[6:11])
                    except:
                        continue

                    if (chain, resid) in residues:
                        found_atoms.append(atom_id)

            if not found_atoms:
                print("   ⚠️ No atoms matched residue_ids in PDB")
                return None

            print(
                f"   ✓ Using {len(found_atoms)} atoms from {len(residues)} residues (from residue_ids)"
            )
            return found_atoms

        except Exception as e:
            print(f"   ⚠️ Failed to map residue_ids to atoms: {e}")
            return None

    def run_caver(self, pdb_files: List[str] = None,
                  probe_radius: float = 0.9) -> Dict[str, Dict]:

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

        import re
        pdb_groups = {}
        
        for pdb in pdb_files:
            match = re.match(r'^(.+?)(\d+)?$', pdb.stem)
            if match:
                prefix = match.group(1)
            else:
                prefix = pdb.stem
            
            if prefix not in pdb_groups:
                pdb_groups[prefix] = []
            pdb_groups[prefix].append(pdb)
        
        print(f"\n📊 Detected {len(pdb_groups)} protein group(s):")
        for prefix, files in pdb_groups.items():
            print(f"   • {prefix}: {len(files)} frame(s)")

        results: Dict[str, Dict] = {}

        for group_idx, (protein_name, group_pdbs) in enumerate(pdb_groups.items(), 1):
            print(f"\n[{group_idx}/{len(pdb_groups)}] 🌀 CAVER for {protein_name} ({len(group_pdbs)} frame(s))")

            output_subdir = self.caver_output / protein_name
            output_subdir.mkdir(parents=True, exist_ok=True)

            temp_pdb_dir = output_subdir / "temp_input"
            temp_pdb_dir.mkdir(exist_ok=True)

            import shutil
            for pdb in group_pdbs:
                temp_pdb_file = temp_pdb_dir / pdb.name
                shutil.copy(str(pdb), str(temp_pdb_file))
            
            print(f"   ✓ Prepared {len(group_pdbs)} PDB file(s) in: {temp_pdb_dir}")

            representative_pdb = group_pdbs[0]
            start_atoms = self._get_atoms_from_residue_ids(representative_pdb, protein_name)
            if not start_atoms:
                print("   ⚠️ No residue-based start atoms, fallback to atom 1")
                start_atoms = [1]

            config_file = self._create_full_caver_config(
                output_subdir,
                start_atoms=start_atoms,
                probe_radius=probe_radius,
                num_frames=len(group_pdbs)  # ✅ New parameter
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
                # Clean up temporary PDB directory
                if temp_pdb_dir.exists():
                    import shutil
                    shutil.rmtree(temp_pdb_dir, ignore_errors=True)

        print("\n" + "=" * 70)
        print("✅ CAVER analysis completed!")
        print("=" * 70)

        return results

    def _create_full_caver_config(self, output_dir: Path,
                                  start_atoms: List[int],
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

        for atom in start_atoms:
            txt += f"starting_point_atom {atom}\n"

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

        with open(config_file, "w") as f:
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
                                fpocket_selected_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        print("\n📋 Generating Summary Report...")

        # Build a per-protein lookup of the selected functional pocket from fpocket
        fp_lookup: Dict[str, Dict] = {}
        if isinstance(fpocket_selected_df, pd.DataFrame) and not fpocket_selected_df.empty:
            for _, r in fpocket_selected_df.iterrows():
                fp_lookup[str(r.get("protein", ""))] = {
                    "vol": r.get("fpocket_volume_A3", 0.0),
                    "fscore": r.get("fpocket_score", 0.0),
                    "hits": r.get("active_site_hit_count", 0),
                }

        report_rows = []
        proteins = sorted(set(p2rank_results.keys()) | set(caver_results.keys()))

        for p in proteins:
            r1 = p2rank_results.get(p, {})
            r2 = caver_results.get(p, {})
            fp = fp_lookup.get(p, {})

            top_score = 0.0
            top_prob = 0.0
            if r1.get("pockets"):
                top = r1["pockets"][0]
                top_score = top.get("score", 0.0)
                top_prob = top.get("probability", 0.0)

            row = {
                "Protein": p,
                "Total_Pockets": r1.get("summary", {}).get("total_pockets", 0),
                "Top_Pocket_Score": round(top_score, 2),
                "Top_Pocket_Probability": round(top_prob, 3),
                "Tunnel_Count": r2.get("summary", {}).get("tunnel_count", 0),
                "Avg_Bottleneck_Radius": round(r2.get("summary", {}).get("avg_bottleneck", 0.0), 3),
                "Avg_Length": round(r2.get("summary", {}).get("avg_length", 0.0), 2),
                "Func_Pocket_Volume_A3": round(float(fp.get("vol", 0.0)), 1),
                "Active_Site_Hits": int(fp.get("hits", 0)),
                "P2Rank_Status": "Success" if "error" not in r1 else "Failed",
                "CAVER_Status": "Success" if "error" not in r2 else "Failed",
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
    def _parse_target_residue_numbers(active_site_resids) -> set:
        out = set()
        for x in str(active_site_resids).split(","):
            x = x.strip()
            if x:
                out.add(x)
        return out

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
        """Keep ATOM + TER/END, drop all HETATM (ligands, cofactors, ions, waters)."""
        kept = 0
        with open(input_pdb, "r", errors="ignore") as fin, open(output_pdb, "w") as fout:
            for line in fin:
                if line.startswith("ATOM"):
                    fout.write(line)
                    kept += 1
                elif line.startswith(("TER", "END")):
                    fout.write(line)
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

    @staticmethod
    def _annotate_fpocket_residue_hits(row, pocket_residues, target_resids):
        hit_residues = []
        for r in pocket_residues:
            parts = r.split("_")
            if len(parts) >= 2:
                resi_only = re.sub(r"[^0-9]", "", parts[1])
                if resi_only and resi_only in target_resids:
                    hit_residues.append(r)
        row["pocket_residues"] = " ".join(pocket_residues)
        row["active_site_hit_count"] = len(hit_residues)
        row["active_site_hit_residues"] = " ".join(hit_residues)
        return row

    @staticmethod
    def _select_active_site_fpocket_pockets(fpocket_df) -> pd.DataFrame:
        """Per protein: most active-site hits → higher fpocket_score → larger volume."""
        if fpocket_df is None or fpocket_df.empty:
            return pd.DataFrame()
        df = fpocket_df.copy()
        for col in ["active_site_hit_count", "fpocket_score", "fpocket_volume_A3"]:
            if col not in df.columns:
                df[col] = 0
        df = df.sort_values(
            by=["protein", "active_site_hit_count", "fpocket_score", "fpocket_volume_A3"],
            ascending=[True, False, False, False])
        return df.groupby("protein", as_index=False).head(1).reset_index(drop=True)

    def run_fpocket(self, pdb_files: List[str] = None,
                    active_site_resids: str = "",
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
        target_resids = self._parse_target_residue_numbers(active_site_resids)
        all_rows = []

        for pdb_path in pdb_files:
            pdb_path = Path(pdb_path)
            safe_stem = self._safe_stem(pdb_path)
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
                active_site_resids: str = "", fpocket_timeout: int = 300) -> Dict:
        """Run P2Rank + CAVER + fpocket, build the merged report, return a result bundle."""
        if pdb_files is None:
            pdb_files = sorted(self.input_dir.glob("*.pdb"))

        t0 = time.time()

        p2rank_results = self.run_p2rank(pdb_files=pdb_files,
                                         min_score=p2rank_min_score,
                                         threads=p2rank_threads)

        try:
            caver_results = self.run_caver(pdb_files=pdb_files,
                                           probe_radius=caver_probe_radius)
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
            p2rank_results, caver_results, fpocket_selected_df)
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
