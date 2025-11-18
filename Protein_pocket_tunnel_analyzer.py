#!/usr/bin/env python3

import os
import subprocess
import json
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

        self.p2rank_path = self.work_dir / "p2rank_2.4.2"
        self.caver_path = self.work_dir

        self._create_directories()
        
    def _create_directories(self):
        for directory in [self.input_dir, self.output_dir, 
                          self.p2rank_output, self.caver_output]:
            directory.mkdir(parents=True, exist_ok=True)

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

            if min_score > 0:
                cmd.extend(["-min_score", str(min_score)])
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                
                if result.returncode == 0:
                    print("   ✅ P2Rank completed")
                    pocket_info = self._parse_p2rank_results(output_subdir, pdb_file.stem)
                    results[pdb_file.stem] = pocket_info
                else:
                    print("   ❌ P2Rank failed")
                    print(result.stderr[:200])
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

    def _parse_p2rank_results(self, output_dir: Path, pdb_name: str) -> Dict:
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

            pocket_info["summary"] = {
                "total_pockets": len(df),
                "output_file": str(csv_files[0])
            }

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
                pocket_info["pockets"].append(pocket)

            if pocket_info["pockets"]:
                top = pocket_info["pockets"][0]
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

            # A_103 A_231 A_235 ...
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

        results: Dict[str, Dict] = {}

        for i, pdb in enumerate(pdb_files, 1):
            pdb_name = pdb.stem
            print(f"\n[{i}/{len(pdb_files)}] 🌀 CAVER for {pdb.name}")

            output_subdir = self.caver_output / pdb_name
            output_subdir.mkdir(parents=True, exist_ok=True)

            start_atoms = self._get_atoms_from_residue_ids(pdb, pdb_name)
            if not start_atoms:
                print("   ⚠️ No residue-based start atoms, fallback to atom 1")
                start_atoms = [1]

            config_file = self._create_full_caver_config(
                output_subdir,
                start_atoms=start_atoms,
                probe_radius=probe_radius
            )
            print(f"   📝 CAVER config: {config_file}")

            cmd = [
                "java", "-jar", str(caver_jar),
                "-home", str(caver_jar.parent),
                "-pdb", str(pdb.parent),  # ⚠ FIX: must be directory, not file
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
                    results[pdb_name] = self._parse_caver_results(output_subdir, pdb_name)
                    tc = results[pdb_name]["summary"]["tunnel_count"]
                    print(f"   ✓ Tunnels found: {tc}")
                else:
                    print("   ❌ CAVER error")
                    print(result.stderr[:200])
                    results[pdb_name] = {"error": result.stderr}

            except subprocess.TimeoutExpired:
                print("   ⏱️ CAVER timeout")
                results[pdb_name] = {"error": "Timeout"}
            except Exception as e:
                print(f"   ❌ CAVER exception: {e}")
                results[pdb_name] = {"error": str(e)}

        print("\n" + "=" * 70)
        print("✅ CAVER analysis completed!")
        print("=" * 70)

        return results

    def _create_full_caver_config(self, output_dir: Path,
                                  start_atoms: List[int],
                                  probe_radius: float = 0.9) -> Path:
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
first_frame 1
last_frame 1 #如果有MD多个pdb文件，可以修改为10

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
            # Fallback: search for tunnel PDB files
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

    def generate_summary_report(self, p2rank_results: Dict, caver_results: Dict) -> pd.DataFrame:
        print("\n📋 Generating Summary Report...")

        report_rows = []
        proteins = sorted(set(p2rank_results.keys()) | set(caver_results.keys()))

        for p in proteins:
            r1 = p2rank_results.get(p, {})
            r2 = caver_results.get(p, {})

            top_score = 0.0
            top_prob = 0.0
            if r1.get("pockets"):
                top = r1["pockets"][0]
                top_score = top.get("score", 0.0)
                top_prob = top.get("probability", 0.0)

            row = {
                "Protein": p,
                "Total_Pockets": r1.get("summary", {}).get("total_pockets", 0),
                "Top_Pocket_Score": top_score,
                "Top_Pocket_Probability": top_prob,
                "Tunnel_Count": r2.get("summary", {}).get("tunnel_count", 0),
                "Avg_Throughput": round(r2.get("summary", {}).get("avg_throughput", 0.0), 3),
                "Avg_Bottleneck_Radius": round(r2.get("summary", {}).get("avg_bottleneck", 0.0), 3),
                "Avg_Length": round(r2.get("summary", {}).get("avg_length", 0.0), 2),
                "Avg_Curvature": round(r2.get("summary", {}).get("avg_curvature", 0.0), 3),
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

def main():
    analyzer = ProteinAnalyzer()
    
    analyzer.setup_environment()

    p2rank_results = analyzer.run_p2rank()

    caver_results = analyzer.run_caver(probe_radius=0.9)

    summary_df = analyzer.generate_summary_report(p2rank_results, caver_results)
    analyzer.save_detailed_results(p2rank_results, caver_results)

    return analyzer, summary_df, p2rank_results, caver_results


if __name__ == "__main__":
    analyzer, summary_df, p2rank_results, caver_results = main()
