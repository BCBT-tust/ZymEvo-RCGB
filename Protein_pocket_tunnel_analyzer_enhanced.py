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
            
        url = "https://www.caver.cz/fil/download/caver30/302/caver_3.0.2.zip"
        zip_path = self.work_dir / "caver.zip"

        try:
            subprocess.run(
                ["wget", "--no-check-certificate", "-O", str(zip_path), url],
                check=True,
                timeout=300
            )
        except Exception as e:
            raise RuntimeError(f"❌ Failed to download CAVER: {e}")

        if not zip_path.exists() or os.path.getsize(zip_path) < 1_000_000:
            raise RuntimeError("❌ Downloaded CAVER file is too small or missing")

        print("📦 Extracting CAVER...")
        try:
            subprocess.run(["unzip", "-q", str(zip_path), "-d", str(self.work_dir)], check=True)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to extract CAVER: {e}")

        zip_path.unlink(missing_ok=True)

        jars = list(self.work_dir.rglob("caver.jar"))
        if jars:
            self.caver_path = jars[0].parent
            print(f"   ✓ Found caver.jar at: {self.caver_path}")
        else:
            raise RuntimeError("❌ caver.jar not found after extraction")

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
            print("⚠️ No PDB files found for P2Rank")
            return {}

        results: Dict[str, Dict] = {}

        for i, pdb in enumerate(pdb_files, 1):
            pdb_name = pdb.stem
            print(f"\n[{i}/{len(pdb_files)}] 📊 Analyzing {pdb.name}...")

            output_subdir = self.p2rank_output / pdb_name
            output_subdir.mkdir(parents=True, exist_ok=True)

            cmd = [
                str(self.p2rank_path / "prank"),
                "predict",
                "-f", str(pdb),
                "-o", str(output_subdir),
                "-threads", str(threads)
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if result.returncode == 0:
                    print("   ✅ P2Rank completed")
                    results[pdb_name] = self._parse_p2rank_results(
                        output_subdir, pdb_name, min_score
                    )

                    if results[pdb_name]["pockets"]:
                        top = results[pdb_name]["pockets"][0]
                        print(
                            f"      ✓ Top pocket: score={top['score']:.2f}, "
                            f"prob={top['probability']:.3f}, "
                            f"center={tuple(round(x,1) for x in top['center'])}"
                        )
                else:
                    print("   ❌ P2Rank error")
                    results[pdb_name] = {"error": result.stderr}

            except subprocess.TimeoutExpired:
                print("   ⏱️ P2Rank timeout")
                results[pdb_name] = {"error": "Timeout"}
            except Exception as e:
                print(f"   ❌ P2Rank exception: {e}")
                results[pdb_name] = {"error": str(e)}

        print("\n" + "=" * 70)
        print("✅ P2Rank analysis completed!")
        print("=" * 70)

        return results

    def _parse_p2rank_results(self, output_dir: Path, pdb_name: str, min_score: float) -> Dict:
        pockets_csv = output_dir / f"{pdb_name}.pdb_predictions.csv"

        if not pockets_csv.exists():
            return {"error": "CSV file not found"}

        try:
            df = pd.read_csv(pockets_csv)
            df = df[df["score"] >= min_score]

            pockets = []
            for _, row in df.iterrows():
                pocket = {
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "probability": float(row["probability"]),
                    "center": [
                        float(row["center_x"]),
                        float(row["center_y"]),
                        float(row["center_z"])
                    ],
                    "residue_ids": row["residue_ids"]
                }
                pockets.append(pocket)

            return {
                "pdb_name": pdb_name,
                "summary": {"total_pockets": len(pockets)},
                "pockets": pockets
            }

        except Exception as e:
            return {"error": str(e)}

    def _get_atoms_from_residue_ids(self, pdb_file: Path, pdb_name: str) -> Optional[List[int]]:
        from Bio.PDB import PDBParser

        p2rank_csv = self.p2rank_output / pdb_name / f"{pdb_name}.pdb_predictions.csv"
        if not p2rank_csv.exists():
            return None

        try:
            df = pd.read_csv(p2rank_csv)
            top_pocket = df.iloc[0]
            residue_ids_str = str(top_pocket["residue_ids"])

            residue_ids = []
            for token in residue_ids_str.replace(",", " ").split():
                if token.isdigit():
                    residue_ids.append(int(token))

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", str(pdb_file))
            model = structure[0]

            found_atoms = []
            for chain in model:
                for residue in chain:
                    res_id = residue.id[1]
                    if res_id in residue_ids:
                        for atom in residue:
                            found_atoms.append(atom.serial_number)

            print(
                f"   ✓ Using {len(found_atoms)} atoms "
                f"from {len(residue_ids)} residues (from residue_ids)"
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
                "-pdb", str(pdb.parent),
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
last_frame 1

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
        """Enhanced CAVER result parser with tunnel characteristics extraction"""
        
        # Find tunnel files
        tunnel_files = []
        for pattern in ["tunnel_*.pdb", "tunnel*.pdb"]:
            tunnel_files.extend(list(output_dir.glob(pattern)))
            data_dir = output_dir / "data"
            if data_dir.exists():
                tunnel_files.extend(list(data_dir.glob(pattern)))

        tunnel_files = list(set(tunnel_files))
        
        # Parse tunnel characteristics CSV if exists
        tunnel_characteristics = None
        characteristics_csv = output_dir / "analysis" / "tunnel_characteristics.csv"
        
        if characteristics_csv.exists():
            try:
                # Read CSV with proper space handling
                df = pd.read_csv(characteristics_csv, skipinitialspace=True)
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                tunnel_characteristics = df.to_dict('records')
                print(f"      ✓ Parsed {len(tunnel_characteristics)} tunnel characteristics")
                
            except Exception as e:
                print(f"      ⚠️ Failed to parse tunnel_characteristics.csv: {e}")
        
        return {
            "pdb_name": pdb_name,
            "summary": {
                "tunnel_count": len(tunnel_files),
                "files": [str(f) for f in tunnel_files]
            },
            "tunnels": [{"id": f.stem, "file": str(f)} for f in tunnel_files],
            "tunnel_characteristics": tunnel_characteristics
        }

    def generate_summary_report(self, p2rank_results: Dict, caver_results: Dict) -> pd.DataFrame:
        """Enhanced summary with tunnel characteristics"""
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

            # Get tunnel statistics if available
            tunnel_stats = {}
            if r2.get("tunnel_characteristics"):
                chars = r2["tunnel_characteristics"]
                if chars:
                    # Calculate average characteristics
                    tunnel_stats = {
                        "Avg_Throughput": sum(t.get("Throughput", 0) for t in chars) / len(chars),
                        "Avg_Bottleneck_Radius": sum(t.get("Bottleneck radius", 0) for t in chars) / len(chars),
                        "Avg_Length": sum(t.get("Length", 0) for t in chars) / len(chars),
                        "Avg_Curvature": sum(t.get("Curvature", 0) for t in chars) / len(chars),
                    }

            row = {
                "Protein": p,
                "Total_Pockets": r1.get("summary", {}).get("total_pockets", 0),
                "Top_Pocket_Score": top_score,
                "Top_Pocket_Probability": top_prob,
                "Tunnel_Count": r2.get("summary", {}).get("tunnel_count", 0),
                **tunnel_stats,
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

    def generate_merged_pymol_script(self, pdb_name: str) -> Optional[Path]:
        """
        Generate a merged PyMOL script combining P2Rank pockets and CAVER tunnels
        """
        print(f"\n🎨 Generating merged PyMOL script for {pdb_name}...")
        
        # Paths
        p2rank_pml = self.p2rank_output / pdb_name / "visualizations" / f"{pdb_name}.pdb.pml"
        caver_pymol_dir = self.caver_output / pdb_name / "pymol"
        
        if not p2rank_pml.exists():
            print(f"   ⚠️ P2Rank PML not found: {p2rank_pml}")
            return None
        
        if not caver_pymol_dir.exists():
            print(f"   ⚠️ CAVER pymol directory not found: {caver_pymol_dir}")
            return None
        
        # Create merged visualization directory
        merged_dir = self.output_dir / "merged_visualizations" / pdb_name
        merged_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy PDB file
        pdb_file = self.input_dir / f"{pdb_name}.pdb"
        if pdb_file.exists():
            shutil.copy(pdb_file, merged_dir / f"{pdb_name}.pdb")
        
        # Read P2Rank PML content
        with open(p2rank_pml, 'r') as f:
            p2rank_content = f.read()
        
        # Find CAVER view files
        caver_view_py = caver_pymol_dir / "view.py"
        caver_zones_py = caver_pymol_dir / "zones.py"
        
        # Create merged script
        merged_script = merged_dir / f"{pdb_name}_merged.pml"
        
        with open(merged_script, 'w') as f:
            f.write("# ========================================\n")
            f.write(f"# Merged Visualization for {pdb_name}\n")
            f.write("# P2Rank Pockets + CAVER Tunnels\n")
            f.write("# ========================================\n\n")
            
            # Load protein
            f.write(f"load {pdb_name}.pdb\n")
            f.write("hide everything\n")
            f.write("show cartoon, all\n")
            f.write("color gray80, all\n\n")
            
            # Add P2Rank pockets section
            f.write("# ========================================\n")
            f.write("# P2Rank Predicted Pockets\n")
            f.write("# ========================================\n\n")
            
            # Extract pocket visualization from P2Rank PML
            # Filter out load commands and just keep pocket definitions
            for line in p2rank_content.split('\n'):
                if 'pocket' in line.lower() and not line.strip().startswith('load'):
                    f.write(line + '\n')
            
            f.write("\n")
            
            # Add CAVER tunnels section
            f.write("# ========================================\n")
            f.write("# CAVER Tunnels\n")
            f.write("# ========================================\n\n")
            
            # Copy CAVER modules to merged directory
            caver_modules_src = caver_pymol_dir / "modules"
            caver_modules_dst = merged_dir / "modules"
            if caver_modules_src.exists():
                if caver_modules_dst.exists():
                    shutil.rmtree(caver_modules_dst)
                shutil.copytree(caver_modules_src, caver_modules_dst)
            
            # Add CAVER tunnel loading
            if caver_view_py.exists():
                f.write(f"run view.py\n")
                shutil.copy(caver_view_py, merged_dir / "view.py")
            
            if caver_zones_py.exists():
                shutil.copy(caver_zones_py, merged_dir / "zones.py")
            
            # Copy atoms.py if exists
            caver_atoms_py = caver_pymol_dir / "atoms.py"
            if caver_atoms_py.exists():
                shutil.copy(caver_atoms_py, merged_dir / "atoms.py")
            
            f.write("\n")
            f.write("# Display settings\n")
            f.write("zoom\n")
            f.write("bg_color white\n")
            f.write("set ray_shadows, 0\n")
            f.write("set antialias, 2\n")
        
        print(f"   ✅ Merged PyMOL script created: {merged_script}")
        print(f"   📁 All files in: {merged_dir}")
        
        return merged_script

    def visualize_in_colab(self, pdb_name: str):
        """
        Visualize protein structure with pockets and tunnels in Google Colab using py3Dmol
        """
        try:
            import py3Dmol
        except ImportError:
            print("⚠️ py3Dmol not installed. Installing...")
            subprocess.run(["pip", "install", "-q", "py3Dmol"], check=True)
            import py3Dmol
        
        print(f"\n🔬 Generating 3D visualization for {pdb_name}...")
        
        # Load PDB file
        pdb_file = self.input_dir / f"{pdb_name}.pdb"
        if not pdb_file.exists():
            print(f"   ❌ PDB file not found: {pdb_file}")
            return None
        
        with open(pdb_file, 'r') as f:
            pdb_data = f.read()
        
        # Create viewer
        view = py3Dmol.view(width=800, height=600)
        view.addModel(pdb_data, 'pdb')
        
        # Style protein
        view.setStyle({'cartoon': {'color': 'spectrum'}})
        
        # Add P2Rank pockets if available
        p2rank_csv = self.p2rank_output / pdb_name / f"{pdb_name}.pdb_predictions.csv"
        if p2rank_csv.exists():
            try:
                df = pd.read_csv(p2rank_csv)
                for idx, row in df.iterrows():
                    if idx >= 5:  # Show top 5 pockets
                        break
                    
                    # Parse residue IDs
                    residue_ids_str = str(row["residue_ids"])
                    residue_ids = [int(x) for x in residue_ids_str.replace(",", " ").split() if x.isdigit()]
                    
                    # Add pocket visualization
                    color_map = ['red', 'orange', 'yellow', 'green', 'blue']
                    color = color_map[idx % len(color_map)]
                    
                    for res_id in residue_ids[:50]:  # Limit to first 50 residues per pocket
                        view.addStyle({'resi': res_id}, 
                                    {'stick': {'color': color, 'radius': 0.3}})
                    
                    # Add sphere at pocket center
                    view.addSphere({
                        'center': {
                            'x': float(row['center_x']),
                            'y': float(row['center_y']),
                            'z': float(row['center_z'])
                        },
                        'radius': 3.0,
                        'color': color,
                        'alpha': 0.3
                    })
                
                print(f"   ✓ Added {min(len(df), 5)} P2Rank pockets")
            except Exception as e:
                print(f"   ⚠️ Could not visualize pockets: {e}")
        
        view.zoomTo()
        view.setBackgroundColor('0xffffff')
        
        print("   ✅ Visualization ready!")
        return view


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
