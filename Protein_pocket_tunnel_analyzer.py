#!/usr/bin/env python3

import os
import subprocess
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import shutil
import re


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
        print("🔧 Setting up environment (P2Rank + CAVER 3.0.2)...")
        print("=" * 70)
        
        self._install_java()
        self._download_p2rank()
        self._download_caver()
        
        print("\n✅ Environment setup completed!")
        print("=" * 70)
    
    def _install_java(self):
        print("\n☕ Installing Java...")
        try:
            result = subprocess.run(["java", "-version"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Java already installed")
                return
        except FileNotFoundError:
            pass
        
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "openjdk-11-jdk"], check=True)
        print("✅ Java installed successfully")
    
    def _download_p2rank(self):
        print("\n📦 Downloading P2Rank v2.4.2...")
        
        if self.p2rank_path.exists():
            print("✅ P2Rank already exists")
            return
        
        p2rank_url = "https://github.com/rdk/p2rank/releases/download/2.4.2/p2rank_2.4.2.tar.gz"
        
        try:
            subprocess.run(
                ["wget", "-q", "-O", f"{self.work_dir}/p2rank.tar.gz", p2rank_url],
                check=True,
                timeout=120
            )
            
            subprocess.run(
                ["tar", "-xzf", f"{self.work_dir}/p2rank.tar.gz", "-C", str(self.work_dir)],
                check=True
            )
            
            os.remove(f"{self.work_dir}/p2rank.tar.gz")
            
            print("✅ P2Rank downloaded and extracted")
            
        except Exception as e:
            print(f"❌ Failed to download P2Rank: {e}")
            raise

    def _download_caver(self):
        print("\n📦 Downloading CAVER 3.0.2 (ZIP version)...")

        existing = list(self.work_dir.rglob("caver.jar"))
        if existing:
            self.caver_path = existing[0].parent
            print(f"✅ CAVER already exists at: {self.caver_path}")
            return

        caver_url = "https://www.caver.cz/fil/download/caver30/302/caver_3.0.2.zip"
        zip_path = f"{self.work_dir}/caver.zip"

        try:
            subprocess.run(["wget", "-q", "-O", zip_path, caver_url], check=True, timeout=120)

            if os.path.getsize(zip_path) < 50000:
                raise Exception("Downloaded CAVER ZIP too small — likely HTML error page.")

            subprocess.run(["unzip", "-o", zip_path, "-d", str(self.work_dir)], check=True)
            os.remove(zip_path)

            jar_files = list(self.work_dir.rglob("caver.jar"))
            if not jar_files:
                raise FileNotFoundError("caver.jar not found after extraction")

            self.caver_path = jar_files[0].parent

            print(f"   ✓ Found caver.jar at: {self.caver_path}")
            print("✅ CAVER 3.0.2 installed successfully")

        except Exception as e:
            print(f"❌ Failed to install CAVER 3.0.2: {e}")
            raise

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
        
        results = {}
        
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
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print("   ✅ P2Rank completed")
                    
                    pocket_info = self._parse_p2rank_results(output_subdir, pdb_file.stem)
                    validation = self.validate_p2rank_results(pocket_info, pdb_file)
                    pocket_info["validation"] = validation
                    
                    if validation["warnings"]:
                        print(f"   ⚠️  Quality: {validation['quality']}")
                        for w in validation["warnings"][:2]:
                            print(f"      - {w}")
                    else:
                        print(f"   ✓ Quality: {validation['quality']}")
                    
                    results[pdb_file.stem] = pocket_info
                    
                else:
                    print("   ❌ P2Rank failed")
                    print(f"   Error: {result.stderr[:200]}")
                    results[pdb_file.stem] = {"error": result.stderr}
                    
            except subprocess.TimeoutExpired:
                print("   ⏱️ Timeout (>300s)")
                results[pdb_file.stem] = {"error": "Timeout"}
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                results[pdb_file.stem] = {"error": str(e)}
        
        print("\n" + "=" * 70)
        print("✅ P2Rank analysis completed!")
        print("=" * 70)
        
        return results

    def _parse_p2rank_results(self, output_dir: Path, pdb_name: str) -> Dict:
        pocket_info = {"pdb_name": pdb_name, "pockets": [], "summary": {}}
        
        csv_pattern = f"{pdb_name}.pdb_predictions.csv"
        csv_files = list(output_dir.glob(csv_pattern)) or \
                    list(output_dir.glob("*predictions.csv")) or \
                    list(output_dir.glob("*.csv"))

        if not csv_files:
            print(f"      ⚠️  No CSV file found in {output_dir}")
            return pocket_info
        
        try:
            df = pd.read_csv(csv_files[0], skipinitialspace=True)
            
            print(f"      📄 CSV: {csv_files[0].name} ({len(df)} pockets)")

            pocket_info["summary"] = {
                "total_pockets": len(df),
                "output_file": str(csv_files[0])
            }

            for idx, row in df.iterrows():
                try:
                    pocket = {
                        "name": str(row.get("name", f"pocket{idx+1}")).strip(),
                        "rank": int(row.get("rank", idx + 1)),
                        "score": float(row.get("score", 0.0)),
                        "probability": float(row.get("probability", 0.0)),
                        "sas_points": int(row.get("sas_points", row.get("sas_poin", 0))),
                        "surf_atoms": int(row.get("surf_atoms", row.get("surf_ato", 0))),
                        "center_x": float(row.get("center_x", 0.0)),
                        "center_y": float(row.get("center_y", 0.0)),
                        "center_z": float(row.get("center_z", 0.0)),
                    }
                    
                    if "residue_ids" in df.columns and pd.notna(row["residue_ids"]):
                        pocket["residues"] = str(row["residue_ids"]).strip()
                    else:
                        pocket["residues"] = ""
                    
                    pocket_info["pockets"].append(pocket)
                    
                except (ValueError, TypeError) as e:
                    print(f"      ⚠️  Warning parsing pocket {idx+1}: {e}")
                    continue

            if pocket_info["pockets"]:
                top = pocket_info["pockets"][0]
                print(f"      ✓ Top pocket: score={top['score']:.2f}, "
                      f"prob={top['probability']:.3f}, "
                      f"center=({top['center_x']:.1f}, {top['center_y']:.1f}, {top['center_z']:.1f})")

        except Exception as e:
            pocket_info["parse_error"] = str(e)
            print(f"      ❌ Failed to parse CSV: {e}")
            import traceback
            traceback.print_exc()

        return pocket_info

    def validate_p2rank_results(self, pocket_info: Dict, pdb_file: Path) -> Dict:
        warnings = []
        quality = "high"
        
        if "parse_error" in pocket_info:
            warnings.append(f"Parse error: {pocket_info['parse_error']}")
            return {"quality": "failed", "warnings": warnings}
        
        if not pocket_info.get("pockets"):
            return {"quality": "low", "warnings": ["No pockets detected"]}
        
        top = pocket_info["pockets"][0]

        if top["score"] == 0.0:
            warnings.append("Zero score - parsing error")
            quality = "failed"
        elif top["score"] < 2.0:
            warnings.append(f"Low score {top['score']:.2f}")
            quality = "low"
        elif top["score"] < 5.0:
            warnings.append(f"Medium score {top['score']:.2f}")
            quality = "medium"
            
        if top["probability"] < 0.3:
            warnings.append(f"Low probability {top['probability']:.3f}")
            if quality == "high":
                quality = "medium"

        coords_sum = abs(top["center_x"]) + abs(top["center_y"]) + abs(top["center_z"])
        if coords_sum == 0.0:
            warnings.append("Center at origin (0,0,0)")
            quality = "failed"

        return {"quality": quality, "warnings": warnings}

    def run_caver(self, pdb_files: List[str] = None,
                  use_p2rank_pockets: bool = True,
                  probe_radius: float = 0.9,
                  max_distance: float = 3.0) -> Dict[str, Dict]:

        print("\n" + "=" * 70)
        print("🌀 Running CAVER 3.0.2 Analysis...")
        print("=" * 70)
        
        if pdb_files is None:
            pdb_files = list(self.input_dir.glob("*.pdb"))
        else:
            pdb_files = [Path(f) for f in pdb_files]

        if not pdb_files:
            print("⚠️ No PDB files found!")
            return {}
        
        results = {}
        
        caver_jar = self._find_caver_jar()
        if caver_jar is None:
            print("❌ CAVER JAR file not found!")
            return {}
        
        print(f"✅ Using CAVER JAR: {caver_jar}")

        for i, pdb_file in enumerate(pdb_files, 1):
            print(f"\n[{i}/{len(pdb_files)}] 🌀 {pdb_file.name}")

            output_subdir = self.caver_output / pdb_file.stem
            output_subdir.mkdir(exist_ok=True)

            temp_pdb_dir = output_subdir / "pdb_input"
            temp_pdb_dir.mkdir(exist_ok=True)
            
            temp_pdb_file = temp_pdb_dir / pdb_file.name
            shutil.copy2(pdb_file, temp_pdb_file)

            start_point = None
            if use_p2rank_pockets:
                start_point = self._get_p2rank_start_point(pdb_file.stem)
            
            config_file = self._create_caver_config(
                output_subdir,
                start_point,
                probe_radius,
                max_distance
            )
            
            if start_point:
                print(f"   ✓ Using P2Rank pocket center {start_point}")
            else:
                print("   ⚠ Auto-detection mode (no P2Rank start point)")

            # CAVER命令行格式
            cmd = [
                "java", "-jar", str(caver_jar),
                "-home", str(caver_jar.parent),  # CAVER安装目录
                "-pdb", str(temp_pdb_dir),       # PDB文件目录(不是文件!)
                "-conf", str(config_file),       # 配置文件
                "-out", str(output_subdir)       # 输出目录
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    print("   ✓ CAVER completed")
                    tunnel_info = self._parse_caver_results(output_subdir, pdb_file.stem)
                    results[pdb_file.stem] = tunnel_info
                    
                    tunnel_count = tunnel_info.get("summary", {}).get("tunnel_count", 0)
                    print(f"      Found {tunnel_count} tunnel(s)")

                else:
                    print("   ❌ CAVER failed")
                    print(f"      stdout: {result.stdout[:150]}")
                    print(f"      stderr: {result.stderr[:150]}")
                    results[pdb_file.stem] = {"error": result.stderr}

            except subprocess.TimeoutExpired:
                print("   ⏱ Timeout (>300s)")
                results[pdb_file.stem] = {"error": "Timeout"}
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                results[pdb_file.stem] = {"error": str(e)}
            finally:

                if temp_pdb_dir.exists():
                    shutil.rmtree(temp_pdb_dir)

        print("\n" + "=" * 70)
        print("✅ CAVER tunnel detection completed!")
        print("=" * 70)

        return results

    def _create_caver_config(self, output_dir: Path, 
                            start_point: Optional[Tuple[float, float, float]],
                            probe_radius: float,
                            max_distance: float) -> Path:
        config_file = output_dir / "config.txt"
        
        if start_point:
            config_content = f"""# CAVER 3.0 Configuration File
# Generated by Protein Analyzer

# Starting point coordinates (from P2Rank)
starting_point_x {start_point[0]:.3f}
starting_point_y {start_point[1]:.3f}
starting_point_z {start_point[2]:.3f}

# Calculation parameters
probe_radius {probe_radius}
shell_depth 4
shell_radius 3
max_distance {max_distance}
desired_radius 1.4

# Output options
save_pdb yes
"""
        else:
            config_content = f"""# CAVER 3.0 Configuration File
# Generated by Protein Analyzer
# Auto-detection mode

# Calculation parameters
probe_radius {probe_radius}
shell_depth 4
shell_radius 3
max_distance {max_distance}
desired_radius 1.4

# Output options
save_pdb yes
"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        return config_file

    def _find_caver_jar(self) -> Optional[Path]:
        candidates = list(self.work_dir.rglob("caver.jar"))
        if not candidates:
            return None
        return candidates[0]

    def _get_p2rank_start_point(self, pdb_name: str) -> Optional[Tuple[float, float, float]]:
        p2rank_dir = self.p2rank_output / pdb_name
        csv_files = list(p2rank_dir.glob("*predictions.csv"))
        if not csv_files:
            return None

        try:
            df = pd.read_csv(csv_files[0], skipinitialspace=True)
            if df.empty:
                return None
            
            row = df.iloc[0]
            x = float(row.get("center_x", 0))
            y = float(row.get("center_y", 0))
            z = float(row.get("center_z", 0))
            
            if x == 0 and y == 0 and z == 0:
                return None

            return (x, y, z)
        except Exception as e:
            print(f"      ⚠️ Failed to get start point: {e}")
            return None

    def _parse_caver_results(self, output_dir: Path, pdb_name: str) -> Dict:
        """解析CAVER输出结果"""
        tunnel_files = list(output_dir.glob("tunnel_*.pdb"))
        
        if not tunnel_files:
            tunnel_files = list(output_dir.glob("tunnel*.pdb"))

        return {
            "pdb_name": pdb_name,
            "summary": {
                "tunnel_count": len(tunnel_files),
                "files": [str(f) for f in tunnel_files]
            },
            "tunnels": [{"id": f.stem, "file": str(f)} for f in tunnel_files]
        }

    def generate_summary_report(self, p2rank_results: Dict, caver_results: Dict) -> pd.DataFrame:
        print("\n📋 Generating Summary Report...")

        report = []
        proteins = sorted(set(p2rank_results.keys()) | set(caver_results.keys()))

        for p in proteins:
            r1 = p2rank_results.get(p, {})
            r2 = caver_results.get(p, {})

            top_score = 0.0
            top_prob = 0.0
            top_center_x = 0.0
            top_center_y = 0.0
            top_center_z = 0.0
            quality = "unknown"
            warnings_str = ""
            
            if r1.get("pockets"):
                top = r1["pockets"][0]
                top_score = top.get("score", 0.0)
                top_prob = top.get("probability", 0.0)
                top_center_x = top.get("center_x", 0.0)
                top_center_y = top.get("center_y", 0.0)
                top_center_z = top.get("center_z", 0.0)
            
            if r1.get("validation"):
                quality = r1["validation"].get("quality", "unknown")
                warnings = r1["validation"].get("warnings", [])
                warnings_str = "; ".join(warnings[:3])  

            row = {
                "Protein": p,
                "Total_Pockets": r1.get("summary", {}).get("total_pockets", 0),
                "Top_Pocket_Score": top_score,
                "Top_Pocket_Probability": top_prob,
                "Pocket_Center_X": top_center_x,
                "Pocket_Center_Y": top_center_y,
                "Pocket_Center_Z": top_center_z,
                "Tunnel_Count": r2.get("summary", {}).get("tunnel_count", 0),
                "P2Rank_Status": "Success" if "error" not in r1 and "parse_error" not in r1 else "Failed",
                "CAVER_Status": "Success" if "error" not in r2 else "Failed",
                "Quality": quality,
                "Warnings": warnings_str
            }
            report.append(row)

        df = pd.DataFrame(report)
        df.to_csv(self.output_dir / "summary_report.csv", index=False)

        print("✅ Summary saved")
        return df

    def save_detailed_results(self, p2rank_results: Dict, caver_results: Dict):
        data = {"p2rank": p2rank_results, "caver": caver_results}
        with open(self.output_dir / "detailed_results.json", "w") as f:
            json.dump(data, f, indent=2)
        print("📄 Detailed results saved")


def main():
    analyzer = ProteinAnalyzer()
    
    analyzer.setup_environment()
    
    p2rank_results = analyzer.run_p2rank()
    caver_results = analyzer.run_caver(use_p2rank_pockets=True)

    df = analyzer.generate_summary_report(p2rank_results, caver_results)
    analyzer.save_detailed_results(p2rank_results, caver_results)

    return analyzer, df


if __name__ == "__main__":
    analyzer, summary = main()
