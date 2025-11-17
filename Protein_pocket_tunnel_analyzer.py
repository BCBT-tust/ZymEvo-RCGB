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
        self.caver_path = self.work_dir / "caver_3.02"

        self._create_directories()
        
    def _create_directories(self):
        for directory in [self.input_dir, self.output_dir, 
                         self.p2rank_output, self.caver_output]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def setup_environment(self):
        print("=" * 70)
        print("🔧 Setting up environment (P2Rank + CAVER 3.0)...")
        print("=" * 70)
        
        self._install_java()
        
        self._download_p2rank()
        
        self._download_caver()
        
        print("\n✅ Environment setup completed!")
        print("=" * 70)
    
    def _install_java(self):
        print("\n☕ Installing Java...")
        try:
            result = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✅ Java already installed")
                return
        except FileNotFoundError:
            pass
        
        subprocess.run(
            ["apt-get", "update", "-qq"],
            check=True,
            capture_output=True
        )
        subprocess.run(
            ["apt-get", "install", "-y", "-qq", "openjdk-11-jdk"],
            check=True,
            capture_output=True
        )
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
        print("\n📦 Downloading CAVER 3.0...")
        
        if self.caver_path.exists():
            print("✅ CAVER already exists")
            return
        
        # CAVER 3.02下载链接
        caver_url = "http://www.caver.cz/download/caver_3.02.tar.gz"

            subprocess.run(
                ["wget", "-q", "-O", f"{self.work_dir}/caver.tar.gz", caver_url],
                check=True,
                timeout=120
            )
            
            subprocess.run(
                ["tar", "-xzf", f"{self.work_dir}/caver.tar.gz", "-C", str(self.work_dir)],
                check=True
            )
            
            os.remove(f"{self.work_dir}/caver.tar.gz")

            caver_jar = self.caver_path / "caver.jar"
            if not caver_jar.exists():
                # 尝试查找JAR文件
                jar_files = list(self.caver_path.rglob("*.jar"))
                if jar_files:
                    print(f"   Found CAVER JAR: {jar_files[0].name}")
                else:
                    raise FileNotFoundError("CAVER JAR not found")
            
            print("✅ CAVER 3.0 downloaded and extracted")
            
        except Exception as e:
            print(f"❌ Failed to download CAVER: {e}")
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
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    print(f"   ✅ P2Rank completed")
                    
                    pocket_info = self._parse_p2rank_results(output_subdir, pdb_file.stem)
                    
                    validation = self.validate_p2rank_results(pocket_info, pdb_file)
                    pocket_info["validation"] = validation
                    
                    if validation["warnings"]:
                        print(f"   ⚠️  Quality: {validation['quality']}")
                        for warning in validation["warnings"][:2]:  # 只显示前2个警告
                            print(f"      - {warning}")
                    else:
                        print(f"   ✓ Quality: {validation['quality']}")
                    
                    results[pdb_file.stem] = pocket_info
                    
                else:
                    print(f"   ❌ P2Rank failed")
                    print(f"   Error: {result.stderr[:200]}")
                    results[pdb_file.stem] = {"error": result.stderr}
                    
            except subprocess.TimeoutExpired:
                print(f"   ⏱️  Timeout (>300s)")
                results[pdb_file.stem] = {"error": "Timeout"}
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                results[pdb_file.stem] = {"error": str(e)}
        
        print("\n" + "=" * 70)
        print("✅ P2Rank analysis completed!")
        print("=" * 70)
        
        return results
    
    def _parse_p2rank_results(self, output_dir: Path, pdb_name: str) -> Dict:
        pocket_info = {
            "pdb_name": pdb_name,
            "pockets": [],
            "summary": {}
        }
        
        csv_pattern = f"{pdb_name}.pdb_predictions.csv"
        csv_files = list(output_dir.glob(csv_pattern))
        
        if not csv_files:
            csv_files = list(output_dir.glob("*predictions.csv"))
        
        if not csv_files:
            csv_files = list(output_dir.glob("*.csv"))
        
        if not csv_files:
            print(f"      ⚠️  No CSV file found in {output_dir}")
            return pocket_info
        
        try:
            df = pd.read_csv(csv_files[0])
            
            print(f"      📄 CSV: {csv_files[0].name} ({len(df)} pockets)")
            
            pocket_info["summary"] = {
                "total_pockets": len(df),
                "output_file": str(csv_files[0])
            }
            
            for idx, row in df.iterrows():
                pocket = {}
                
                try:
                    pocket["name"] = str(row["name"]) if "name" in df.columns else f"pocket{idx+1}"
                    pocket["rank"] = int(row["rank"]) if "rank" in df.columns else idx + 1
                    pocket["score"] = float(row["score"]) if "score" in df.columns else 0.0
                    pocket["probability"] = float(row["probability"]) if "probability" in df.columns else 0.0
                    
                    if "sas_points" in df.columns:
                        pocket["sas_points"] = int(row["sas_points"])
                    elif "sas_poin" in df.columns:
                        pocket["sas_points"] = int(row["sas_poin"])
                    else:
                        pocket["sas_points"] = 0
                    
                    if "surf_atoms" in df.columns:
                        pocket["surf_atoms"] = int(row["surf_atoms"])
                    elif "surf_ato" in df.columns:
                        pocket["surf_atoms"] = int(row["surf_ato"])
                    else:
                        pocket["surf_atoms"] = 0
                    
                    pocket["center_x"] = float(row["center_x"]) if "center_x" in df.columns else 0.0
                    pocket["center_y"] = float(row["center_y"]) if "center_y" in df.columns else 0.0
                    pocket["center_z"] = float(row["center_z"]) if "center_z" in df.columns else 0.0
                    
                    residue_col = None
                    for possible_name in ["residue_ids", "residue_", "residues"]:
                        if possible_name in df.columns:
                            residue_col = possible_name
                            break
                    
                    if residue_col and pd.notna(row[residue_col]):
                        pocket["residues"] = str(row[residue_col])
                    else:
                        pocket["residues"] = ""

                    if "surf_atom_ids" in df.columns and pd.notna(row["surf_atom_ids"]):
                        pocket["surf_atom_ids"] = str(row["surf_atom_ids"])
                    
                except (KeyError, ValueError, TypeError) as e:
                    print(f"      ⚠️  Warning parsing pocket {idx+1}: {e}")
                    continue
                
                pocket_info["pockets"].append(pocket)
            
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
            quality = "failed"
            return {"quality": quality, "warnings": warnings}
        
        if not pocket_info.get("pockets"):
            warnings.append("No pockets detected")
            quality = "low"
            return {"quality": quality, "warnings": warnings}
        
        top_pocket = pocket_info["pockets"][0]
        
        if top_pocket["score"] == 0.0:
            warnings.append("Top pocket has zero score - possible parsing error")
            quality = "failed"
        elif top_pocket["score"] < 2.0:
            warnings.append(f"Top pocket score ({top_pocket['score']:.2f}) is very low")
            quality = "low"
        elif top_pocket["score"] < 5.0:
            warnings.append(f"Top pocket score ({top_pocket['score']:.2f}) is low")
            quality = "medium"

        if top_pocket["probability"] == 0.0:
            warnings.append("Top pocket has zero probability - possible parsing error")
            quality = "failed"
        elif top_pocket["probability"] < 0.3:
            warnings.append(f"Top pocket probability ({top_pocket['probability']:.3f}) is low")
            quality = "medium"
        elif top_pocket["probability"] > 1.0:
            warnings.append(f"Top pocket probability ({top_pocket['probability']:.3f}) > 1.0 - data error")
            quality = "failed"
        
        coords_sum = abs(top_pocket["center_x"]) + abs(top_pocket["center_y"]) + abs(top_pocket["center_z"])
        if coords_sum == 0.0:
            warnings.append("Pocket center at origin (0,0,0) - possible error")
            quality = "failed"
        elif coords_sum > 1000:
            warnings.append(f"Pocket center far from origin (sum={coords_sum:.1f})")
            quality = "medium"
        
        if top_pocket.get("sas_points", 0) < 10:
            warnings.append(f"Very few SAS points ({top_pocket.get('sas_points', 0)})")
            quality = "medium"
        
        mutation_patterns = re.findall(r'([A-Z])(\d+)([A-Z])', pdb_file.stem)
        if mutation_patterns and top_pocket.get("residues"):
            residues_str = top_pocket["residues"]
            mutation_sites = [int(m[1]) for m in mutation_patterns]
            
            found_mutations = []
            missing_mutations = []
            
            for site in mutation_sites:
                if str(site) in residues_str:
                    found_mutations.append(site)
                else:
                    missing_mutations.append(site)
            
            if missing_mutations and not found_mutations:
                warnings.append(
                    f"Mutation site(s) {missing_mutations} not in top pocket"
                )
                quality = "medium"

        total_pockets = pocket_info["summary"].get("total_pockets", 0)
        if total_pockets > 15:
            warnings.append(f"Unusually high number of pockets ({total_pockets})")
        
        return {"quality": quality, "warnings": warnings}
    
    def run_caver(self, pdb_files: List[str] = None,
                  use_p2rank_pockets: bool = True,
                  probe_radius: float = 0.9,
                  max_distance: float = 3.0) -> Dict[str, Dict]:

        print("\n" + "=" * 70)
        print("🌀 Running CAVER 3.0 Analysis...")
        print("=" * 70)
        
        if pdb_files is None:
            pdb_files = list(self.input_dir.glob("*.pdb"))
        else:
            pdb_files = [Path(f) for f in pdb_files]
        
        if not pdb_files:
            print("⚠️  No PDB files found!")
            return {}
        
        results = {}
        
        caver_jar = self._find_caver_jar()
        if caver_jar is None:
            print("❌ CAVER JAR file not found!")
            print(f"   Searched in: {self.caver_path}")
            return {}
        
        print(f"✅ Using CAVER JAR: {caver_jar.name}")
        
        for i, pdb_file in enumerate(pdb_files, 1):
            print(f"\n[{i}/{len(pdb_files)}] 🌀 Analyzing tunnels in {pdb_file.name}...")
            
            output_subdir = self.caver_output / pdb_file.stem
            output_subdir.mkdir(exist_ok=True)
            
            cmd = [
                "java", "-jar", str(caver_jar),
                "-p", str(pdb_file),
                "-out", str(output_subdir),
                "-pr", str(probe_radius)
            ]
            
            if use_p2rank_pockets:
                p2rank_result_dir = self.p2rank_output / pdb_file.stem
                
                start_point = self._get_p2rank_start_point(pdb_file.stem)
                
                if start_point:
                    cmd.extend([
                        "-s", f"{start_point[0]:.2f}",
                        f"{start_point[1]:.2f}",
                        f"{start_point[2]:.2f}"
                    ])
                    print(f"   ✓ Using P2Rank pocket center as starting point")
                    print(f"      Start: ({start_point[0]:.1f}, {start_point[1]:.1f}, {start_point[2]:.1f})")
                else:
                    print(f"   ⚠️  No P2Rank pocket found, using auto mode")
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    print(f"   ✅ CAVER completed")
                    
                    tunnel_info = self._parse_caver_results(output_subdir, pdb_file.stem)
                    results[pdb_file.stem] = tunnel_info

                    tunnel_count = tunnel_info.get("summary", {}).get("tunnel_count", 0)
                    if tunnel_count > 0:
                        print(f"      ✓ Found {tunnel_count} tunnel(s)")
                    else:
                        print(f"      ℹ️  No tunnels detected")
                    
                else:
                    print(f"   ❌ CAVER failed")
                    print(f"   Error: {result.stderr[:200]}")
                    results[pdb_file.stem] = {"error": result.stderr}
                    
            except subprocess.TimeoutExpired:
                print(f"   ⏱️  Timeout (>300s)")
                results[pdb_file.stem] = {"error": "Timeout"}
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                results[pdb_file.stem] = {"error": str(e)}
        
        print("\n" + "=" * 70)
        print("✅ CAVER 3.0 analysis completed!")
        print("=" * 70)
        
        return results
    
    def _find_caver_jar(self) -> Optional[Path]:

        standard_jar = self.caver_path / "caver.jar"
        if standard_jar.exists():
            return standard_jar
        
        jar_files = list(self.caver_path.rglob("*.jar"))
        
        if not jar_files:
            return None

        candidates = []
        for jar in jar_files:
            score = 0
            jar_name_lower = jar.name.lower()
            
            if "caver" in jar_name_lower:
                score += 10
            if jar.stat().st_size > 100000:  # >100KB
                score += 5
            
            candidates.append((score, jar))
        
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        if candidates:
            return candidates[0][1]
        
        return None
    
    def _get_p2rank_start_point(self, pdb_name: str) -> Optional[Tuple[float, float, float]]:
        try:
            csv_pattern = f"{pdb_name}.pdb_predictions.csv"
            p2rank_dir = self.p2rank_output / pdb_name
            csv_files = list(p2rank_dir.glob(csv_pattern))
            
            if not csv_files:
                csv_files = list(p2rank_dir.glob("*predictions.csv"))
            
            if not csv_files:
                return None

            df = pd.read_csv(csv_files[0])
            
            if len(df) == 0:
                return None

            first_row = df.iloc[0]
            x = float(first_row.get("center_x", 0))
            y = float(first_row.get("center_y", 0))
            z = float(first_row.get("center_z", 0))
            
            if x == 0 and y == 0 and z == 0:
                return None
            
            return (x, y, z)
            
        except Exception as e:
            print(f"      ⚠️  Failed to get P2Rank start point: {e}")
            return None
    
    def _parse_caver_results(self, output_dir: Path, pdb_name: str) -> Dict:
        tunnel_info = {
            "pdb_name": pdb_name,
            "tunnels": [],
            "summary": {}
        }
        
        summary_file = output_dir / "summary.txt"
        tunnel_files = list(output_dir.glob("tunnel_*.pdb"))
        
        tunnel_info["summary"] = {
            "tunnel_count": len(tunnel_files),
            "output_files": [str(f) for f in tunnel_files]
        }
        
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary_text = f.read()
                    tunnel_info["summary"]["summary_text"] = summary_text[:500]  # 前500字符
            except Exception as e:
                print(f"      ⚠️  Failed to read summary: {e}")
        
        for tunnel_file in tunnel_files:
            tunnel_id = tunnel_file.stem.replace("tunnel_", "")
            tunnel_info["tunnels"].append({
                "id": tunnel_id,
                "file": str(tunnel_file)
            })
        
        return tunnel_info
    
    def generate_summary_report(self, p2rank_results: Dict, 
                               caver_results: Dict) -> pd.DataFrame:
        print("\n" + "=" * 70)
        print("📋 Generating Summary Report...")
        print("=" * 70)
        
        report_data = []
        
        all_proteins = set(p2rank_results.keys()) | set(caver_results.keys())
        
        for protein in sorted(all_proteins):
            p2rank_data = p2rank_results.get(protein, {})
            caver_data = caver_results.get(protein, {})
            
            row = {
                "Protein": protein,
                "Total_Pockets": p2rank_data.get("summary", {}).get("total_pockets", 0),
                "Top_Pocket_Score": 0.0,
                "Top_Pocket_Probability": 0.0,
                "Top_Pocket_SAS_Points": 0,
                "Top_Pocket_Center_X": 0.0,
                "Top_Pocket_Center_Y": 0.0,
                "Top_Pocket_Center_Z": 0.0,
                "Tunnel_Count": caver_data.get("summary", {}).get("tunnel_count", 0),
                "P2Rank_Status": "Success" if "error" not in p2rank_data and "parse_error" not in p2rank_data else "Failed",
                "CAVER_Status": "Success" if "error" not in caver_data else "Failed",
                "Quality": "unknown",
                "Warnings": ""
            }
            
            if p2rank_data.get("pockets"):
                top_pocket = p2rank_data["pockets"][0]
                row["Top_Pocket_Score"] = top_pocket.get("score", 0.0)
                row["Top_Pocket_Probability"] = top_pocket.get("probability", 0.0)
                row["Top_Pocket_SAS_Points"] = top_pocket.get("sas_points", 0)
                row["Top_Pocket_Center_X"] = top_pocket.get("center_x", 0.0)
                row["Top_Pocket_Center_Y"] = top_pocket.get("center_y", 0.0)
                row["Top_Pocket_Center_Z"] = top_pocket.get("center_z", 0.0)
            
            if p2rank_data.get("validation"):
                row["Quality"] = p2rank_data["validation"]["quality"]
                if p2rank_data["validation"]["warnings"]:
                    row["Warnings"] = "; ".join(p2rank_data["validation"]["warnings"])
            
            report_data.append(row)
        
        df = pd.DataFrame(report_data)
        
        report_path = self.output_dir / "summary_report.csv"
        df.to_csv(report_path, index=False)
        
        print(f"\n✅ Summary report saved to: {report_path}")
        print("\n" + "=" * 70)
        
        return df
    
    def save_detailed_results(self, p2rank_results: Dict, caver_results: Dict):
        results = {
            "p2rank": p2rank_results,
            "caver": caver_results
        }
        
        json_path = self.output_dir / "detailed_results.json"
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Detailed results saved to: {json_path}")


def main():
    analyzer = ProteinAnalyzer()
    
    analyzer.setup_environment()
    
    p2rank_results = analyzer.run_p2rank()
    caver_results = analyzer.run_caver(use_p2rank_pockets=True)

    summary_df = analyzer.generate_summary_report(p2rank_results, caver_results)
    analyzer.save_detailed_results(p2rank_results, caver_results)
    
    return analyzer, summary_df


if __name__ == "__main__":
    analyzer, summary = main()
