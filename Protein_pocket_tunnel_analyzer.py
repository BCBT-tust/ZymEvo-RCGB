#!/usr/bin/env python3

import os
import subprocess
import json
import pandas as pd
import numpy as np
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
        tar_path = f"{self.work_dir}/p2rank.tar.gz"

        print("⏳ Downloading from accelerated mirror...")
        subprocess.run(
            ["wget", "-O", tar_path, p2rank_url],
            check=True,
            timeout=600
        )

        if not os.path.exists(tar_path):
            raise Exception("P2Rank download failed")

        if os.path.getsize(tar_path) < 100_000_000:
            raise Exception("P2Rank package too small - likely download error")

        print("📦 Extracting P2Rank...")
        subprocess.run(["tar", "-xzf", tar_path, "-C", str(self.work_dir)], check=True)
        os.remove(tar_path)

        print("✅ P2Rank downloaded and extracted")

    def _download_caver(self):
        print("\n📦 Downloading CAVER 3.0.2 (ZIP version)...")

        existing = list(self.work_dir.rglob("caver.jar"))
        if existing:
            self.caver_path = existing[0].parent
            print(f"✅ CAVER already exists at: {self.caver_path}")
            return

        caver_url = "https://www.caver.cz/fil/download/caver30/302/caver_3.0.2.zip"
        zip_path = f"{self.work_dir}/caver.zip"

        subprocess.run(["wget", "-q", "-O", zip_path, caver_url], check=True, timeout=120)

        if os.path.getsize(zip_path) < 50000:
            raise Exception("Downloaded CAVER ZIP too small")

        subprocess.run(["unzip", "-o", zip_path, "-d", str(self.work_dir)], check=True)
        os.remove(zip_path)

        jar_files = list(self.work_dir.rglob("caver.jar"))
        if not jar_files:
            raise FileNotFoundError("caver.jar not found")

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

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
            if result.returncode == 0:
                print("   ✅ P2Rank completed")
                pocket_info = self._parse_p2rank_results(output_subdir, pdb_file.stem)
                results[pdb_file.stem] = pocket_info
                    
            else:
                print("   ❌ P2Rank failed")
                print(result.stderr[:200])
                results[pdb_file.stem] = {"error": result.stderr}

        print("\n" + "=" * 70)
        print("✅ P2Rank analysis completed!")
        print("=" * 70)
        
        return results

    def _parse_p2rank_results(self, output_dir: Path, pdb_name: str) -> Dict:
        pocket_info = {"pdb_name": pdb_name, "pockets": [], "summary": {}}
        
        csv_files = list(output_dir.glob("*predictions.csv"))
        if not csv_files:
            print(f"⚠️ No CSV found in {output_dir}")
            return pocket_info
        
        df = pd.read_csv(csv_files[0], skipinitialspace=True)
        pocket_info["summary"] = {
            "total_pockets": len(df),
            "output_file": str(csv_files[0])
        }

        for idx, row in df.iterrows():
            pocket = {
                "name": row.get("name", f"pocket{idx+1}"),
                "rank": int(row.get("rank", idx+1)),
                "score": float(row.get("score", 0)),
                "probability": float(row.get("probability", 0)),
                "center_x": float(row.get("center_x", 0)),
                "center_y": float(row.get("center_y", 0)),
                "center_z": float(row.get("center_z", 0)),
                "residue_ids": str(row.get("residue_ids", "")),
            }
            pocket_info["pockets"].append(pocket)

        return pocket_info

    def _get_atoms_from_residue_ids(self, pdb_file: Path, pdb_name: str) -> Optional[List[int]]:

        p2rank_csv = list((self.p2rank_output / pdb_name).glob("*predictions.csv"))
        if not p2rank_csv:
            return None

        df = pd.read_csv(p2rank_csv[0], skipinitialspace=True)
        residues_str = str(df.iloc[0].get("residue_ids", "")).strip()

        if not residues_str:
            return None

        residues = []
        for item in residues_str.split():
            try:
                chain, num = item.split("_")
                residues.append((chain, int(num)))
            except:
                pass

        found_atoms = []
        with open(pdb_file, "r") as f:
            for line in f:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                
                chain = line[21].strip()
                try:
                    resid = int(line[22:26])
                    atom_id = int(line[6:11])
                except:
                    continue

                if (chain, resid) in residues:
                    found_atoms.append(atom_id)

        print(f"   ✓ Using {len(found_atoms)} atoms from {len(residues)} pocket residues")
        return found_atoms

    def run_caver(self, pdb_files: List[str] = None,
                  use_p2rank_pockets: bool = True,
                  atom_id_strategy: str = "use_residue_ids",
                  probe_radius: float = 0.9) -> Dict[str, Dict]:

        print("\n" + "=" * 70)
        print("🌀 Running CAVER 3.0.2 Analysis...")
        print("=" * 70)

        if pdb_files is None:
            pdb_files = list(self.input_dir.glob("*.pdb"))
        else:
            pdb_files = [Path(f) for f in pdb_files]

        caver_jar = self._find_caver_jar()
        if not caver_jar:
            print("❌ caver.jar not found")
            return {}

        results = {}

        for pdb in pdb_files:
            pdb_name = pdb.stem
            print(f"\n🌀 Processing {pdb.name}")

            start_atoms = self._get_atoms_from_residue_ids(pdb, pdb_name)
            if not start_atoms:
                print("⚠️ No residue-based atoms, fallback to atom 1")
                start_atoms = [1]

            output_subdir = self.caver_output / pdb_name
            output_subdir.mkdir(exist_ok=True)

            config_file = self._create_caver_config(output_subdir, start_atoms, probe_radius)

            cmd = [
                "java", "-jar", str(caver_jar),
                "-home", str(caver_jar.parent),
                "-pdb", str(pdb),
                "-conf", str(config_file),
                "-out", str(output_subdir),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print("   ✓ CAVER finished")
                results[pdb_name] = self._parse_caver_results(output_subdir, pdb_name)
            else:
                print("   ❌ CAVER error")
                results[pdb_name] = {"error": result.stderr}

        return results

    def _create_caver_config(self, output_dir, start_atoms, probe_radius):
        config_file = output_dir / "config.txt"

        txt = "# CAVER config\n"
        for atom in start_atoms:
            txt += f"starting_point_atom {atom}\n"

        txt += f"""
probe_radius {probe_radius}
generate_summary yes
"""

        with open(config_file, "w") as f:
            f.write(txt)

        return config_file

    def _find_caver_jar(self):
        jars = list(self.work_dir.rglob("caver.jar"))
        return jars[0] if jars else None

    def _parse_caver_results(self, output_dir, pdb_name):
        tunnel_files = list(output_dir.rglob("tunnel*.pdb"))
        return {
            "pdb_name": pdb_name,
            "summary": {
                "tunnel_count": len(tunnel_files),
                "files": [str(f) for f in tunnel_files]
            },
            "tunnels": [{"id": f.stem, "file": str(f)} for f in tunnel_files]
        }

def main():
    analyzer = ProteinAnalyzer()
    
    analyzer.setup_environment()
    
    p2rank_results = analyzer.run_p2rank()
    
    caver_results = analyzer.run_caver(
        use_p2rank_pockets=True,
        atom_id_strategy="use_residue_ids"
    )

    print("\n🎉 All processing finished.")
    return analyzer, p2rank_results, caver_results


if __name__ == "__main__":
    analyzer, p2, cav = main()
