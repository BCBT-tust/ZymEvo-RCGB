#!/usr/bin/env python3
"""
ZymEvo AutoPre-Dock - FIXED VERSION
Correct parameters for prepare_flexreceptor4.py: -g (not -o)
"""

import os
import sys
import subprocess
import shutil
import threading
import time
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

class PDBTools:

    @staticmethod
    def fix_missing_chain_id(pdb_file: str, chain_id: str = "A") -> bool:
        """Automatically fix missing chain ID in PDB ATOM/HETATM lines."""
        try:
            with open(pdb_file, "r") as f:
                lines = f.readlines()

            fixed = []
            changed = False

            for line in lines:
                if line.startswith(("ATOM", "HETATM")):
                    if len(line) >= 22:
                        chain_char = line[21]
                        if chain_char in (" ", "_"):
                            line = line[:21] + chain_id + line[22:]
                            changed = True
                fixed.append(line)

            if changed:
                with open(pdb_file, "w") as f:
                    f.writelines(fixed)

            return changed

        except Exception as e:
            print(f"[ChainFix] Error fixing chain ID: {e}")
            return False

class Config:
    """Global configuration for MGLTools paths and environment"""
    
    MGLTOOLS_PATH = "/usr/local/autodocktools/bin/pythonsh"
    PREPARE_RECEPTOR = "/usr/local/autodocktools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"
    PREPARE_LIGAND = "/usr/local/autodocktools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"
    PREPARE_FLEXRECEPTOR = "/usr/local/autodocktools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_flexreceptor4.py"
    
    PYTHONPATH = "/usr/local/autodocktools/MGLToolsPckgs"
    
    DEFAULT_WORKERS = 4
    TIMEOUT_SECONDS = 300
    
    @classmethod
    def setup_environment(cls):
        os.environ['PYTHONPATH'] = cls.PYTHONPATH
    
    @classmethod
    def verify_mgltools(cls) -> Tuple[bool, str]:
        if not os.path.exists(cls.MGLTOOLS_PATH):
            return False, f"MGLTools not found at {cls.MGLTOOLS_PATH}"
        if not os.path.exists(cls.PREPARE_RECEPTOR):
            return False, f"prepare_receptor4.py not found"
        return True, "MGLTools OK"
    
    @classmethod
    def verify_openbabel(cls) -> bool:
        try:
            result = subprocess.run(['obabel', '-V'], 
                                  capture_output=True, 
                                  timeout=5)
            return result.returncode == 0
        except:
            return False

class PDBQTValidator:
    
    @staticmethod
    def validate_and_fix(pdbqt_file: str, verbose: bool = True, 
                        is_flexible_part: bool = False) -> Tuple[bool, str]:
        """Validate and fix PDBQT files."""
        if not os.path.exists(pdbqt_file):
            return False, "File not found"
        
        try:
            with open(pdbqt_file, 'r') as f:
                lines = f.readlines()
            
            has_torsdof = False
            has_atoms = False
            n_atoms = 0
            carbon_issues = []
            torsdof_value = 0
            
            for i, line in enumerate(lines):
                if line.startswith(('ATOM', 'HETATM')):
                    has_atoms = True
                    n_atoms += 1
                    
                    if len(line) >= 79:
                        atom_name = line[12:16].strip()
                        atom_type = line[77:79].strip()
                        
                        if atom_name.startswith('C') and atom_type == "C":
                            carbon_issues.append(i)
                
                elif line.startswith('TORSDOF'):
                    has_torsdof = True
                    try:
                        torsdof_value = int(line.split()[1])
                    except:
                        pass
            
            if not has_atoms:
                return False, "No ATOM records found"
            
            needs_fix = False
            messages = []
            
            if carbon_issues:
                for i in carbon_issues:
                    lines[i] = lines[i][:77] + " A" + (lines[i][79:] if len(lines[i]) > 79 else "\n")
                needs_fix = True
                messages.append(f"Fixed {len(carbon_issues)} carbon atoms (C->A)")
                if verbose:
                    print(f"  🔧 Fixed {len(carbon_issues)} carbon atoms")
            
            if not has_torsdof:
                if is_flexible_part:
                    torsdof = PDBQTValidator._count_rotatable_bonds(lines)
                    lines.append(f"TORSDOF {torsdof}\n")
                    torsdof_value = torsdof
                    messages.append(f"Added TORSDOF {torsdof} (flexible part)")
                    if verbose:
                        print(f"  ➕ Added TORSDOF {torsdof} (flexible residues)")
                
                elif n_atoms > 100:
                    lines.append("TORSDOF 0\n")
                    torsdof_value = 0
                    messages.append("Added TORSDOF 0 (rigid)")
                    if verbose:
                        print(f"  ➕ Added TORSDOF 0 (rigid receptor)")
                else:
                    torsdof = PDBQTValidator._count_rotatable_bonds(lines)
                    lines.append(f"TORSDOF {torsdof}\n")
                    torsdof_value = torsdof
                    messages.append(f"Added TORSDOF {torsdof} (ligand)")
                    if verbose:
                        print(f"  ➕ Added TORSDOF {torsdof} (ligand)")
                
                needs_fix = True

            if needs_fix:
                with open(pdbqt_file, 'w') as f:
                    f.writelines(lines)
            
            if messages:
                message = "; ".join(messages)
            else:
                message = f"OK ({n_atoms} atoms, TORSDOF={torsdof_value})"
            
            return True, message
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def _count_rotatable_bonds(lines: List[str]) -> int:
        branch_count = 0
        for line in lines:
            if line.startswith('BRANCH'):
                branch_count += 1
        return branch_count if branch_count > 0 else 0
    
    @staticmethod
    def quick_check(pdbqt_file: str) -> Dict[str, Union[bool, int]]:
        result = {
            'has_atoms': False,
            'has_torsdof': False,
            'n_atoms': 0,
            'torsdof_value': 0,
            'carbon_issues': 0,
            'is_valid': False
        }
        
        try:
            with open(pdbqt_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.startswith(('ATOM', 'HETATM')):
                    result['has_atoms'] = True
                    result['n_atoms'] += 1
                    
                    if len(line) >= 79:
                        atom_name = line[12:16].strip()
                        atom_type = line[77:79].strip()
                        if atom_name.startswith('C') and atom_type == "C":
                            result['carbon_issues'] += 1
                
                elif line.startswith('TORSDOF'):
                    result['has_torsdof'] = True
                    try:
                        result['torsdof_value'] = int(line.split()[1])
                    except:
                        pass
            
            result['is_valid'] = result['has_atoms'] and result['has_torsdof']
            
        except:
            pass
        
        return result

class ReceptorProcessor:
    
    def __init__(self, mode: str = 'rigid', flexible_residues: Optional[str] = None, 
                 verbose: bool = True):
        self.mode = mode
        self.flexible_residues = flexible_residues
        self.verbose = verbose
        
        if mode == 'flexible' and not flexible_residues:
            raise ValueError("flexible_residues required for flexible mode")
        
        Config.setup_environment()
    
    def process(self, pdb_file: str, output_dir: str) -> Tuple[Optional[List[str]], Optional[str]]:
        if not os.path.exists(pdb_file):
            return None, f"File not found: {pdb_file}"
        
        filename = Path(pdb_file).stem
        os.makedirs(output_dir, exist_ok=True)
        
        if self.verbose:
            print(f"\n🧬 Processing receptor: {filename}.pdb")
            if self.mode == 'flexible':
                print(f"   🔧 Mode: Flexible (residues: {self.flexible_residues})")
            else:
                print(f"   🔒 Mode: Rigid")
        
        chain_fixed = PDBTools.fix_missing_chain_id(pdb_file)
        if chain_fixed and self.verbose:
            print(f"   ✅ Fixed missing chain ID")
        
        base_pdbqt = os.path.join(output_dir, f"{filename}.pdbqt")
        
        success, error = self._run_prepare_receptor(pdb_file, base_pdbqt)
        if not success:
            return None, error
        
        if self.mode == 'flexible':
            output_files, warning = self._make_flexible(base_pdbqt, filename, output_dir)
            return output_files, warning
        
        else:
            success, msg = PDBQTValidator.validate_and_fix(base_pdbqt, 
                                                          verbose=self.verbose,
                                                          is_flexible_part=False)
            if not success:
                return None, f"Validation failed: {msg}"
            
            if self.verbose:
                if msg and msg != "OK":
                    print(f"   {msg}")
                print(f"   ✅ Generated: {filename}.pdbqt")
            
            return [base_pdbqt], None
    
    def _run_prepare_receptor(self, pdb_file: str, 
                             output_pdbqt: str) -> Tuple[bool, Optional[str]]:
        cmd = [
            Config.MGLTOOLS_PATH,
            Config.PREPARE_RECEPTOR,
            "-r", pdb_file,
            "-o", output_pdbqt,
            "-A", "hydrogens",
            "-U", "nphs_lps_waters"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=os.environ,
                timeout=Config.TIMEOUT_SECONDS
            )
            
            if result.returncode != 0:
                error = result.stderr.strip() or "Unknown MGLTools error"
                return False, f"prepare_receptor4 failed: {error[:200]}"
            
            if not os.path.exists(output_pdbqt):
                return False, "PDBQT file not generated"
            
            if os.path.getsize(output_pdbqt) < 100:
                return False, "PDBQT file too small (corrupt?)"
            
            return True, None
        
        except subprocess.TimeoutExpired:
            return False, f"Timeout (>{Config.TIMEOUT_SECONDS}s)"
        
        except Exception as e:
            return False, f"Exception: {str(e)}"
    
    def _make_flexible(self, base_pdbqt: str, filename: str, 
                      output_dir: str) -> Tuple[List[str], Optional[str]]:
        """
        FIXED: Use -g parameter instead of -o for prepare_flexreceptor4.py
        """
        rigid_pdbqt = os.path.join(output_dir, f"{filename}_rigid.pdbqt")
        flex_pdbqt = os.path.join(output_dir, f"{filename}_flex.pdbqt")
        
        os.rename(base_pdbqt, rigid_pdbqt)
        
        # CRITICAL FIX: Use -g instead of -o
        cmd = [
            Config.MGLTOOLS_PATH,
            Config.PREPARE_FLEXRECEPTOR,
            "-r", rigid_pdbqt,
            "-s", self.flexible_residues,
            "-g", rigid_pdbqt,  # FIXED: -g not -o!
            "-x", flex_pdbqt
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=os.environ,
                timeout=Config.TIMEOUT_SECONDS
            )
            
            if result.returncode == 0 and os.path.exists(flex_pdbqt):
                PDBQTValidator.validate_and_fix(rigid_pdbqt, 
                                               verbose=False, 
                                               is_flexible_part=False)
                
                PDBQTValidator.validate_and_fix(flex_pdbqt, 
                                               verbose=False, 
                                               is_flexible_part=True)
                
                if self.verbose:
                    rigid_info = PDBQTValidator.quick_check(rigid_pdbqt)
                    flex_info = PDBQTValidator.quick_check(flex_pdbqt)
                    
                    print(f"   ✅ Generated: {filename}_rigid.pdbqt "
                          f"(TORSDOF={rigid_info['torsdof_value']})")
                    print(f"   ✅ Generated: {filename}_flex.pdbqt "
                          f"(TORSDOF={flex_info['torsdof_value']})")
                
                return [rigid_pdbqt, flex_pdbqt], None
            
            else:
                error = result.stderr.strip()[:200] if result.stderr else "Unknown error"
                warning = f"Flexible generation failed: {error}"
                
                if self.verbose:
                    print(f"   ⚠️  {warning}")
                    print(f"   ✅ Kept rigid receptor: {filename}_rigid.pdbqt")
                
                return [rigid_pdbqt], warning
        
        except subprocess.TimeoutExpired:
            warning = "Flexible generation timeout - kept rigid only"
            if self.verbose:
                print(f"   ⚠️  {warning}")
            return [rigid_pdbqt], warning
        
        except Exception as e:
            warning = f"Flexible error: {str(e)}"
            if self.verbose:
                print(f"   ⚠️  {warning}")
            return [rigid_pdbqt], warning

class LigandProcessor:
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        Config.setup_environment()
    
    def process(self, ligand_file: str, output_dir: str) -> Tuple[Optional[str], Optional[str]]:
        if not os.path.exists(ligand_file):
            return None, f"File not found: {ligand_file}"
        
        filename = Path(ligand_file).stem
        file_ext = Path(ligand_file).suffix.lower()
        os.makedirs(output_dir, exist_ok=True)
        
        if self.verbose:
            print(f"\n💊 Processing ligand: {filename}{file_ext}")
        
        if file_ext != '.pdb':
            pdb_file, error = self._convert_to_pdb(ligand_file, output_dir, filename)
            if not pdb_file:
                return None, error
            work_file = pdb_file
            if self.verbose:
                print(f"   🔄 Converted {file_ext.upper()} -> PDB")
        else:
            work_file = ligand_file
            if self.verbose:
                print(f"   ✅ PDB format (no conversion needed)")
        
        output_pdbqt = os.path.join(output_dir, f"{filename}.pdbqt")
        
        success, error = self._run_prepare_ligand(work_file, output_pdbqt)
        if not success:
            return None, error
        
        success, msg = PDBQTValidator.validate_and_fix(output_pdbqt, 
                                                       verbose=self.verbose,
                                                       is_flexible_part=False)
        if not success:
            return None, f"Validation failed: {msg}"
        
        if self.verbose:
            if msg != "OK":
                print(f"   ✅ {msg}")
            print(f"   ✅ Generated: {filename}.pdbqt")
        
        return output_pdbqt, None
    
    def _convert_to_pdb(self, input_file: str, output_dir: str, 
                       filename: str) -> Tuple[Optional[str], Optional[str]]:
        pdb_file = os.path.join(output_dir, f"{filename}_converted.pdb")
        
        cmd = ['obabel', input_file, '-O', pdb_file, '--gen3d']
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                error = result.stderr.strip() or "OpenBabel failed"
                return None, f"Conversion failed: {error[:200]}"
            
            if not os.path.exists(pdb_file):
                return None, "PDB file not generated"
            
            if os.path.getsize(pdb_file) < 50:
                return None, "PDB file too small (corrupt?)"
            
            self._fix_pdb_format(pdb_file)
            
            return pdb_file, None
        
        except subprocess.TimeoutExpired:
            return None, "OpenBabel timeout (>60s)"
        
        except FileNotFoundError:
            return None, "OpenBabel not installed (run: apt-get install openbabel)"
        
        except Exception as e:
            return None, f"Conversion error: {str(e)}"
    
    def _fix_pdb_format(self, pdb_file: str):
        try:
            with open(pdb_file, 'r') as f:
                lines = f.readlines()
            
            fixed_lines = []
            atom_count = 0
            has_ter = False
            
            for line in lines:
                if line.startswith('HETATM'):
                    line = 'ATOM  ' + line[6:]
                    atom_count += 1
                elif line.startswith('ATOM'):
                    atom_count += 1
                
                if line.startswith(('CONECT', 'MASTER', 'SSBOND', 'LINK', 'CISPEP')):
                    continue
                
                if line.startswith('END') and not has_ter:
                    fixed_lines.append(f"TER   {atom_count+1:5d}      UNL A   1\n")
                    has_ter = True
                
                fixed_lines.append(line)
            
            with open(pdb_file, 'w') as f:
                f.writelines(fixed_lines)
        
        except:
            pass
    
    def _run_prepare_ligand(self, ligand_file: str, 
                           output_pdbqt: str) -> Tuple[bool, Optional[str]]:
        cmd = [
            Config.MGLTOOLS_PATH,
            Config.PREPARE_LIGAND,
            "-l", ligand_file,
            "-o", output_pdbqt,
            "-A", "hydrogens",
            "-U", "nphs"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=os.environ,
                timeout=Config.TIMEOUT_SECONDS
            )
            
            if result.returncode != 0:
                error = result.stderr.strip() or "Unknown MGLTools error"
                return False, f"prepare_ligand4 failed: {error[:200]}"
            
            if not os.path.exists(output_pdbqt):
                return False, "PDBQT file not generated"
            
            if os.path.getsize(output_pdbqt) < 50:
                return False, "PDBQT file too small"
            
            return True, None
        
        except subprocess.TimeoutExpired:
            return False, f"Timeout (>{Config.TIMEOUT_SECONDS}s)"
        
        except Exception as e:
            return False, f"Exception: {str(e)}"

class BatchProcessor:
    
    def __init__(self, n_workers: int = Config.DEFAULT_WORKERS, verbose: bool = True):
        self.n_workers = n_workers
        self.verbose = verbose
        self.lock = threading.Lock()
        self.completed = 0
        self.total = 0
        self.failed = 0
        self.start_time = 0
    
    def process_receptors(self,
                         receptor_files: List[str],
                         output_dir: str,
                         mode: str = 'rigid',
                         flexible_residues: Optional[str] = None) -> Dict:
        
        self._reset_counters()
        self.total = len(receptor_files)
        self.start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🚀 Batch Receptor Processing")
            print(f"{'='*60}")
            print(f"Files: {self.total}")
            print(f"Mode: {mode.upper()}")
            if mode == 'flexible':
                print(f"Flexible residues: {flexible_residues}")
            print(f"Workers: {self.n_workers}")
            print(f"{'='*60}\n")
        
        processor = ReceptorProcessor(mode=mode, 
                                     flexible_residues=flexible_residues,
                                     verbose=False)
        
        successful = []
        failed = []
        flexible_count = 0
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_file = {
                executor.submit(processor.process, file, output_dir): file
                for file in receptor_files
            }
            
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                
                with self.lock:
                    self.completed += 1
                
                try:
                    output_files, error = future.result()
                    
                    if output_files:
                        successful.extend(output_files)
                        
                        if len(output_files) == 2:
                            flexible_count += 1
                        
                        if self.verbose:
                            if len(output_files) == 2:
                                print(f"✅ {Path(filename).name} → "
                                     f"{Path(output_files[0]).name}, {Path(output_files[1]).name}")
                            else:
                                print(f"✅ {Path(filename).name} → {Path(output_files[0]).name}")
                    else:
                        failed.append((filename, error))
                        with self.lock:
                            self.failed += 1
                        
                        if self.verbose:
                            print(f"❌ {Path(filename).name}: {error}")
                
                except Exception as e:
                    failed.append((filename, f"Exception: {str(e)}"))
                    with self.lock:
                        self.failed += 1
                    
                    if self.verbose:
                        print(f"❌ {Path(filename).name}: {str(e)}")
                
                if self.verbose:
                    self._print_progress()
        
        elapsed = time.time() - self.start_time
        
        if self.verbose:
            self._print_summary("Receptor", len(successful), len(failed), flexible_count, elapsed)
        
        return {
            'successful': successful,
            'failed': failed,
            'stats': {
                'completed': self.completed,
                'failed': self.failed,
                'flexible_count': flexible_count,
                'elapsed_time': elapsed
            }
        }
    
    def process_ligands(self,
                       ligand_files: List[str],
                       output_dir: str) -> Dict:
        
        self._reset_counters()
        self.total = len(ligand_files)
        self.start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🚀 Batch Ligand Processing")
            print(f"{'='*60}")
            print(f"Files: {self.total}")
            print(f"Workers: {self.n_workers}")
            print(f"{'='*60}\n")
        
        processor = LigandProcessor(verbose=False)
        
        successful = []
        failed = []
        converted = 0
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_file = {
                executor.submit(processor.process, file, output_dir): file
                for file in ligand_files
            }
            
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                
                with self.lock:
                    self.completed += 1
                
                try:
                    output_file, error = future.result()
                    
                    if output_file:
                        successful.append(output_file)
                        
                        if Path(filename).suffix.lower() != '.pdb':
                            converted += 1
                        
                        if self.verbose:
                            print(f"✅ {Path(filename).name}")
                    else:
                        failed.append((filename, error))
                        with self.lock:
                            self.failed += 1
                        
                        if self.verbose:
                            print(f"❌ {Path(filename).name}: {error}")
                
                except Exception as e:
                    failed.append((filename, f"Exception: {str(e)}"))
                    with self.lock:
                        self.failed += 1
                    
                    if self.verbose:
                        print(f"❌ {Path(filename).name}: {str(e)}")
                
                if self.verbose:
                    self._print_progress()
        
        elapsed = time.time() - self.start_time
        
        if self.verbose:
            self._print_summary("Ligand", len(successful), len(failed), converted, elapsed)
        
        return {
            'successful': successful,
            'failed': failed,
            'stats': {
                'completed': self.completed,
                'failed': self.failed,
                'converted': converted,
                'elapsed_time': elapsed
            }
        }
    
    def _reset_counters(self):
        self.completed = 0
        self.total = 0
        self.failed = 0
        self.start_time = 0
    
    def _print_progress(self):
        if self.total == 0:
            return
        
        pct = (self.completed / self.total) * 100
        elapsed = time.time() - self.start_time
        
        print(f"📊 Progress: {self.completed}/{self.total} ({pct:.0f}%) | "
              f"✅ {self.completed - self.failed} | ❌ {self.failed} | "
              f"⏱️ {elapsed:.1f}s")
    
    def _print_summary(self, file_type: str, successful: int, failed: int, 
                      extra_count: int, elapsed: float):
        print(f"\n{'='*60}")
        print(f"📋 {file_type} Processing Summary")
        print(f"{'='*60}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        
        if file_type == "Receptor" and extra_count > 0:
            print(f"🔧 Flexible receptors: {extra_count}")
        elif file_type == "Ligand" and extra_count > 0:
            print(f"🔄 Format conversions: {extra_count}")
        
        print(f"⏱️  Total time: {elapsed:.1f}s")
        
        if successful > 0:
            avg_time = elapsed / successful
            print(f"⚡ Average: {avg_time:.1f}s per file")
        
        print(f"{'='*60}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ZymEvo PDBQT Preprocessing - FIXED VERSION (use -g not -o)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rigid receptors
  python autopre_dock.py --receptor protein1.pdb protein2.pdb --mode rigid
  
  # Flexible receptors (use RESNAME+NUMBER format)
  python autopre_dock.py --receptor protein.pdb --mode flexible --flex-res "ASP231_HIS235_GLU261"
  
  # Ligands
  python autopre_dock.py --ligand ligand1.sdf ligand2.mol2
        """
    )
    
    parser.add_argument('--receptor', nargs='+', help='Receptor PDB files')
    parser.add_argument('--ligand', nargs='+', help='Ligand files (PDB/SDF/MOL/MOL2)')
    parser.add_argument('--mode', choices=['rigid', 'flexible'], default='rigid',
                       help='Receptor processing mode (default: rigid)')
    parser.add_argument('--flex-res', help='Flexible residues (e.g., ASP231_HIS235_GLU261)')
    parser.add_argument('--output', default='processed', help='Output directory')
    parser.add_argument('--workers', type=int, default=Config.DEFAULT_WORKERS,
                       help=f'Number of parallel workers (default: {Config.DEFAULT_WORKERS})')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    parser.add_argument('--verify', action='store_true', help='Verify installation only')
    
    args = parser.parse_args()
    
    if args.verify:
        print("Verifying installation...")
        mgl_ok, mgl_msg = Config.verify_mgltools()
        print(f"MGLTools: {mgl_msg}")
        
        obabel_ok = Config.verify_openbabel()
        print(f"OpenBabel: {'OK' if obabel_ok else 'Not installed (optional for ligand conversion)'}")
        
        sys.exit(0 if mgl_ok else 1)
    
    if not args.receptor and not args.ligand:
        parser.print_help()
        print("\n❌ Error: Provide --receptor and/or --ligand files")
        sys.exit(1)
    
    if args.mode == 'flexible' and not args.flex_res:
        print("❌ Error: --flex-res required for flexible mode")
        sys.exit(1)
    
    processor = BatchProcessor(n_workers=args.workers, verbose=not args.quiet)
    
    if args.receptor:
        results = processor.process_receptors(
            args.receptor,
            args.output,
            mode=args.mode,
            flexible_residues=args.flex_res
        )
        
        if results['stats']['failed'] > 0 and not args.quiet:
            print("\n⚠️  Failed files:")
            for filename, error in results['failed'][:5]:
                print(f"   • {Path(filename).name}: {error}")
            if len(results['failed']) > 5:
                print(f"   ... and {len(results['failed']) - 5} more")
    
    if args.ligand:
        results = processor.process_ligands(args.ligand, args.output)
        
        if results['stats']['failed'] > 0 and not args.quiet:
            print("\n⚠️  Failed files:")
            for filename, error in results['failed'][:5]:
                print(f"   • {Path(filename).name}: {error}")
            if len(results['failed']) > 5:
                print(f"   ... and {len(results['failed']) - 5} more")


if __name__ == "__main__":
    main()
