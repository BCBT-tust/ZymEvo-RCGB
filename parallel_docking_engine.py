#!/usr/bin/env python3
"""
ZymEvo Parallel Docking Engine
High-throughput molecular docking with AutoDock Vina
Version: 2.0
"""

import os
import sys
import glob
import shutil
import time
import subprocess
import zipfile
import multiprocessing
import psutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed


class VinaInstaller:
    
    VINA_URL = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
    VINA_BINARY = "vina"
    
    @staticmethod
    def ensure_vina(work_dir: str = ".") -> Optional[str]:
        """Ensure Vina is available"""
        # Check if already in PATH
        vina_path = shutil.which("vina")
        if vina_path:
            print(f"  ✅ Vina found: {vina_path}")
            return vina_path

        local_vina = os.path.join(work_dir, VinaInstaller.VINA_BINARY)
        if os.path.exists(local_vina) and os.access(local_vina, os.X_OK):
            print(f"  ✅ Vina found: {local_vina}")
            return local_vina
        
        print("  📥 Downloading AutoDock Vina v1.2.5...")
        try:
            download_path = os.path.join(work_dir, "vina_download")
            subprocess.run([
                "wget", "-q", VinaInstaller.VINA_URL, "-O", download_path
            ], check=True, timeout=60)
            
            os.chmod(download_path, 0o755)
            shutil.move(download_path, local_vina)
            
            print(f"  ✅ Vina installed: {local_vina}")
            return local_vina
            
        except Exception as e:
            print(f"  ❌ Failed to download Vina: {e}")
            return None


class SystemResources:
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        try:
            memory_info = psutil.virtual_memory()
            self.memory_gb = memory_info.total / (1024**3)
        except:
            self.memory_gb = 12.0
    
    def calculate_parallel_config(self, total_tasks: int) -> Tuple[int, int]:
        """
        Calculate optimal parallel configuration
        
        Returns:
            (max_workers, cores_per_job)
        """
        # Conservative: max 4 parallel jobs, leave some CPU for system
        max_parallel = min(4, self.cpu_count // 2, total_tasks)
        cores_per_job = max(1, self.cpu_count // max_parallel)
        
        return max_parallel, cores_per_job
    
    def print_info(self):
        """Print system information"""
        print(f"💻 System Resources:")
        print(f"   • CPU cores: {self.cpu_count}")
        print(f"   • Memory: {self.memory_gb:.1f} GB")


class ParameterReader:
    """Docking parameter file reader"""
    
    REQUIRED_PARAMS = ['center_x', 'center_y', 'center_z', 'size_x', 'size_y', 'size_z']
    
    @staticmethod
    def read_parameters(param_file: str) -> Optional[Dict[str, float]]:
        """Read docking parameters from config file"""
        if not os.path.exists(param_file):
            return None
        
        params = {}
        
        try:
            with open(param_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        
                        if key in ParameterReader.REQUIRED_PARAMS:
                            params[key] = float(value.strip())
            
            missing = [p for p in ParameterReader.REQUIRED_PARAMS if p not in params]
            if missing:
                print(f"  ⚠️  Missing parameters in {param_file}: {missing}")
                return None
            
            return params
            
        except Exception as e:
            print(f"  ❌ Error reading {param_file}: {e}")
            return None
    
    @staticmethod
    def find_matching_parameter(receptor_name: str, param_files: List[str]) -> Optional[str]:
        receptor_base = Path(receptor_name).stem.lower()
        
        for param_file in param_files:
            param_base = Path(param_file).stem.lower()
            
            receptor_clean = receptor_base.replace('_receptor', '').replace('_protein', '')
            param_clean = param_base.replace('_docking_params', '').replace('_params', '')
            
            if receptor_clean == param_clean or receptor_clean in param_clean or param_clean in receptor_clean:
                return param_file

        return param_files[0] if param_files else None


class VinaDocking:
    
    @staticmethod
    def run_docking(receptor: str, ligand: str, params: Dict[str, float], 
                   vina_path: str, cores: int = 1, 
                   exhaustiveness: int = 8, num_modes: int = 10,
                   timeout: int = 300) -> Dict:
        try:
            # Generate output paths
            receptor_name = Path(receptor).stem
            ligand_name = Path(ligand).stem
            output_name = f"{receptor_name}_{ligand_name}"
            
            # Ensure results directory exists
            os.makedirs("results", exist_ok=True)
            
            output_file = f"results/{output_name}.pdbqt"
            log_file = f"results/{output_name}.log"
            
            # Build Vina command
            cmd = [
                vina_path,
                "--receptor", receptor,
                "--ligand", ligand,
                "--center_x", str(params['center_x']),
                "--center_y", str(params['center_y']),
                "--center_z", str(params['center_z']),
                "--size_x", str(params['size_x']),
                "--size_y", str(params['size_y']),
                "--size_z", str(params['size_z']),
                "--out", output_file,
                "--cpu", str(cores),
                "--exhaustiveness", str(exhaustiveness),
                "--num_modes", str(num_modes)
            ]
            
            # Execute docking
            start_time = time.time()
            
            with open(log_file, 'w') as log:
                result = subprocess.run(
                    cmd, 
                    stdout=log, 
                    stderr=subprocess.STDOUT,
                    timeout=timeout
                )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0 and os.path.exists(output_file):
                # Extract best score from log
                best_score = VinaDocking.extract_best_score(log_file)
                
                return {
                    'status': 'success',
                    'output': output_file,
                    'log': log_file,
                    'name': output_name,
                    'receptor': receptor_name,
                    'ligand': ligand_name,
                    'best_score': best_score,
                    'elapsed': elapsed
                }
            else:
                return {
                    'status': 'failed',
                    'name': output_name,
                    'error': f"Exit code: {result.returncode}",
                    'elapsed': elapsed
                }
        
        except subprocess.TimeoutExpired:
            return {
                'status': 'failed',
                'name': output_name,
                'error': f"Timeout (>{timeout}s)",
                'elapsed': timeout
            }
        except Exception as e:
            return {
                'status': 'failed',
                'name': output_name,
                'error': str(e),
                'elapsed': 0
            }
    
    @staticmethod
    def extract_best_score(log_file: str) -> Optional[float]:
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Find result table
            for i, line in enumerate(lines):
                if 'mode |' in line and 'affinity' in line:
                    # Next line should be first result
                    if i + 2 < len(lines):
                        result_line = lines[i + 2]
                        parts = result_line.split()
                        if len(parts) >= 2:
                            return float(parts[1])
            return None
        except:
            return None


class ParallelDockingEngine:
    
    def __init__(self, work_dir: str = "."):
        self.work_dir = work_dir
        self.start_time = datetime.now()
        
        self.resources = SystemResources()
        
        # Ensure Vina is available
        self.vina_path = VinaInstaller.ensure_vina(work_dir)
        if not self.vina_path:
            raise RuntimeError("Failed to setup AutoDock Vina")
    
    def prepare_tasks(self, receptor_dir: str, ligand_dir: str, 
                     param_dir: str) -> List[Tuple]:

        receptors = glob.glob(os.path.join(receptor_dir, "*.pdbqt"))
        ligands = glob.glob(os.path.join(ligand_dir, "*.pdbqt"))
        parameters = glob.glob(os.path.join(param_dir, "*.txt"))
        
        if not receptors:
            raise ValueError(f"No receptor files found in {receptor_dir}")
        if not ligands:
            raise ValueError(f"No ligand files found in {ligand_dir}")
        if not parameters:
            raise ValueError(f"No parameter files found in {param_dir}")
        
        print(f"\n📊 Files loaded:")
        print(f"   • Receptors: {len(receptors)}")
        print(f"   • Ligands: {len(ligands)}")
        print(f"   • Parameters: {len(parameters)}")
        
        # Build task list
        tasks = []
        skipped = 0
        
        for receptor in receptors:
            # Find matching parameters
            param_file = ParameterReader.find_matching_parameter(receptor, parameters)
            
            if not param_file:
                print(f"  ⚠️  No parameters for {Path(receptor).name}")
                skipped += 1
                continue
            
            params = ParameterReader.read_parameters(param_file)
            if not params:
                skipped += 1
                continue
            
            # Add task for each ligand
            for ligand in ligands:
                tasks.append((receptor, ligand, params))
        
        if skipped > 0:
            print(f"  ⚠️  Skipped {skipped} receptors (missing/invalid parameters)")
        
        return tasks
    
    def run_parallel(self, tasks: List[Tuple], 
                    exhaustiveness: int = 8,
                    num_modes: int = 10,
                    timeout: int = 300) -> Tuple[List[Dict], List[Dict]]:

        total_tasks = len(tasks)
        
        if total_tasks == 0:
            raise ValueError("No valid docking tasks")
        
        print(f"\n🚀 Starting {total_tasks} docking tasks...")
        
        # Calculate parallel config
        max_workers, cores_per_job = self.resources.calculate_parallel_config(total_tasks)
        
        print(f"\n⚙️  Parallel configuration:")
        print(f"   • Parallel jobs: {max_workers}")
        print(f"   • CPU cores/job: {cores_per_job}")
        print(f"   • Total CPU usage: {max_workers * cores_per_job}/{self.resources.cpu_count}")
        
        # Prepare arguments
        args_list = [
            (r, l, p, self.vina_path, cores_per_job, exhaustiveness, num_modes, timeout)
            for r, l, p in tasks
        ]
        
        successful = []
        failed = []
        completed = 0
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_args = {
                executor.submit(self._run_single, args): args
                for args in args_list
            }
            
            # Collect results with progress tracking
            for future in as_completed(future_to_args):
                try:
                    result = future.result(timeout=10)
                    completed += 1
                    
                    if result['status'] == 'success':
                        successful.append(result)
                    else:
                        failed.append(result)
                    
                    # Progress display
                    self._print_progress(completed, total_tasks, len(successful), 
                                        len(failed), start_time)
                    
                except Exception as e:
                    completed += 1
                    failed.append({
                        'name': 'unknown',
                        'status': 'failed',
                        'error': str(e)
                    })
                    self._print_progress(completed, total_tasks, len(successful), 
                                        len(failed), start_time)
        
        print()  # New line after progress
        
        # Print summary
        self._print_summary(successful, failed, time.time() - start_time)
        
        return successful, failed
    
    @staticmethod
    def _run_single(args):
        """Wrapper for single docking execution"""
        return VinaDocking.run_docking(*args)
    
    @staticmethod
    def _print_progress(completed: int, total: int, success: int, failed: int, start_time: float):
        """Print progress bar"""
        progress = (completed / total) * 100
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total - completed) / rate if rate > 0 else 0
        
        print(f"\r🔄 Progress: {completed}/{total} ({progress:.1f}%) | "
              f"✅ {success} | ❌ {failed} | "
              f"⏱️ {elapsed:.0f}s | ETA: {eta:.0f}s",
              end='', flush=True)
    
    @staticmethod
    def _print_summary(successful: List[Dict], failed: List[Dict], total_time: float):
        """Print results summary"""
        total = len(successful) + len(failed)
        success_rate = (len(successful) / total * 100) if total > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"✅ Docking Completed!")
        print(f"{'='*70}")
        print(f"   • Total tasks: {total}")
        print(f"   • Successful: {len(successful)} ({success_rate:.1f}%)")
        print(f"   • Failed: {len(failed)}")
        print(f"   • Total time: {total_time:.1f}s")
        
        if successful:
            avg_time = sum(r['elapsed'] for r in successful) / len(successful)
            print(f"   • Avg time/task: {avg_time:.1f}s")
            
            # Show top 3 best scores
            scored = [r for r in successful if r.get('best_score')]
            if scored:
                scored.sort(key=lambda x: x['best_score'])
                print(f"\n🏆 Top 3 Binding Affinities:")
                for i, r in enumerate(scored[:3], 1):
                    print(f"   {i}. {r['ligand']} → {r['receptor']}: "
                          f"{r['best_score']:.1f} kcal/mol")
        
        if failed:
            print(f"\n⚠️  Failed tasks (showing first 3):")
            for i, r in enumerate(failed[:3], 1):
                print(f"   {i}. {r['name']}: {r.get('error', 'Unknown error')}")
            if len(failed) > 3:
                print(f"   ... and {len(failed) - 3} more")


class ResultsPackager:
    """Results packaging and summary generation"""
    
    @staticmethod
    def create_package(successful: List[Dict], failed: List[Dict], 
                      output_name: Optional[str] = None) -> Optional[str]:
        """Create ZIP package with all results"""
        if not successful:
            print("❌ No successful results to package")
            return None
        
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"docking_results_{timestamp}.zip"
        
        print(f"\n📦 Creating results package...")
        
        try:
            with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add result files
                for result in successful:
                    if os.path.exists(result['output']):
                        zipf.write(result['output'], f"results/{Path(result['output']).name}")
                    
                    if os.path.exists(result['log']):
                        zipf.write(result['log'], f"logs/{Path(result['log']).name}")
                
                # Create summary
                summary = ResultsPackager.generate_summary(successful, failed)
                zipf.writestr('SUMMARY.txt', summary)
                
                # Create CSV results
                csv_content = ResultsPackager.generate_csv(successful)
                zipf.writestr('results_summary.csv', csv_content)
            
            file_size = os.path.getsize(output_name) / (1024 * 1024)
            print(f"✅ Package created: {output_name} ({file_size:.2f} MB)")
            
            return output_name
            
        except Exception as e:
            print(f"❌ Failed to create package: {e}")
            return None
    
    @staticmethod
    def generate_summary(successful: List[Dict], failed: List[Dict]) -> str:
        """Generate text summary"""
        total = len(successful) + len(failed)
        success_rate = (len(successful) / total * 100) if total > 0 else 0
        
        summary = f"""ZymEvo Parallel Docking Results
{'='*70}

Run Information:
  Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Total Tasks: {total}
  Successful: {len(successful)} ({success_rate:.1f}%)
  Failed: {len(failed)}

{'='*70}

Successful Docking Results:
{'-'*70}
"""
        
        # Sort by score if available
        scored = [r for r in successful if r.get('best_score')]
        unscored = [r for r in successful if not r.get('best_score')]
        
        scored.sort(key=lambda x: x['best_score'])
        
        for i, result in enumerate(scored + unscored, 1):
            score_str = f"{result['best_score']:.2f} kcal/mol" if result.get('best_score') else "N/A"
            summary += f"{i:3d}. {result['ligand']:20s} → {result['receptor']:20s} | {score_str}\n"
        
        if failed:
            summary += f"\n{'='*70}\n"
            summary += f"Failed Tasks:\n"
            summary += f"{'-'*70}\n"
            for i, result in enumerate(failed, 1):
                summary += f"{i:3d}. {result['name']}: {result.get('error', 'Unknown')}\n"
        
        summary += f"\n{'='*70}\n"
        summary += """
Usage Notes:
  • Each .pdbqt file contains multiple docked poses
  • The first pose typically has the best (lowest) binding energy
  • Log files contain detailed scoring and RMSD information
  • Binding energies are in kcal/mol (more negative = stronger binding)
"""
        
        return summary
    
    @staticmethod
    def generate_csv(successful: List[Dict]) -> str:
        """Generate CSV results table"""
        lines = ["Receptor,Ligand,Best_Score_kcal_mol,Output_File,Time_seconds\n"]
        
        for result in successful:
            receptor = result.get('receptor', 'unknown')
            ligand = result.get('ligand', 'unknown')
            score = result.get('best_score', '')
            score_str = f"{score:.2f}" if score else "N/A"
            output = Path(result['output']).name
            elapsed = f"{result.get('elapsed', 0):.1f}"
            
            lines.append(f"{receptor},{ligand},{score_str},{output},{elapsed}\n")
        
        return ''.join(lines)


def main():
    """Main entry point for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ZymEvo Parallel Docking Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python parallel_docking_engine.py \\
    --receptors ./receptors \\
    --ligands ./ligands \\
    --parameters ./parameters \\
    --output results.zip
        """
    )
    
    parser.add_argument('--receptors', required=True, help='Directory with receptor .pdbqt files')
    parser.add_argument('--ligands', required=True, help='Directory with ligand .pdbqt files')
    parser.add_argument('--parameters', required=True, help='Directory with parameter .txt files')
    parser.add_argument('--output', default='docking_results.zip', help='Output ZIP file')
    parser.add_argument('--exhaustiveness', type=int, default=8, help='Vina exhaustiveness')
    parser.add_argument('--num_modes', type=int, default=10, help='Number of binding modes')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout per task (seconds)')
    
    args = parser.parse_args()
    
    try:
        # Initialize engine
        print("="*70)
        print("🚀 ZymEvo Parallel Docking Engine")
        print("="*70)
        
        engine = ParallelDockingEngine()
        engine.resources.print_info()
        
        # Prepare tasks
        tasks = engine.prepare_tasks(args.receptors, args.ligands, args.parameters)
        
        # Run docking
        successful, failed = engine.run_parallel(
            tasks,
            exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes,
            timeout=args.timeout
        )
        
        # Package results
        package = ResultsPackager.create_package(successful, failed, args.output)
        
        if package:
            print(f"\n🎉 Pipeline completed!")
            print(f"📥 Results: {package}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
