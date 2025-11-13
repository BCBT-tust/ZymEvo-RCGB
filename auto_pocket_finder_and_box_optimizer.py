#!/usr/bin/env python3

import os
import sys
import platform
import subprocess
import urllib.request
import shutil
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import cdist
from scipy.stats import norm
import warnings
import tempfile

warnings.filterwarnings('ignore')


@dataclass
class AtomData:
    """Protein structure data"""
    coordinates: np.ndarray
    elements: List[str]
    atom_names: List[str]
    residues: List[str]
    residue_ids: List[int]


@dataclass
class DockingParameters:
    """Optimized docking box parameters"""
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float
    n_atoms: int
    box_volume: float
    pocket_score: float
    docking_score: float
    iterations: int
    optimized_by: str

class PDBQTValidator:
    """PDBQT file validator and auto-fixer"""
    
    @staticmethod
    def validate_and_fix_pdbqt(pdbqt_file: str, verbose: bool = True) -> Tuple[bool, str, Optional[str]]:
        """
        Validate PDBQT file and auto-fix if needed
        
        Returns:
            Tuple[is_valid, message, fixed_file_path]
            - is_valid: Whether file is OK for Vina
            - message: Diagnostic message
            - fixed_file_path: Path to fixed file (or original if no fix needed)
        """
        
        if not os.path.exists(pdbqt_file):
            return False, "File not found", None
        
        # Read file
        try:
            with open(pdbqt_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            return False, f"Cannot read file: {e}", None
        
        has_atoms = False
        has_torsdof = False
        torsdof_value = 0
        n_atoms = 0
        needs_fix = False
        carbon_issues = []
        
        for i, line in enumerate(lines):
            if line.startswith('ATOM') or line.startswith('HETATM'):
                has_atoms = True
                n_atoms += 1
                
                # Check carbon atom type issue
                if len(line) >= 79:
                    atom_name = line[12:16].strip()
                    atom_type = line[77:79].strip()
                    
                    # Carbon atom with wrong type "C" instead of "A"
                    if atom_name.startswith('C') and atom_type == "C":
                        needs_fix = True
                        carbon_issues.append(i)
            
            elif line.startswith('TORSDOF'):
                has_torsdof = True
                try:
                    torsdof_value = int(line.split()[1])
                except:
                    pass
        
        if not has_atoms:
            return False, "No ATOM records found", None
        
        if not has_torsdof:
            return False, "Missing TORSDOF record", None
        
        if torsdof_value == 0:
            msg = f"⚠️  TORSDOF=0 (rigid molecule) - may cause Vina issues"
            if verbose:
                print(f"  {msg}")
        
        if needs_fix:
            if verbose:
                print(f"  🔧 Detected {len(carbon_issues)} carbon atoms with wrong type")
                print(f"  🔧 Auto-fixing...")
            
            fixed_file = pdbqt_file.replace('.pdbqt', '_fixed.pdbqt')
            
            # Apply fixes
            fixed_lines = []
            for i, line in enumerate(lines):
                if i in carbon_issues:
                    # Replace "C " with " A"
                    fixed_line = line[:77] + " A" + (line[79:] if len(line) > 79 else "\n")
                    fixed_lines.append(fixed_line)
                else:
                    fixed_lines.append(line)
            
            try:
                with open(fixed_file, 'w') as f:
                    f.writelines(fixed_lines)
                
                if verbose:
                    print(f"  ✅ Fixed file: {fixed_file}")
                
                return True, f"Auto-fixed {len(carbon_issues)} carbon atoms", fixed_file
            
            except Exception as e:
                return False, f"Fix failed: {e}", None
        
        else:

            return True, f"File OK ({n_atoms} atoms, TORSDOF={torsdof_value})", pdbqt_file


class VinaInstaller:
    
    VINA_URLS = {
        'linux': 'https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64',
        'darwin': 'https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_mac_catalina_64bit',
        'windows': 'https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_windows_x86_64.exe'
    }
    
    @staticmethod
    def get_vina_path() -> Optional[str]:
        """Check if Vina is already available"""
        vina_cmd = 'vina' if platform.system() != 'Windows' else 'vina.exe'
        result = shutil.which(vina_cmd)
        if result:
            return result
        
        local_vina = Path('./vina')
        if local_vina.exists() and os.access(local_vina, os.X_OK):
            return str(local_vina)
        
        return None
    
    @staticmethod
    def download_vina(output_dir: str = '.') -> Optional[str]:
        system = platform.system().lower()
        
        if system not in VinaInstaller.VINA_URLS:
            print(f"❌ Unsupported OS: {system}")
            return None
        
        url = VinaInstaller.VINA_URLS[system]
        vina_name = 'vina' if system != 'windows' else 'vina.exe'
        vina_path = os.path.join(output_dir, vina_name)
        
        print(f"📥 Downloading Vina v1.2.5...")
        
        try:
            urllib.request.urlretrieve(url, vina_path)
            os.chmod(vina_path, 0o755)
            
            if os.path.exists(vina_path) and os.path.getsize(vina_path) > 1000000:
                print(f"✅ Vina ready: {vina_path}")
                return vina_path
            else:
                print(f"❌ Download incomplete")
                return None
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    
    @staticmethod
    def ensure_vina(force_download: bool = False) -> Optional[str]:
        """Ensure Vina availability"""
        if not force_download:
            vina_path = VinaInstaller.get_vina_path()
            if vina_path:
                print(f"✅ Vina found: {vina_path}")
                return vina_path
        
        print("⚙️  Vina not found, downloading...")
        return VinaInstaller.download_vina()


class PDBParser:
    
    ATOMIC_MASSES = {
        'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
        'S': 32.06, 'P': 30.974, 'F': 18.998, 'CL': 35.45,
        'BR': 79.904, 'I': 126.90, 'SE': 78.971, 'FE': 55.845,
        'ZN': 65.38, 'MG': 24.305, 'CA': 40.078, 'NA': 22.990,
        'K': 39.098, 'MN': 54.938, 'CU': 63.546, 'CO': 58.933
    }
    
    @staticmethod
    def parse_pdb(pdb_file: str) -> Optional[AtomData]:

        if not os.path.exists(pdb_file):
            return None
        
        coordinates, elements, atom_names, residues, residue_ids = [], [], [], [], []
        
        try:
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        try:
                            atom_name = line[12:16].strip()
                            residue = line[17:20].strip()
                            res_id = int(line[22:26].strip())
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                            element = line[76:78].strip().upper()
                            
                            if abs(x) > 9999 or abs(y) > 9999 or abs(z) > 9999:
                                continue
                            
                            if not element:
                                element = ''.join(c for c in atom_name if c.isalpha())[:2].upper()
                            
                            coordinates.append([x, y, z])
                            elements.append(element)
                            atom_names.append(atom_name)
                            residues.append(residue)
                            residue_ids.append(res_id)
                        except:
                            continue
        except:
            return None
        
        if len(coordinates) == 0:
            return None
        
        return AtomData(
            coordinates=np.array(coordinates, dtype=np.float64),
            elements=elements,
            atom_names=atom_names,
            residues=residues,
            residue_ids=residue_ids
        )


class MockDockingScorer:
    
    @staticmethod
    def calculate_mock_score(coords: np.ndarray, center: np.ndarray, 
                            box_size: np.ndarray, elements: List[str], 
                            residues: List[str]) -> float:
        """Calculate mock docking score (kcal/mol range)"""
        distances = np.linalg.norm(coords - center, axis=1)
        nearby_mask = distances < 15.0
        
        if np.sum(nearby_mask) < 5:
            return 0.0
        
        nearby_residues = [residues[i] for i in range(len(residues)) if nearby_mask[i]]
        
        pocket_depth = np.sum(nearby_mask) / 100.0
        
        polar_res = {'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'SER', 'THR', 'ASN', 'GLN', 'CYS'}
        hydrophobic = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'}
        
        polar_count = sum(1 for r in nearby_residues if r in polar_res)
        hydrophobic_count = sum(1 for r in nearby_residues if r in hydrophobic)
        
        chemical = (polar_count * 0.3 + hydrophobic_count * 0.2) / max(len(nearby_residues), 1)
        
        volume_penalty = np.prod(box_size) / 50000.0
        
        score = -(pocket_depth * 2.0 + chemical * 3.0 - volume_penalty * 0.5)
        
        return max(min(score, 0.0), -15.0)


class VinaDockingScorer:
    
    def __init__(self, vina_path: str):
        self.vina_path = vina_path
    
    def run_vina_docking(self, receptor_pdbqt: str, ligand_pdbqt: str,
                        center: np.ndarray, box_size: np.ndarray,
                        exhaustiveness: int = 8, verbose: bool = False) -> Tuple[Optional[float], str]:
        """Run Vina docking and return best score"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, 'config.txt')
            output_file = os.path.join(tmpdir, 'output.pdbqt')
            
            with open(config_file, 'w') as f:
                f.write(f"receptor = {receptor_pdbqt}\n")
                f.write(f"ligand = {ligand_pdbqt}\n")
                f.write(f"center_x = {center[0]:.3f}\n")
                f.write(f"center_y = {center[1]:.3f}\n")
                f.write(f"center_z = {center[2]:.3f}\n")
                f.write(f"size_x = {box_size[0]:.3f}\n")
                f.write(f"size_y = {box_size[1]:.3f}\n")
                f.write(f"size_z = {box_size[2]:.3f}\n")
                f.write(f"exhaustiveness = {exhaustiveness}\n")
                f.write(f"out = {output_file}\n")
            
            try:
                result = subprocess.run(
                    [self.vina_path, '--config', config_file],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode != 0:
                    error_msg = result.stderr[:200] if result.stderr else "Unknown error"
                    return None, f"exit_code_{result.returncode}"
                
                if result.stdout:
                    for line in result.stdout.split('\n'):
                        if 'mode' in line.lower() and '1' in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    score = float(parts[1])
                                    return score, ""
                                except:
                                    pass
                
                return None, "no_score_in_output"
                
            except subprocess.TimeoutExpired:
                return None, "timeout"
            except Exception as e:
                return None, f"exception: {str(e)[:50]}"
        
        return None, "unknown_error"


class ActiveSiteDetector:
    
    @staticmethod
    def calculate_pocket_score(coords: np.ndarray, center: np.ndarray, 
                              elements: List[str], residues: List[str]) -> float:
        """Calculate pocket quality score [0, 1]"""
        distances = np.linalg.norm(coords - center, axis=1)
        nearby_mask = distances < 15.0
        
        if np.sum(nearby_mask) < 10:
            return 0.0
        
        nearby_coords = coords[nearby_mask]
        nearby_residues = [residues[i] for i in range(len(residues)) if nearby_mask[i]]
        
        density = len(nearby_coords) / 4188.79
        
        unique_res = len(set(nearby_residues))
        diversity = min(unique_res / 20.0, 1.0)
        
        polar_res = {'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'SER', 'THR', 'ASN', 'GLN', 'CYS'}
        polar_count = sum(1 for r in nearby_residues if r in polar_res)
        polar_score = min(polar_count / len(nearby_residues), 0.5) if nearby_residues else 0
        
        pairwise_dist = cdist(nearby_coords, nearby_coords)
        avg_dist = np.mean(pairwise_dist[pairwise_dist > 0])
        concavity = 1.0 / (1.0 + avg_dist / 10.0)
        
        return 0.3 * density + 0.25 * diversity + 0.25 * polar_score + 0.2 * concavity
    
    @staticmethod
    def find_active_site(atom_data: AtomData) -> Tuple[np.ndarray, float]:
        """Find most likely active site"""
        coords = atom_data.coordinates
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        
        candidates = []
        
        ca_mask = [name == 'CA' for name in atom_data.atom_names]
        ca_coords = coords[ca_mask] if any(ca_mask) else coords[::10]
        candidates.extend(ca_coords[:min(50, len(ca_coords))])
        
        grid_size = 8
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    point = min_coords + (max_coords - min_coords) * np.array([
                        i / grid_size, j / grid_size, k / grid_size
                    ])
                    if np.min(np.linalg.norm(coords - point, axis=1)) < 5.0:
                        candidates.append(point)
        
        candidates = np.array(candidates)
        
        scores = [ActiveSiteDetector.calculate_pocket_score(
            coords, cand, atom_data.elements, atom_data.residues
        ) for cand in candidates]
        
        best_idx = np.argmax(scores)
        return candidates[best_idx], scores[best_idx]


class BayesianOptimizer:
    
    def __init__(self, bounds: np.ndarray, acquisition: str = 'ei'):
        self.bounds = bounds
        self.n_params = bounds.shape[0]
        self.acquisition = acquisition
        self.X_samples = []
        self.y_samples = []
    
    def suggest_next_point(self) -> np.ndarray:
        """Suggest next point to evaluate"""
        if len(self.X_samples) < 3:
            return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        
        X = np.array(self.X_samples)
        y = np.array(self.y_samples)
        
        n_candidates = 200
        candidates = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], 
            size=(n_candidates, self.n_params)
        )
        
        acquisition_values = []
        for candidate in candidates:
            distances = np.linalg.norm(X - candidate, axis=1)
            weights = np.exp(-distances / np.mean(distances))
            weights /= np.sum(weights)
            
            mu = np.sum(weights * y)
            sigma = np.sqrt(np.sum(weights * (y - mu)**2)) + 1e-6
            
            if self.acquisition == 'ei':
                y_best = np.min(y)
                z = (y_best - mu) / sigma
                ei = (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                acquisition_values.append(ei)
            else:
                kappa = 2.0
                ucb = mu - kappa * sigma
                acquisition_values.append(-ucb)
        
        best_idx = np.argmax(acquisition_values)
        return candidates[best_idx]
    
    def add_observation(self, X: np.ndarray, y: float):
        self.X_samples.append(X.copy())
        self.y_samples.append(y)
    
    def get_best(self) -> Tuple[np.ndarray, float]:
        if not self.y_samples:
            return None, None
        best_idx = np.argmin(self.y_samples)
        return np.array(self.X_samples[best_idx]), self.y_samples[best_idx]


class AutoPocketOptimizer:
    
    def __init__(self, padding: float = 10.0, vina_path: Optional[str] = None):
        self.padding = padding
        self.vina_path = vina_path
        self.vina_scorer = VinaDockingScorer(vina_path) if vina_path else None
    
    def calculate_center_of_mass(self, atom_data: AtomData) -> np.ndarray:
        masses = np.array([
            PDBParser.ATOMIC_MASSES.get(elem, 12.0) for elem in atom_data.elements
        ])
        total_mass = np.sum(masses)
        weighted_coords = atom_data.coordinates * masses[:, np.newaxis]
        return np.sum(weighted_coords, axis=0) / total_mass
    
    def validate_docking_box(self, center: np.ndarray, box_size: np.ndarray, 
                            atom_data: AtomData) -> Tuple[bool, str]:
        """Validate if docking box parameters are reasonable"""
        protein_coords = atom_data.coordinates
        
        distances_to_center = np.linalg.norm(protein_coords - center, axis=1)
        min_distance = np.min(distances_to_center)
        
        if min_distance > 15.0:
            return False, f"center_too_far (min_dist={min_distance:.1f}Å)"
        
        protein_min = np.min(protein_coords, axis=0)
        protein_max = np.max(protein_coords, axis=0)
        
        box_min = center - box_size / 2
        box_max = center + box_size / 2
        
        overlap_x = box_max[0] > protein_min[0] and box_min[0] < protein_max[0]
        overlap_y = box_max[1] > protein_min[1] and box_min[1] < protein_max[1]
        overlap_z = box_max[2] > protein_min[2] and box_min[2] < protein_max[2]
        
        if not (overlap_x and overlap_y and overlap_z):
            return False, "no_overlap_with_protein"
        
        if np.any(box_size < 15.0):
            return False, f"box_too_small (min={np.min(box_size):.1f}Å)"
        
        if np.any(box_size > 60.0):
            return False, f"box_too_large (max={np.max(box_size):.1f}Å)"
        
        atoms_in_box = np.sum(
            (protein_coords >= box_min) & (protein_coords <= box_max)
        ) / 3
        
        if atoms_in_box < 10:
            return False, f"too_few_atoms_in_box ({int(atoms_in_box)})"
        
        return True, "valid"
    
    def optimize_box(self, atom_data: AtomData, 
                    n_iter: int = 10,
                    mode: str = 'mock',
                    receptor_pdbqt: Optional[str] = None,
                    ligand_pdbqt: Optional[str] = None,
                    hybrid_switch: float = 0.5) -> DockingParameters:
        """Main optimization routine"""
        
        active_center, pocket_score = ActiveSiteDetector.find_active_site(atom_data)
        com = self.calculate_center_of_mass(atom_data)
        
        initial_center = 0.7 * active_center + 0.3 * com
        
        center_range = 12.0
        bounds = np.array([
            [initial_center[0] - center_range, initial_center[0] + center_range],
            [initial_center[1] - center_range, initial_center[1] + center_range],
            [initial_center[2] - center_range, initial_center[2] + center_range],
            [18.0, 45.0],
            [18.0, 45.0],
            [18.0, 45.0],
        ])
        
        optimizer = BayesianOptimizer(bounds)
        
        if mode == 'hybrid':
            switch_iter = int(n_iter * hybrid_switch)
            print(f"⚙️  Hybrid: {switch_iter} mock → {n_iter - switch_iter} Vina iterations")
        elif mode == 'vina':
            print(f"⚙️  Full Vina: {n_iter} iterations")
        else:
            print(f"⚙️  Mock: {n_iter} iterations")
        
        best_score = float('inf')
        best_params = None
        vina_failure_count = 0
        
        for iteration in range(n_iter):
            x = optimizer.suggest_next_point()
            center = x[:3]
            box_size = x[3:]
            
            is_valid, reason = self.validate_docking_box(center, box_size, atom_data)
            
            if not is_valid:
                score = MockDockingScorer.calculate_mock_score(
                    atom_data.coordinates, center, box_size,
                    atom_data.elements, atom_data.residues
                )
                score += 5.0
                scorer_used = f"mock(invalid:{reason})"
                
                optimizer.add_observation(x, score)
                continue
            
            use_vina_now = False
            if mode == 'vina':
                use_vina_now = True
            elif mode == 'hybrid' and iteration >= switch_iter:
                use_vina_now = True
            
            if use_vina_now and self.vina_scorer and receptor_pdbqt and ligand_pdbqt:
                score, error_msg = self.vina_scorer.run_vina_docking(
                    receptor_pdbqt, ligand_pdbqt, center, box_size, 
                    exhaustiveness=8, verbose=(iteration >= switch_iter)
                )
                
                if score is None:
                    vina_failure_count += 1
                    score = MockDockingScorer.calculate_mock_score(
                        atom_data.coordinates, center, box_size,
                        atom_data.elements, atom_data.residues
                    )
                    scorer_used = f"mock(vina_failed:{error_msg})"
                else:
                    scorer_used = "vina"
            else:
                score = MockDockingScorer.calculate_mock_score(
                    atom_data.coordinates, center, box_size,
                    atom_data.elements, atom_data.residues
                )
                scorer_used = "mock"
            
            optimizer.add_observation(x, score)
            
            if score < best_score:
                best_score = score
                best_params = x.copy()
            
            print(f"  Iter {iteration+1}/{n_iter}: score={score:.3f} ({scorer_used}), best={best_score:.3f}")
        
        if vina_failure_count > 0:
            print(f"\n⚠️  Vina failures: {vina_failure_count}/{n_iter} iterations")
        
        final_center = best_params[:3]
        final_box_size = best_params[3:]
        
        final_pocket_score = ActiveSiteDetector.calculate_pocket_score(
            atom_data.coordinates, final_center,
            atom_data.elements, atom_data.residues
        )
        
        box_volume = np.prod(final_box_size)
        
        return DockingParameters(
            center_x=float(final_center[0]),
            center_y=float(final_center[1]),
            center_z=float(final_center[2]),
            size_x=float(final_box_size[0]),
            size_y=float(final_box_size[1]),
            size_z=float(final_box_size[2]),
            n_atoms=len(atom_data.coordinates),
            box_volume=float(box_volume),
            pocket_score=float(final_pocket_score),
            docking_score=float(best_score),
            iterations=n_iter,
            optimized_by=mode
        )
    
    def calculate_simple(self, atom_data: AtomData) -> DockingParameters:
        active_center, pocket_score = ActiveSiteDetector.find_active_site(atom_data)
        com = self.calculate_center_of_mass(atom_data)
        center = 0.7 * active_center + 0.3 * com
        
        coords = atom_data.coordinates
        distances = np.linalg.norm(coords - center, axis=1)
        nearby_mask = distances < 20.0
        nearby_coords = coords[nearby_mask]
        
        if len(nearby_coords) >= 10:
            relative = nearby_coords - center
            p95 = np.percentile(np.abs(relative), 95, axis=0)
            box_size = 2 * p95 + self.padding
        else:
            relative = coords - center
            box_size = 2 * np.max(np.abs(relative), axis=0) + self.padding
        
        box_size = np.maximum(box_size, 15.0)
        box_size = np.minimum(box_size, 50.0)
        box_volume = np.prod(box_size)
        
        mock_score = MockDockingScorer.calculate_mock_score(
            coords, center, box_size, atom_data.elements, atom_data.residues
        )
        
        return DockingParameters(
            center_x=float(center[0]),
            center_y=float(center[1]),
            center_z=float(center[2]),
            size_x=float(box_size[0]),
            size_y=float(box_size[1]),
            size_z=float(box_size[2]),
            n_atoms=len(coords),
            box_volume=float(box_volume),
            pocket_score=float(pocket_score),
            docking_score=float(mock_score),
            iterations=0,
            optimized_by="simple"
        )


class ParameterWriter:
    
    @staticmethod
    def write_vina_config(params: DockingParameters, output_file: str, 
                         protein_name: str, padding: float):
        """Write Vina-compatible configuration file"""
        try:
            with open(output_file, 'w') as f:
                f.write("# AutoDock Vina Configuration File\n")
                f.write(f"# Generated for: {protein_name}\n")
                f.write(f"# Atoms: {params.n_atoms}\n")
                f.write(f"# Box volume: {params.box_volume:.2f} ų\n")
                f.write(f"# Pocket score: {params.pocket_score:.3f}\n")
                f.write(f"# Docking score: {params.docking_score:.3f} kcal/mol\n")
                f.write(f"# Optimization: {params.optimized_by} ({params.iterations} iter)\n")
                f.write(f"# Padding: {padding:.1f} Å\n")
                f.write("#\n# Reference: Trott & Olson (2010) J Comput Chem 31:455-461\n\n")
                
                f.write("# Docking box center (Å)\n")
                f.write(f"center_x = {params.center_x:.3f}\n")
                f.write(f"center_y = {params.center_y:.3f}\n")
                f.write(f"center_z = {params.center_z:.3f}\n\n")
                
                f.write("# Docking box size (Å)\n")
                f.write(f"size_x = {params.size_x:.3f}\n")
                f.write(f"size_y = {params.size_y:.3f}\n")
                f.write(f"size_z = {params.size_z:.3f}\n\n")
                
                f.write("# Recommended Vina parameters\n")
                f.write("exhaustiveness = 32\n")
                f.write("num_modes = 20\n")
                f.write("energy_range = 4\n")
            return True
        except:
            return False
    
    @staticmethod
    def write_csv_summary(results: List[Tuple[str, DockingParameters]], output_file: str):
        """Write summary CSV"""
        try:
            with open(output_file, 'w') as f:
                f.write("Protein,Center_X,Center_Y,Center_Z,Size_X,Size_Y,Size_Z,"
                       "N_Atoms,Box_Volume,Pocket_Score,Docking_Score,Iterations,Optimized_By\n")
                
                for name, params in results:
                    f.write(f"{name},"
                           f"{params.center_x:.3f},{params.center_y:.3f},{params.center_z:.3f},"
                           f"{params.size_x:.3f},{params.size_y:.3f},{params.size_z:.3f},"
                           f"{params.n_atoms},{params.box_volume:.2f},"
                           f"{params.pocket_score:.3f},{params.docking_score:.3f},"
                           f"{params.iterations},{params.optimized_by}\n")
            return True
        except:
            return False


def process_single_protein(pdb_file: str, output_dir: str, 
                          mode: str = 'mock',
                          n_iter: int = 10,
                          receptor_pdbqt: Optional[str] = None,
                          ligand_pdbqt: Optional[str] = None,
                          vina_path: Optional[str] = None,
                          padding: float = 10.0) -> Optional[Tuple[str, DockingParameters]]:
    protein_name = Path(pdb_file).stem
    
    # Validate and fix PDBQT files if in hybrid/vina mode
    if mode in ['hybrid', 'vina']:
        print(f"\n🔍 Validating PDBQT files...")
        
        # Validate receptor
        if receptor_pdbqt:
            is_valid, msg, fixed_path = PDBQTValidator.validate_and_fix_pdbqt(receptor_pdbqt)
            print(f"  Receptor: {msg}")
            if not is_valid:
                print(f"  ❌ Cannot proceed without valid receptor")
                return None
            receptor_pdbqt = fixed_path
        
        # Validate ligand
        if ligand_pdbqt:
            is_valid, msg, fixed_path = PDBQTValidator.validate_and_fix_pdbqt(ligand_pdbqt)
            print(f"  Ligand: {msg}")
            if not is_valid:
                print(f"  ❌ Cannot proceed without valid ligand")
                return None
            ligand_pdbqt = fixed_path
    
    # Parse PDB
    atom_data = PDBParser.parse_pdb(pdb_file)
    if atom_data is None:
        print(f"  ❌ Failed to parse: {protein_name}")
        return None
    
    print(f"\n🔬 Processing: {protein_name}")
    
    optimizer = AutoPocketOptimizer(padding=padding, vina_path=vina_path)
    
    if mode != 'mock' or n_iter > 0:
        params = optimizer.optimize_box(
            atom_data, n_iter=n_iter, mode=mode,
            receptor_pdbqt=receptor_pdbqt, ligand_pdbqt=ligand_pdbqt
        )
    else:
        params = optimizer.calculate_simple(atom_data)
    
    # Write output
    output_file = os.path.join(output_dir, f"{protein_name}_docking_params.txt")
    success = ParameterWriter.write_vina_config(params, output_file, protein_name, padding)
    
    if success:
        print(f"  ✓ Score: {params.docking_score:.3f}, Pocket: {params.pocket_score:.3f}")
        return (protein_name, params)
    return None


def batch_process(pdb_files: List[str], output_dir: str,
                 mode: str = 'mock',
                 n_iter: int = 10,
                 receptor_pdbqts: Optional[Dict[str, str]] = None,
                 ligand_pdbqt: Optional[str] = None,
                 vina_path: Optional[str] = None,
                 padding: float = 10.0,
                 n_workers: int = 1) -> List[Tuple[str, DockingParameters]]:
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for pdb_file in pdb_files:
        protein_name = Path(pdb_file).stem
        receptor_pdbqt = receptor_pdbqts.get(protein_name) if receptor_pdbqts else None
        
        result = process_single_protein(
            pdb_file, output_dir, mode, n_iter,
            receptor_pdbqt, ligand_pdbqt, vina_path, padding
        )
        if result:
            results.append(result)
    
    # Write summary
    if results:
        summary_file = os.path.join(output_dir, "docking_parameters_summary.csv")
        ParameterWriter.write_csv_summary(results, summary_file)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ZymEvo AutoPocket Finder and Docking Box Optimizer - Enhanced'
    )
    parser.add_argument('--pdbs', nargs='+', required=True, help='PDB files')
    parser.add_argument('--output', default='docking_params', help='Output directory')
    parser.add_argument('--mode', choices=['mock', 'hybrid', 'vina'], default='mock',
                       help='Optimization mode')
    parser.add_argument('--iterations', type=int, default=10, help='Optimization iterations')
    parser.add_argument('--padding', type=float, default=10.0, help='Box padding (Å)')
    parser.add_argument('--receptor_pdbqt', help='Receptor PDBQT (for hybrid/vina)')
    parser.add_argument('--ligand_pdbqt', help='Ligand PDBQT (for hybrid/vina)')
    parser.add_argument('--vina_path', help='Vina executable path')
    parser.add_argument('--download_vina', action='store_true', help='Download Vina')
    
    args = parser.parse_args()
    
    # Setup Vina if needed
    vina_path = args.vina_path
    if args.mode in ['hybrid', 'vina'] or args.download_vina:
        if not vina_path:
            vina_path = VinaInstaller.ensure_vina(args.download_vina)
        if not vina_path and args.mode in ['hybrid', 'vina']:
            print("⚠️  Vina unavailable, falling back to mock mode")
            args.mode = 'mock'
    
    print(f"\n{'='*70}")
    print(f"🧬 ZymEvo AutoPocket Optimizer - Enhanced Edition")
    print(f"{'='*70}")
    print(f"Mode: {args.mode}")
    print(f"Iterations: {args.iterations}")
    print(f"{'='*70}\n")
    
    results = batch_process(
        args.pdbs, args.output, args.mode, args.iterations,
        None, args.ligand_pdbqt, vina_path, args.padding
    )
    
    print(f"\n{'='*70}")
    print(f"✅ Complete: {len(results)}/{len(args.pdbs)} processed")
    print(f"{'='*70}")
