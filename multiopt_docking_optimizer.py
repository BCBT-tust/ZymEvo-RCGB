#!/usr/bin/env python3

import os
import sys
import platform
import subprocess
import urllib.request
import zipfile
import tarfile
import shutil
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import cdist
from scipy.stats import norm
import warnings
import tempfile

warnings.filterwarnings('ignore')


@dataclass
class AtomData:
    coordinates: np.ndarray
    elements: List[str]
    atom_names: List[str]
    residues: List[str]
    residue_ids: List[int]


@dataclass
class DockingParameters:
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


class VinaInstaller:
    
    VINA_RELEASES = {
        'linux': 'https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64',
        'darwin': 'https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_mac_catalina_64bit',
        'windows': 'https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_windows_x86_64.exe'
    }
    
    @staticmethod
    def get_vina_path() -> Optional[str]:
        """Check if Vina is already available"""
        # Check in PATH
        vina_cmd = 'vina' if platform.system() != 'Windows' else 'vina.exe'
        result = shutil.which(vina_cmd)
        if result:
            return result
        
        # Check in local directory
        local_vina = Path('./vina')
        if local_vina.exists() and os.access(local_vina, os.X_OK):
            return str(local_vina)
        
        return None
    
    @staticmethod
    def download_vina(output_dir: str = '.') -> Optional[str]:
        """Download AutoDock Vina executable"""
        system = platform.system().lower()
        
        if system not in VinaInstaller.VINA_RELEASES:
            print(f"❌ Unsupported OS: {system}")
            return None
        
        url = VinaInstaller.VINA_RELEASES[system]
        vina_name = 'vina' if system != 'windows' else 'vina.exe'
        vina_path = os.path.join(output_dir, vina_name)
        
        print(f"📥 Downloading AutoDock Vina from {url}")
        
        try:
            urllib.request.urlretrieve(url, vina_path)
            os.chmod(vina_path, 0o755)
            print(f"✅ Vina downloaded: {vina_path}")
            return vina_path
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    
    @staticmethod
    def ensure_vina(force_download: bool = False) -> Optional[str]:
        """Ensure Vina is available, download if needed"""
        if not force_download:
            vina_path = VinaInstaller.get_vina_path()
            if vina_path:
                print(f"✅ Vina found: {vina_path}")
                return vina_path
        
        print("⚙️ Vina not found, downloading...")
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
    def calculate_mock_score(coords: np.ndarray, center: np.ndarray, box_size: np.ndarray,
                            elements: List[str], residues: List[str]) -> float:
        """
        Calculate mock docking score
        Lower is better (like real docking scores in kcal/mol)
        """
        distances = np.linalg.norm(coords - center, axis=1)
        nearby_mask = distances < 15.0
        
        if np.sum(nearby_mask) < 5:
            return 0.0  # No pocket, neutral score
        
        nearby_coords = coords[nearby_mask]
        nearby_residues = [residues[i] for i in range(len(residues)) if nearby_mask[i]]
        
        # Pocket depth (deeper is better)
        pocket_depth = len(nearby_coords) / 100.0
        
        # Chemical environment
        polar_residues = {'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'SER', 'THR', 'ASN', 'GLN', 'CYS'}
        hydrophobic = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'}
        
        polar_count = sum(1 for r in nearby_residues if r in polar_residues)
        hydrophobic_count = sum(1 for r in nearby_residues if r in hydrophobic)
        
        chemical_score = (polar_count * 0.3 + hydrophobic_count * 0.2) / max(len(nearby_residues), 1)
        
        # Box size penalty (prefer compact boxes)
        volume = np.prod(box_size)
        size_penalty = volume / 50000.0  # Normalize
        
        # Combined score (negative for minimization)
        score = -(pocket_depth * 2.0 + chemical_score * 3.0 - size_penalty * 0.5)
        
        # Scale to kcal/mol range (-15 to 0)
        score = max(min(score, 0.0), -15.0)
        
        return score


class VinaDockingScorer:
    
    def __init__(self, vina_path: str):
        self.vina_path = vina_path
    
    def run_vina_docking(self, receptor_pdbqt: str, ligand_pdbqt: str,
                        center: np.ndarray, box_size: np.ndarray,
                        exhaustiveness: int = 8) -> Optional[float]:
        """Run Vina docking and return best score"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, 'config.txt')
            output_file = os.path.join(tmpdir, 'output.pdbqt')
            
            # Write config
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
                
                # Parse output for best score
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'mode' in line.lower() and '1' in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    score = float(parts[1])
                                    return score
                                except:
                                    pass
            except:
                pass
        
        return None


class ActiveSiteDetector:
    def calculate_pocket_score(coords: np.ndarray, center: np.ndarray, 
                              elements: List[str], residues: List[str]) -> float:
        distances = np.linalg.norm(coords - center, axis=1)
        nearby_mask = distances < 15.0
        
        if np.sum(nearby_mask) < 10:
            return 0.0
        
        nearby_coords = coords[nearby_mask]
        nearby_residues = [residues[i] for i in range(len(residues)) if nearby_mask[i]]
        
        density = len(nearby_coords) / 4188.79
        unique_residues = len(set(nearby_residues))
        diversity = min(unique_residues / 20.0, 1.0)
        
        polar_residues = {'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'SER', 'THR', 'ASN', 'GLN', 'CYS'}
        polar_count = sum(1 for r in nearby_residues if r in polar_residues)
        polar_score = min(polar_count / len(nearby_residues), 0.5) if len(nearby_residues) > 0 else 0
        
        pairwise_dist = cdist(nearby_coords, nearby_coords)
        avg_internal_dist = np.mean(pairwise_dist[pairwise_dist > 0])
        concavity = 1.0 / (1.0 + avg_internal_dist / 10.0)
        
        score = 0.3 * density + 0.25 * diversity + 0.25 * polar_score + 0.2 * concavity
        return score
    
    @staticmethod
    def find_active_site(atom_data: AtomData) -> Tuple[np.ndarray, float]:
        coords = atom_data.coordinates
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        
        candidates = []
        
        ca_mask = [name == 'CA' for name in atom_data.atom_names]
        ca_coords = coords[ca_mask] if any(ca_mask) else coords[::10]
        candidates.extend(ca_coords[:min(50, len(ca_coords))])
        
        grid_points = 8
        for i in range(grid_points):
            for j in range(grid_points):
                for k in range(grid_points):
                    point = min_coords + (max_coords - min_coords) * np.array([
                        i / grid_points, j / grid_points, k / grid_points
                    ])
                    if np.min(np.linalg.norm(coords - point, axis=1)) < 5.0:
                        candidates.append(point)
        
        candidates = np.array(candidates)
        
        scores = [ActiveSiteDetector.calculate_pocket_score(
            coords, candidate, atom_data.elements, atom_data.residues
        ) for candidate in candidates]
        
        best_idx = np.argmax(scores)
        return candidates[best_idx], scores[best_idx]


class BayesianOptimizer:
    
    def __init__(self, bounds: np.ndarray, acquisition_func: str = 'ei'):
        """
        bounds: (n_params, 2) array of (min, max) for each parameter
        acquisition_func: 'ei' (expected improvement) or 'ucb' (upper confidence bound)
        """
        self.bounds = bounds
        self.n_params = bounds.shape[0]
        self.acquisition_func = acquisition_func
        
        self.X_samples = []
        self.y_samples = []
    
    def suggest_next_point(self) -> np.ndarray:
        if len(self.X_samples) < 3:
            # Random sampling for initial points
            return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        
        X = np.array(self.X_samples)
        y = np.array(self.y_samples)
        
        # Simple GP approximation using local mean and std
        n_candidates = 200
        candidates = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], 
            size=(n_candidates, self.n_params)
        )
        
        acquisition_values = []
        for candidate in candidates:
            # Calculate distances to existing samples
            distances = np.linalg.norm(X - candidate, axis=1)
            weights = np.exp(-distances / np.mean(distances))
            weights /= np.sum(weights)
            
            # Weighted mean and std
            mu = np.sum(weights * y)
            sigma = np.sqrt(np.sum(weights * (y - mu)**2)) + 1e-6
            
            # Acquisition function
            if self.acquisition_func == 'ei':
                # Expected Improvement (for minimization)
                y_best = np.min(y)
                z = (y_best - mu) / sigma
                ei = (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                acquisition_values.append(ei)
            else:  # ucb
                kappa = 2.0
                ucb = mu - kappa * sigma  # Negative for minimization
                acquisition_values.append(-ucb)
        
        best_idx = np.argmax(acquisition_values)
        return candidates[best_idx]
    
    def add_observation(self, X: np.ndarray, y: float):
        self.X_samples.append(X.copy())
        self.y_samples.append(y)
    
    def get_best(self) -> Tuple[np.ndarray, float]:
        """Get best observed point"""
        if len(self.y_samples) == 0:
            return None, None
        
        best_idx = np.argmin(self.y_samples)
        return np.array(self.X_samples[best_idx]), self.y_samples[best_idx]


class MultiOptDockingOptimizer:
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
    
    def optimize_with_bayes(self, atom_data: AtomData, 
                           n_iter: int = 20,
                           use_vina: bool = False,
                           hybrid: bool = True,
                           hybrid_switch: float = 0.5,
                           receptor_pdbqt: Optional[str] = None,
                           ligand_pdbqt: Optional[str] = None) -> DockingParameters:
        
        initial_center, pocket_score = ActiveSiteDetector.find_active_site(atom_data)
        com = self.calculate_center_of_mass(atom_data)
        
        # Blend initial guess
        initial_center = 0.7 * initial_center + 0.3 * com
        
        # Define optimization bounds
        # Parameters: [center_x, center_y, center_z, size_x, size_y, size_z]
        center_range = 15.0
        bounds = np.array([
            [initial_center[0] - center_range, initial_center[0] + center_range],
            [initial_center[1] - center_range, initial_center[1] + center_range],
            [initial_center[2] - center_range, initial_center[2] + center_range],
            [15.0, 50.0],  # size_x
            [15.0, 50.0],  # size_y
            [15.0, 50.0],  # size_z
        ])
        
        optimizer = BayesianOptimizer(bounds, acquisition_func='ei')
        
        if use_vina and hybrid:
            mode = "hybrid"
            switch_iter = int(n_iter * hybrid_switch)
            print(f"⚙️ Hybrid mode: {switch_iter} mock iterations → {n_iter - switch_iter} Vina iterations")
        elif use_vina:
            mode = "vina_only"
            print(f"⚙️ Full Vina mode: {n_iter} Vina iterations")
        else:
            mode = "mock_only"
            print(f"⚙️ Quick mock mode: {n_iter} iterations")
        
        best_score = float('inf')
        best_params = None
        
        for iteration in range(n_iter):
            # Get next point to evaluate
            x = optimizer.suggest_next_point()
            center = x[:3]
            box_size = x[3:]
            
            # Determine scorer for this iteration
            use_vina_now = False
            if mode == "vina_only":
                use_vina_now = True
            elif mode == "hybrid" and iteration >= switch_iter:
                use_vina_now = True
            
            # Evaluate
            if use_vina_now and self.vina_scorer and receptor_pdbqt and ligand_pdbqt:
                score = self.vina_scorer.run_vina_docking(
                    receptor_pdbqt, ligand_pdbqt, center, box_size, exhaustiveness=8
                )
                if score is None:
                    # Fallback to mock if Vina fails
                    score = MockDockingScorer.calculate_mock_score(
                        atom_data.coordinates, center, box_size,
                        atom_data.elements, atom_data.residues
                    )
                    scorer_used = "mock(fallback)"
                else:
                    scorer_used = "vina"
            else:
                score = MockDockingScorer.calculate_mock_score(
                    atom_data.coordinates, center, box_size,
                    atom_data.elements, atom_data.residues
                )
                scorer_used = "mock"
            
            optimizer.add_observation(x, score)
            
            # Track best
            if score < best_score:
                best_score = score
                best_params = x.copy()
            
            print(f"  Iter {iteration+1}/{n_iter}: score={score:.3f} ({scorer_used}), best={best_score:.3f}")
        
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
    
    def calculate_parameters_simple(self, atom_data: AtomData) -> DockingParameters:
        initial_center, pocket_score = ActiveSiteDetector.find_active_site(atom_data)
        com = self.calculate_center_of_mass(atom_data)
        center = 0.7 * initial_center + 0.3 * com
        
        coords = atom_data.coordinates
        distances = np.linalg.norm(coords - center, axis=1)
        nearby_mask = distances < 20.0
        nearby_coords = coords[nearby_mask]
        
        if len(nearby_coords) >= 10:
            relative_coords = nearby_coords - center
            percentile_95 = np.percentile(np.abs(relative_coords), 95, axis=0)
            box_size = 2 * percentile_95 + self.padding
        else:
            relative_coords = coords - center
            box_size = 2 * np.max(np.abs(relative_coords), axis=0) + self.padding
        
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
    def write_vina_config(params: DockingParameters, output_file: str, 
                         protein_file: str, padding: float):
        try:
            with open(output_file, 'w') as f:
                f.write("# AutoDock Vina Configuration File\n")
                f.write(f"# Generated for: {Path(protein_file).name}\n")
                f.write(f"# Number of atoms: {params.n_atoms}\n")
                f.write(f"# Docking box volume: {params.box_volume:.2f} ų\n")
                f.write(f"# Pocket score: {params.pocket_score:.3f}\n")
                f.write(f"# Docking score: {params.docking_score:.3f} kcal/mol\n")
                f.write(f"# Optimization: {params.optimized_by} ({params.iterations} iter)\n")
                f.write(f"# Padding: {padding:.1f} Å\n")
                f.write("#\n# Reference: AutoDock Vina (Trott & Olson, 2010)\n")
                f.write("# DOI: 10.1002/jcc.21334\n\n")
                
                f.write("# Docking box center (Å)\n")
                f.write(f"center_x = {params.center_x:.3f}\n")
                f.write(f"center_y = {params.center_y:.3f}\n")
                f.write(f"center_z = {params.center_z:.3f}\n\n")
                
                f.write("# Docking box size (Å)\n")
                f.write(f"size_x = {params.size_x:.3f}\n")
                f.write(f"size_y = {params.size_y:.3f}\n")
                f.write(f"size_z = {params.size_z:.3f}\n\n")
                
                f.write("# Recommended parameters\n")
                f.write("exhaustiveness = 32\n")
                f.write("num_modes = 20\n")
                f.write("energy_range = 4\n")
            return True
        except:
            return False
    
    def write_csv_summary(results: List[Tuple[str, DockingParameters]], output_file: str):
        try:
            with open(output_file, 'w') as f:
                f.write("Protein,Center_X,Center_Y,Center_Z,Size_X,Size_Y,Size_Z,"
                       "N_Atoms,Box_Volume,Pocket_Score,Docking_Score,Iterations,Optimized_By\n")
                
                for protein_name, params in results:
                    f.write(f"{protein_name},"
                           f"{params.center_x:.3f},{params.center_y:.3f},{params.center_z:.3f},"
                           f"{params.size_x:.3f},{params.size_y:.3f},{params.size_z:.3f},"
                           f"{params.n_atoms},{params.box_volume:.2f},"
                           f"{params.pocket_score:.3f},{params.docking_score:.3f},"
                           f"{params.iterations},{params.optimized_by}\n")
            return True
        except:
            return False


def process_single_pdb(pdb_file: str, output_dir: str, 
                      optimize: bool = False,
                      n_iter: int = 20,
                      use_vina: bool = False,
                      hybrid: bool = True,
                      vina_path: Optional[str] = None,
                      padding: float = 10.0) -> Optional[Tuple[str, DockingParameters]]:
    protein_name = Path(pdb_file).stem
    
    atom_data = PDBParser.parse_pdb(pdb_file)
    if atom_data is None:
        print(f"  ❌ Failed to parse: {protein_name}")
        return None
    
    print(f"\n🔬 Processing: {protein_name}")
    
    optimizer = MultiOptDockingOptimizer(padding=padding, vina_path=vina_path)
    
    if optimize:
        params = optimizer.optimize_with_bayes(
            atom_data, n_iter=n_iter, use_vina=use_vina, hybrid=hybrid
        )
    else:
        params = optimizer.calculate_parameters_simple(atom_data)
    
    output_file = os.path.join(output_dir, f"{protein_name}_docking_params.txt")
    success = ParameterWriter.write_vina_config(params, output_file, pdb_file, padding)
    
    if success:
        print(f"  ✓ Score: {params.docking_score:.3f}, Mode: {params.optimized_by}")
        return (protein_name, params)
    return None


def batch_process_pdbs(pdb_files: List[str], output_dir: str, 
                      optimize: bool = False,
                      n_iter: int = 20,
                      use_vina: bool = False,
                      hybrid: bool = True,
                      vina_path: Optional[str] = None,
                      padding: float = 10.0,
                      n_workers: int = 1) -> List[Tuple[str, DockingParameters]]:
    """Batch process PDB files"""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    if n_workers > 1 and len(pdb_files) > 1 and not optimize:
        # Parallel only for simple mode
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(process_single_pdb, pdb, output_dir, 
                              False, n_iter, False, False, vina_path, padding): pdb
                for pdb in pdb_files
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
    else:
        # Sequential for optimization mode
        for pdb_file in pdb_files:
            result = process_single_pdb(
                pdb_file, output_dir, optimize, n_iter, 
                use_vina, hybrid, vina_path, padding
            )
            if result:
                results.append(result)
    
    if results:
        summary_file = os.path.join(output_dir, "docking_parameters_summary.csv")
        ParameterWriter.write_csv_summary(results, summary_file)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MultiOpt Docking Optimizer')
    parser.add_argument('--pdbs', nargs='+', required=True, help='PDB files')
    parser.add_argument('--output', default='docking_params', help='Output directory')
    parser.add_argument('--mode', choices=['quick', 'hybrid', 'full'], default='quick',
                       help='Mode: quick(mock), hybrid(mock+vina), full(vina)')
    parser.add_argument('--optimize', action='store_true', help='Enable Bayesian optimization')
    parser.add_argument('--iterations', type=int, default=20, help='Optimization iterations')
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers')
    parser.add_argument('--padding', type=float, default=10.0, help='Box padding (Å)')
    parser.add_argument('--vina_path', help='Path to Vina executable')
    parser.add_argument('--download_vina', action='store_true', help='Auto-download Vina')
    
    args = parser.parse_args()
    
    # Setup Vina
    vina_path = None
    if args.mode in ['hybrid', 'full'] or args.download_vina:
        vina_path = VinaInstaller.ensure_vina(force_download=args.download_vina)
        if not vina_path and args.mode in ['hybrid', 'full']:
            print("⚠️ Vina not available, falling back to mock mode")
    
    # Determine mode settings
    use_vina = args.mode in ['hybrid', 'full'] and vina_path is not None
    hybrid = args.mode == 'hybrid'
    
    print(f"\n{'='*70}")
    print(f"🧬 MultiOpt Docking Optimizer")
    print(f"{'='*70}")
    print(f"Mode: {args.mode}")
    print(f"Optimize: {args.optimize}")
    print(f"Iterations: {args.iterations if args.optimize else 'N/A'}")
    print(f"Workers: {args.workers}")
    print(f"{'='*70}\n")
    
    results = batch_process_pdbs(
        args.pdbs, args.output, args.optimize, args.iterations,
        use_vina, hybrid, vina_path, args.padding, args.workers
    )
    
    print(f"\n{'='*70}")
    print(f"✅ Complete: {len(results)}/{len(args.pdbs)} processed")
    print(f"{'='*70}")
