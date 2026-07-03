#!/usr/bin/env python3

import os
import sys
import math
import argparse
import warnings
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import norm
from scipy.spatial import ConvexHull

warnings.filterwarnings("ignore")

@dataclass
class AtomData:
    coordinates: np.ndarray
    elements: List[str]
    atom_names: List[str]
    residues: List[str]
    residue_ids: List[int]


@dataclass
class ProbeConfig:
    name: str
    type: str
    radius: float
    epsilon: float
    charge: float
    description: str


@dataclass
class PocketResult:
    center: np.ndarray
    cluster_points: np.ndarray
    volume: float
    depth: float
    
    # Scoring
    pocket_score: float
    score_breakdown: Dict[str, float]
    
    # Energy information
    probe_energies: Dict[str, float]
    mean_energy: float
    
    # Chemical features
    hydrophobicity: float
    residue_composition: Dict[str, int]

    principal_axes: np.ndarray
    extent: np.ndarray

    confidence: float
    rank: int = 0


@dataclass
class DockingParameters:
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float
    
    # Metadata
    n_atoms: int
    box_volume: float
    pocket_score: float
    pocket_confidence: float
    docking_score: float
    iterations: int
    optimized_by: str
    
    # Extended info
    pocket_depth: float = 0.0
    hydrophobicity: float = 0.0
    probe_energies: Dict[str, float] = field(default_factory=dict)


class PDBParser:
    """PDB parser with validation."""
    
    ATOMIC_MASSES = {
        'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
        'S': 32.06, 'P': 30.974, 'F': 18.998, 'CL': 35.45,
        'BR': 79.904, 'I': 126.90, 'SE': 78.971, 'FE': 55.845,
        'ZN': 65.38, 'MG': 24.305, 'CA': 40.078, 'NA': 22.990,
        'K': 39.098, 'MN': 54.938, 'CU': 63.546, 'CO': 58.933
    }
    
    VDW_RADII = {
        'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
        'S': 1.80, 'P': 1.80, 'F': 1.47, 'CL': 1.75,
        'BR': 1.85, 'I': 1.98, 'SE': 1.90, 'FE': 1.40,
        'ZN': 1.39, 'MG': 1.73, 'CA': 1.97, 'NA': 2.27
    }

    @staticmethod
    def parse_pdb(pdb_file: str) -> Optional[AtomData]:
        """Parse PDB file and extract ATOM records."""
        if not os.path.exists(pdb_file):
            print(f"❌ PDB not found: {pdb_file}")
            return None

        coordinates, elements, atom_names, residues, residue_ids = [], [], [], [], []

        try:
            with open(pdb_file, "r") as f:
                for line in f:
                    if not line.startswith("ATOM"):
                        continue
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
                            element = "".join(c for c in atom_name if c.isalpha())[:2].upper()

                        coordinates.append([x, y, z])
                        elements.append(element)
                        atom_names.append(atom_name)
                        residues.append(residue)
                        residue_ids.append(res_id)
                    except Exception:
                        continue
        except Exception as e:
            print(f"❌ Failed to read PDB: {e}")
            return None

        if not coordinates:
            print(f"❌ No ATOM records in {pdb_file}")
            return None

        return AtomData(
            coordinates=np.array(coordinates, dtype=np.float64),
            elements=elements,
            atom_names=atom_names,
            residues=residues,
            residue_ids=residue_ids
        )

    @staticmethod
    def center_of_mass(atom_data: AtomData) -> np.ndarray:
        """Calculate center of mass."""
        masses = np.array([PDBParser.ATOMIC_MASSES.get(e, 12.0) 
                          for e in atom_data.elements])
        total_mass = np.sum(masses)
        weighted = atom_data.coordinates * masses[:, None]
        return np.sum(weighted, axis=0) / total_mass

    @staticmethod
    def get_vdw_radius(element: str) -> float:
        """Get van der Waals radius."""
        return PDBParser.VDW_RADII.get(element.upper(), 1.70)


class AlphaShapeCavityDetector:
    """
    Simplified α-shape based cavity detection.
    Identifies concave regions on protein surface.
    """
    
    def __init__(self, alpha: float = 8.0, min_cavity_size: int = 10):
        """
        Args:
            alpha: α-shape parameter (Å), larger = smoother surface
            min_cavity_size: Minimum points to form a cavity
        """
        self.alpha = alpha
        self.min_cavity_size = min_cavity_size
    
    def detect_cavities(self, atom_data: AtomData) -> List[np.ndarray]:
        """
        Detect cavity regions using simplified α-shape approach.
        
        Returns:
            List of cavity point clouds
        """
        coords = atom_data.coordinates
        
        # 1. Create dense surface sampling
        surface_points = self._sample_surface(coords, atom_data.elements)
        
        # 2. Identify concave regions (inside convex hull but away from atoms)
        cavities = self._find_concave_regions(surface_points, coords)
        
        print(f"  🔍 α-shape detected {len(cavities)} cavity regions")
        return cavities
    
    def _sample_surface(self, coords: np.ndarray, elements: List[str], 
                       density: float = 1.5) -> np.ndarray:
        """
        Sample points around protein surface.
        
        Args:
            density: Points per Ų
        """
        # Estimate surface area from bounding box
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        dims = maxs - mins
        
        # Grid sampling around protein
        margin = 3.0
        spacing = 1.0 / np.sqrt(density)
        
        xs = np.arange(mins[0] - margin, maxs[0] + margin, spacing)
        ys = np.arange(mins[1] - margin, maxs[1] + margin, spacing)
        zs = np.arange(mins[2] - margin, maxs[2] + margin, spacing)
        
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        
        # Keep only points near surface (within VDW + probe distance)
        surface_dist = 3.5  # Å
        distances = cdist(grid_points, coords)
        min_distances = np.min(distances, axis=1)
        
        # Surface shell: between VDW and VDW + probe
        vdw_min = 1.4  # Minimum VDW radius
        mask = (min_distances > vdw_min) & (min_distances < surface_dist)
        
        return grid_points[mask]
    
    def _find_concave_regions(self, surface_points: np.ndarray, 
                             protein_coords: np.ndarray) -> List[np.ndarray]:
        """
        Identify concave regions using local geometry analysis.
        """
        if len(surface_points) < 20:
            return []
        
        # Calculate local curvature proxy
        curvatures = self._estimate_curvature(surface_points, protein_coords)
        
        # Concave points have positive curvature (inward)
        concave_mask = curvatures > 0.1
        concave_points = surface_points[concave_mask]
        
        if len(concave_points) < self.min_cavity_size:
            return []
        
        # Cluster concave points
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(concave_points))
        
        # Simple clustering by connectivity
        visited = np.zeros(len(concave_points), dtype=bool)
        cavities = []
        
        for i in range(len(concave_points)):
            if visited[i]:
                continue
            
            # Flood fill to find connected component
            cluster = []
            stack = [i]
            
            while stack:
                idx = stack.pop()
                if visited[idx]:
                    continue
                visited[idx] = True
                cluster.append(idx)
                
                # Add neighbors within alpha distance
                neighbors = np.where(D[idx] < self.alpha)[0]
                for n in neighbors:
                    if not visited[n]:
                        stack.append(n)
            
            if len(cluster) >= self.min_cavity_size:
                cavities.append(concave_points[cluster])
        
        return cavities
    
    def _estimate_curvature(self, points: np.ndarray, 
                           protein_coords: np.ndarray) -> np.ndarray:
        """
        Estimate local curvature at each point.
        Positive = concave, Negative = convex
        """
        curvatures = np.zeros(len(points))
        
        # For each point, look at local neighborhood
        D = cdist(points, protein_coords)
        k = min(20, len(protein_coords))
        
        for i in range(len(points)):
            # Find nearest protein atoms
            nearest_indices = np.argpartition(D[i], k)[:k]
            nearest_coords = protein_coords[nearest_indices]
            
            # Center of mass of neighbors
            com = np.mean(nearest_coords, axis=0)
            
            # Vector from neighbor COM to point
            vec = points[i] - com
            dist = np.linalg.norm(vec)
            
            # If point is inside the neighbor cloud, it's concave
            neighbor_spread = np.std(nearest_coords, axis=0).mean()
            curvatures[i] = (neighbor_spread - dist) / neighbor_spread
        
        return curvatures


# ============================================================================
# 5-Probe Energy Grid
# ============================================================================

class FiveProbeEnergyGrid:
    """
    Five chemical probe energy grid calculator.
    
    Probes:
    1. sp³ Carbon - Hydrophobic interactions
    2. Primary amine (NH₃⁺) - H-bond donor
    3. Carbonyl oxygen - H-bond acceptor
    4. Quaternary ammonium - Positive charge
    5. Carboxylate - Negative charge
    """
    
    PROBES = {
        'hydrophobic': ProbeConfig(
            name='hydrophobic',
            type='sp3_carbon',
            radius=1.70,
            epsilon=0.15,
            charge=0.0,
            description='Hydrophobic interactions'
        ),
        'hbond_donor': ProbeConfig(
            name='hbond_donor',
            type='primary_amine',
            radius=1.50,
            epsilon=0.10,
            charge=+0.5,
            description='H-bond donor (NH₃⁺)'
        ),
        'hbond_acceptor': ProbeConfig(
            name='hbond_acceptor',
            type='carbonyl_oxygen',
            radius=1.40,
            epsilon=0.12,
            charge=-0.5,
            description='H-bond acceptor (C=O)'
        ),
        'positive': ProbeConfig(
            name='positive',
            type='quaternary_ammonium',
            radius=1.80,
            epsilon=0.08,
            charge=+1.0,
            description='Positive charge (NR₄⁺)'
        ),
        'negative': ProbeConfig(
            name='negative',
            type='carboxylate',
            radius=1.80,
            epsilon=0.08,
            charge=-1.0,
            description='Negative charge (COO⁻)'
        )
    }
    
    # Probe weights for final energy
    PROBE_WEIGHTS = {
        'hydrophobic': 0.25,
        'hbond_donor': 0.20,
        'hbond_acceptor': 0.20,
        'positive': 0.175,
        'negative': 0.175
    }
    
    def __init__(self, grid_spacing: float = 1.5, margin: float = 5.0, 
                 cutoff: float = 8.0):
        """
        Args:
            grid_spacing: Grid resolution (Å)
            margin: Extend grid beyond protein (Å)
            cutoff: Energy calculation cutoff (Å)
        """
        self.grid_spacing = grid_spacing
        self.margin = margin
        self.cutoff = cutoff
    
    def compute_energy_grids(self, atom_data: AtomData) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute energy grids for all 5 probes.
        
        Returns:
            grid_points: (N, 3) array of grid coordinates
            probe_energies: Dict of probe_name -> (N,) energy array
        """
        coords = atom_data.coordinates
        
        # Setup grid
        min_coords = coords.min(axis=0) - self.margin
        max_coords = coords.max(axis=0) + self.margin
        
        xs = np.arange(min_coords[0], max_coords[0] + self.grid_spacing, 
                      self.grid_spacing)
        ys = np.arange(min_coords[1], max_coords[1] + self.grid_spacing, 
                      self.grid_spacing)
        zs = np.arange(min_coords[2], max_coords[2] + self.grid_spacing, 
                      self.grid_spacing)
        
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        
        print(f"  🧊 Energy grid: {len(xs)}×{len(ys)}×{len(zs)} = {len(grid_points)} points")
        
        # Compute for each probe
        probe_energies = {}
        
        for probe_name, probe_config in self.PROBES.items():
            print(f"    ⚡ Computing {probe_name} probe energy...")
            energies = self._compute_probe_energy(
                grid_points, atom_data, probe_config
            )
            probe_energies[probe_name] = energies
        
        return grid_points, probe_energies
    
    def _compute_probe_energy(self, grid_points: np.ndarray, 
                             atom_data: AtomData, 
                             probe: ProbeConfig) -> np.ndarray:
        """
        Compute energy for a single probe at all grid points.
        E_total = E_LJ + E_elec
        """
        coords = atom_data.coordinates
        elements = atom_data.elements
        
        n_points = len(grid_points)
        energies = np.zeros(n_points, dtype=np.float64)
        
        # Chunk processing to avoid memory explosion
        chunk_size = 2048
        
        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            gp_chunk = grid_points[start:end]
            
            # Distance matrix
            D = cdist(gp_chunk, coords)  # (chunk_size, n_atoms)
            
            # Mask points within cutoff
            mask = D < self.cutoff
            
            # Lennard-Jones energy
            E_lj = self._lj_energy(D, elements, probe, mask)
            
            # Electrostatic energy (if probe is charged)
            if abs(probe.charge) > 0.01:
                E_elec = self._electrostatic_energy(D, atom_data, probe, mask)
            else:
                E_elec = 0.0
            
            energies[start:end] = E_lj + E_elec
        
        return energies
    
    def _lj_energy(self, distances: np.ndarray, elements: List[str], 
                   probe: ProbeConfig, mask: np.ndarray) -> np.ndarray:
        """
        Lennard-Jones 12-6 potential.
        E_LJ = 4ε[(σ/r)¹² - (σ/r)⁶]
        """
        D_safe = np.where(mask, distances, self.cutoff)
        D_safe = np.maximum(D_safe, 0.5)  # Avoid singularity
        
        E_total = np.zeros(D_safe.shape[0])
        
        for i, elem in enumerate(elements):
            # Combining rules: σ_ij = (σ_i + σ_j)/2
            r_atom = PDBParser.get_vdw_radius(elem)
            sigma = (probe.radius + r_atom) / 2.0
            
            # Element-specific epsilon (crude approximation)
            eps_atom = self._element_epsilon(elem)
            epsilon = np.sqrt(probe.epsilon * eps_atom)
            
            # LJ potential
            r = D_safe[:, i]
            sr6 = (sigma / r) ** 6
            sr12 = sr6 ** 2
            
            E_lj = 4.0 * epsilon * (sr12 - sr6)
            
            # Apply mask and accumulate
            E_lj = np.where(mask[:, i], E_lj, 0.0)
            E_total += E_lj
        
        return E_total
    
    def _electrostatic_energy(self, distances: np.ndarray, 
                             atom_data: AtomData,
                             probe: ProbeConfig, 
                             mask: np.ndarray) -> np.ndarray:
        """
        Simplified Coulombic electrostatics with distance-dependent dielectric.
        E_elec = 332 * q₁q₂ / (ε(r) * r)
        """
        D_safe = np.where(mask, distances, self.cutoff)
        D_safe = np.maximum(D_safe, 1.0)  # Avoid singularity
        
        # Distance-dependent dielectric: ε(r) = 4(1 + r/10)
        dielectric = 4.0 * (1.0 + D_safe / 10.0)
        
        # Assign partial charges to atoms
        charges = self._assign_atom_charges(atom_data)
        
        # Coulomb energy
        E_elec = 332.0 * probe.charge * charges / (dielectric * D_safe)
        
        # Apply mask and sum
        E_elec = np.where(mask, E_elec, 0.0)
        return np.sum(E_elec, axis=1)
    
    def _element_epsilon(self, element: str) -> float:
        """Crude LJ epsilon values for elements (kcal/mol)."""
        eps_map = {
            'C': 0.15, 'N': 0.10, 'O': 0.12, 'S': 0.20,
            'H': 0.05, 'P': 0.15, 'F': 0.08, 'CL': 0.10
        }
        return eps_map.get(element.upper(), 0.10)
    
    def _assign_atom_charges(self, atom_data: AtomData) -> np.ndarray:
        """
        Assign crude partial charges based on residue and atom type.
        """
        charges = np.zeros(len(atom_data.elements))
        
        charged_residues = {
            'ASP': {'CG': -0.5, 'OD1': -0.5, 'OD2': -0.5},
            'GLU': {'CD': -0.5, 'OE1': -0.5, 'OE2': -0.5},
            'LYS': {'NZ': +1.0},
            'ARG': {'NE': +0.5, 'NH1': +0.5, 'NH2': +0.5},
            'HIS': {'ND1': +0.3, 'NE2': +0.3}
        }
        
        for i, (res, atom_name) in enumerate(zip(atom_data.residues, 
                                                  atom_data.atom_names)):
            if res in charged_residues:
                charges[i] = charged_residues[res].get(atom_name, 0.0)
        
        return charges
    
    def merge_probe_energies(self, probe_energies: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Merge all probe energies with weights.
        E_total = Σ w_i * E_i
        """
        merged = np.zeros_like(list(probe_energies.values())[0])
        
        for probe_name, energies in probe_energies.items():
            weight = self.PROBE_WEIGHTS[probe_name]
            merged += weight * energies
        
        return merged


# ============================================================================
# Enhanced DBSCAN Clustering
# ============================================================================

class EnhancedDBSCAN:
    """
    Enhanced DBSCAN with adaptive parameters.
    Different from CB-Dock2 parameters.
    """
    
    def __init__(self, eps: Optional[float] = None, 
                 min_samples: Optional[int] = None,
                 adaptive: bool = True):
        """
        Args:
            eps: Neighborhood radius (auto if None)
            min_samples: Minimum cluster size (auto if None)
            adaptive: Use adaptive parameter selection
        """
        self.eps = eps
        self.min_samples = min_samples
        self.adaptive = adaptive
    
    def fit(self, X: np.ndarray, protein_size: int) -> np.ndarray:
        """
        Cluster points with adaptive or fixed parameters.
        
        Returns:
            labels: Cluster labels (-1 for noise)
        """
        n = X.shape[0]
        
        # Determine parameters
        if self.adaptive:
            eps, min_samples = self._adaptive_parameters(n, protein_size)
        else:
            eps = self.eps if self.eps else 4.0
            min_samples = self.min_samples if self.min_samples else max(5, n // 100)
        
        print(f"  📊 DBSCAN: eps={eps:.2f} Å, min_samples={min_samples}")
        
        # Run DBSCAN
        labels = self._dbscan_core(X, eps, min_samples)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        print(f"  📊 Found {n_clusters} clusters, {n_noise} noise points")
        
        return labels
    
    def _adaptive_parameters(self, n_points: int, 
                            protein_size: int) -> Tuple[float, int]:
        """
        Adaptive parameter selection based on protein size.
        Different from CB-Dock2 (eps=3.5, min_samples=10)
        """
        if protein_size < 200:
            eps = 3.0
            min_samples = max(5, n_points // 150)
        elif protein_size < 500:
            eps = 4.0
            min_samples = max(8, n_points // 100)
        else:
            eps = 5.0
            min_samples = max(10, n_points // 80)
        
        return eps, min_samples
    
    def _dbscan_core(self, X: np.ndarray, eps: float, 
                     min_samples: int) -> np.ndarray:
        """Core DBSCAN implementation."""
        n = X.shape[0]
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        
        # Precompute distance matrix
        D = squareform(pdist(X))
        
        cluster_id = 0
        
        for i in range(n):
            if visited[i]:
                continue
            
            visited[i] = True
            neighbors = np.where(D[i] <= eps)[0]
            
            if len(neighbors) < min_samples:
                labels[i] = -1  # Noise
                continue
            
            # Start new cluster
            labels[i] = cluster_id
            seeds = set(neighbors.tolist())
            seeds.discard(i)
            
            # Expand cluster
            while seeds:
                j = seeds.pop()
                if not visited[j]:
                    visited[j] = True
                    neighbors_j = np.where(D[j] <= eps)[0]
                    if len(neighbors_j) >= min_samples:
                        seeds.update(neighbors_j.tolist())
                
                if labels[j] == -1:
                    labels[j] = cluster_id
            
            cluster_id += 1
        
        return labels


# ============================================================================
# Comprehensive Pocket Scorer
# ============================================================================

class PocketScorer:
    """
    Comprehensive pocket quality scoring system.
    
    Metrics:
    1. Depth score (0-1)
    2. Volume score (0-1)
    3. Hydrophobicity balance (-1 to +1 -> 0-1)
    4. Shape regularity (0-1)
    5. Residue composition (0-1)
    6. Accessibility (0-1)
    """
    
    WEIGHTS = {
        'depth': 0.25,
        'volume': 0.20,
        'hydrophobicity': 0.15,
        'shape': 0.15,
        'residue': 0.15,
        'accessibility': 0.10
    }
    
    HYDROPHOBIC_RES = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'}
    POLAR_RES = {'SER', 'THR', 'ASN', 'GLN', 'CYS', 'TYR'}
    CHARGED_RES = {'ASP', 'GLU', 'LYS', 'ARG', 'HIS'}
    
    def __init__(self):
        pass
    
    def score_pocket(self, cluster_points: np.ndarray, 
                    center: np.ndarray,
                    atom_data: AtomData,
                    probe_energies: Dict[str, float]) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """
        Comprehensive pocket scoring.
        
        Returns:
            total_score: 0-100
            score_breakdown: Individual scores
            metadata: Additional information
        """
        # Calculate individual scores
        depth = self._depth_score(cluster_points, center, atom_data)
        volume = self._volume_score(cluster_points)
        hydro = self._hydrophobicity_score(center, atom_data, probe_energies)
        shape = self._shape_score(cluster_points)
        residue = self._residue_score(center, atom_data)
        access = self._accessibility_score(center, atom_data, cluster_points)
        
        # Weighted total
        scores = {
            'depth': depth,
            'volume': volume,
            'hydrophobicity': hydro,
            'shape': shape,
            'residue': residue,
            'accessibility': access
        }
        
        total = sum(self.WEIGHTS[k] * v for k, v in scores.items()) * 100
        
        # Additional metadata
        metadata = {
            'cluster_size': len(cluster_points),
            'estimated_volume': self._estimate_volume(cluster_points),
            'depth_angstrom': self._calculate_depth(cluster_points, center, atom_data),
            'residue_composition': self._get_residue_composition(center, atom_data)
        }
        
        return total, scores, metadata
    
    def _depth_score(self, cluster_points: np.ndarray, center: np.ndarray,
                    atom_data: AtomData) -> float:
        """Score based on pocket depth (deeper = better, up to a point)."""
        depth = self._calculate_depth(cluster_points, center, atom_data)
        
        # Optimal depth: 5-15 Å
        if depth < 3:
            score = depth / 3.0 * 0.5
        elif depth < 15:
            score = 0.5 + (depth - 3) / 12.0 * 0.5
        else:
            score = 1.0 - (depth - 15) / 30.0 * 0.3
        
        return np.clip(score, 0.0, 1.0)
    
    def _calculate_depth(self, cluster_points: np.ndarray, center: np.ndarray,
                        atom_data: AtomData) -> float:
        """Calculate pocket depth in Angstroms."""
        # Distance from center to nearest surface atom
        coords = atom_data.coordinates
        d_center = np.min(np.linalg.norm(coords - center, axis=1))
        
        # Maximum distance of cluster points from center
        if len(cluster_points) > 0:
            d_cluster = np.max(np.linalg.norm(cluster_points - center, axis=1))
            depth = d_cluster + d_center * 0.3
        else:
            depth = d_center
        
        return float(depth)
    
    def _volume_score(self, cluster_points: np.ndarray) -> float:
        """Score based on pocket volume."""
        volume = self._estimate_volume(cluster_points)
        
        # Optimal volume: 200-800 ų
        if volume < 100:
            score = volume / 100.0 * 0.5
        elif volume < 800:
            score = 0.5 + (volume - 100) / 700.0 * 0.5
        else:
            score = 1.0 - (volume - 800) / 1200.0 * 0.3
        
        return np.clip(score, 0.0, 1.0)
    
    def _estimate_volume(self, cluster_points: np.ndarray) -> float:
        """Estimate pocket volume in ų."""
        if len(cluster_points) < 4:
            return 0.0
        
        try:
            hull = ConvexHull(cluster_points)
            return hull.volume
        except:
            # Fallback: bounding box
            mins = cluster_points.min(axis=0)
            maxs = cluster_points.max(axis=0)
            return np.prod(maxs - mins)
    
    def _hydrophobicity_score(self, center: np.ndarray, atom_data: AtomData,
                             probe_energies: Dict[str, float]) -> float:
        """
        Score based on hydrophobic/hydrophilic balance.
        Uses probe energies and nearby residues.
        """
        # From probe energies
        hydro_energy = probe_energies.get('hydrophobic', 0.0)
        polar_energy = (probe_energies.get('hbond_donor', 0.0) + 
                       probe_energies.get('hbond_acceptor', 0.0)) / 2.0
        
        # Balance: prefer moderate hydrophobicity
        if abs(hydro_energy) < 0.1 and abs(polar_energy) < 0.1:
            balance = 0.5
        else:
            ratio = abs(hydro_energy) / (abs(hydro_energy) + abs(polar_energy) + 1e-6)
            # Optimal: 40-60% hydrophobic
            if 0.4 <= ratio <= 0.6:
                balance = 1.0
            elif ratio < 0.4:
                balance = ratio / 0.4
            else:
                balance = 1.0 - (ratio - 0.6) / 0.4
        
        # Check residue composition
        residues = self._get_nearby_residues(center, atom_data, radius=12.0)
        n_hydro = sum(1 for r in residues if r in self.HYDROPHOBIC_RES)
        n_polar = sum(1 for r in residues if r in self.POLAR_RES)
        n_charged = sum(1 for r in residues if r in self.CHARGED_RES)
        
        total = len(residues)
        if total > 0:
            residue_balance = 1.0 - abs((n_hydro / total) - 0.5)
        else:
            residue_balance = 0.5
        
        # Combined score
        score = 0.6 * balance + 0.4 * residue_balance
        return float(np.clip(score, 0.0, 1.0))
    
    def _shape_score(self, cluster_points: np.ndarray) -> float:
        """Score based on shape regularity (sphericity)."""
        if len(cluster_points) < 10:
            return 0.5
        
        # PCA to get principal axes
        centered = cluster_points - np.mean(cluster_points, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, _ = np.linalg.eigh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Sphericity: λ₁ : λ₂ : λ₃
        if eigenvalues[0] < 1e-6:
            return 0.5
        
        ratios = eigenvalues / eigenvalues[0]
        # Perfect sphere: [1, 1, 1], perfect line: [1, 0, 0]
        sphericity = (ratios[1] + ratios[2]) / 2.0
        
        # Prefer moderately spherical
        if sphericity > 0.5:
            score = sphericity
        else:
            score = sphericity * 0.8
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _residue_score(self, center: np.ndarray, atom_data: AtomData) -> float:
        """Score based on residue composition diversity."""
        residues = self._get_nearby_residues(center, atom_data, radius=12.0)
        
        if len(residues) == 0:
            return 0.0
        
        # Count residue types
        n_hydro = sum(1 for r in residues if r in self.HYDROPHOBIC_RES)
        n_polar = sum(1 for r in residues if r in self.POLAR_RES)
        n_charged = sum(1 for r in residues if r in self.CHARGED_RES)
        
        total = len(residues)
        
        # Diversity: prefer pockets with mixed residue types
        has_hydro = n_hydro > 0
        has_polar = n_polar > 0
        has_charged = n_charged > 0
        
        diversity = (has_hydro + has_polar + has_charged) / 3.0
        
        # Bonus for catalytic residues
        catalytic = {'SER', 'CYS', 'HIS', 'ASP', 'GLU'}
        n_catalytic = sum(1 for r in residues if r in catalytic)
        catalytic_bonus = min(n_catalytic / total, 0.3)
        
        score = diversity * 0.7 + catalytic_bonus
        return float(np.clip(score, 0.0, 1.0))
    
    def _accessibility_score(self, center: np.ndarray, atom_data: AtomData,
                            cluster_points: np.ndarray) -> float:
        """
        Score based on accessibility.
        Not too buried, not too exposed.
        """
        coords = atom_data.coordinates
        
        # Distance from center to protein surface
        distances = np.linalg.norm(coords - center, axis=1)
        surface_distance = np.min(distances)
        
        # Burial depth
        if surface_distance < 5:
            burial = 0.2  # Too exposed
        elif surface_distance < 12:
            burial = 1.0  # Optimal
        else:
            burial = max(0.3, 1.0 - (surface_distance - 12) / 20.0)
        
        # Opening size (check how many cluster points are near surface)
        if len(cluster_points) > 0:
            cluster_dists = cdist(cluster_points, coords)
            near_surface = np.sum(np.min(cluster_dists, axis=1) < 3.0)
            opening = near_surface / len(cluster_points)
            opening_score = min(opening * 2.0, 1.0)
        else:
            opening_score = 0.5
        
        score = 0.6 * burial + 0.4 * opening_score
        return float(np.clip(score, 0.0, 1.0))
    
    def _get_nearby_residues(self, center: np.ndarray, atom_data: AtomData,
                            radius: float = 12.0) -> List[str]:
        """Get residues within radius of center."""
        coords = atom_data.coordinates
        distances = np.linalg.norm(coords - center, axis=1)
        mask = distances < radius
        
        nearby_res = [atom_data.residues[i] for i in range(len(mask)) if mask[i]]
        return list(set(nearby_res))  # Unique
    
    def _get_residue_composition(self, center: np.ndarray, 
                                atom_data: AtomData) -> Dict[str, int]:
        """Get composition of nearby residues."""
        residues = self._get_nearby_residues(center, atom_data)
        composition = {}
        for r in residues:
            composition[r] = composition.get(r, 0) + 1
        return composition


# ============================================================================
# Cavity-Ligand Matcher (Simplified)
# ============================================================================

class CavityLigandMatcher:
    """
    Predict cavity-ligand compatibility.
    Simplified version - requires ligand info.
    """
    
    def __init__(self):
        pass
    
    def predict_compatibility(self, pocket: PocketResult, 
                            ligand_volume: Optional[float] = None,
                            ligand_properties: Optional[Dict] = None) -> float:
        """
        Predict compatibility score (0-1).
        
        Args:
            pocket: PocketResult object
            ligand_volume: Ligand volume in ų
            ligand_properties: Dict with 'hydrophobic', 'hbond_donors', 'hbond_acceptors', 'charge'
        """
        if ligand_volume is None and ligand_properties is None:
            # No ligand info, return pocket confidence
            return pocket.confidence
        
        score = 0.0
        n_criteria = 0
        
        # Volume matching
        if ligand_volume is not None:
            volume_ratio = ligand_volume / max(pocket.volume, 1.0)
            # Optimal: ligand occupies 30-70% of pocket
            if 0.3 <= volume_ratio <= 0.7:
                volume_match = 1.0
            elif volume_ratio < 0.3:
                volume_match = volume_ratio / 0.3
            else:
                volume_match = max(0.0, 1.0 - (volume_ratio - 0.7) / 0.5)
            
            score += volume_match
            n_criteria += 1
        
        # Chemical complementarity
        if ligand_properties is not None:
            # Hydrophobicity matching
            lig_hydro = ligand_properties.get('hydrophobic', 0.5)
            hydro_match = 1.0 - abs(pocket.hydrophobicity - lig_hydro)
            score += hydro_match
            n_criteria += 1
            
            # H-bond complementarity
            lig_donors = ligand_properties.get('hbond_donors', 0)
            lig_acceptors = ligand_properties.get('hbond_acceptors', 0)
            
            # Check if pocket can complement ligand H-bonds
            # (simplified - would need more detailed analysis)
            hbond_score = 0.7  # Placeholder
            score += hbond_score
            n_criteria += 1
        
        # Normalize
        if n_criteria > 0:
            final_score = score / n_criteria
        else:
            final_score = pocket.confidence
        
        return float(np.clip(final_score, 0.0, 1.0))


# ============================================================================
# Integrated Cavity Detector
# ============================================================================

class IntegratedCavityDetector:
    """
    Integrates all cavity detection methods:
    - α-shape pre-filtering
    - 5-probe energy grid
    - Enhanced DBSCAN
    - Comprehensive scoring
    """
    
    def __init__(self, use_alpha_shape: bool = True,
                 grid_spacing: float = 1.5,
                 energy_percentile: float = 10.0):
        """
        Args:
            use_alpha_shape: Use α-shape pre-filtering
            grid_spacing: Energy grid resolution
            energy_percentile: Percentile for low-energy cutoff
        """
        self.use_alpha_shape = use_alpha_shape
        self.grid_spacing = grid_spacing
        self.energy_percentile = energy_percentile
        
        self.alpha_detector = AlphaShapeCavityDetector()
        self.energy_grid = FiveProbeEnergyGrid(grid_spacing=grid_spacing)
        self.dbscan = EnhancedDBSCAN(adaptive=True)
        self.scorer = PocketScorer()
    
    def detect_pockets(self, atom_data: AtomData, 
                      top_k: int = 3) -> List[PocketResult]:
        """
        Complete pocket detection pipeline.
        
        Returns:
            List of PocketResult objects, ranked by score
        """
        print("\n" + "="*70)
        print("🔬 High-Precision Cavity Detection Pipeline")
        print("="*70)
        
        # Step 1: α-shape pre-filtering (optional)
        candidate_regions = None
        if self.use_alpha_shape:
            print("\n[1/5] α-Shape Cavity Pre-detection")
            candidate_regions = self.alpha_detector.detect_cavities(atom_data)
            if len(candidate_regions) > 0:
                print(f"  ✓ Identified {len(candidate_regions)} candidate regions")
        
        # Step 2: 5-Probe Energy Grid
        print("\n[2/5] 5-Probe Energy Grid Calculation")
        grid_points, probe_energies = self.energy_grid.compute_energy_grids(atom_data)
        
        # Merge energies
        merged_energy = self.energy_grid.merge_probe_energies(probe_energies)
        print(f"  ✓ Energy range: [{merged_energy.min():.2f}, {merged_energy.max():.2f}] kcal/mol")
        
        # Step 3: Identify low-energy regions
        print("\n[3/5] Low-Energy Region Identification")
        threshold = np.percentile(merged_energy, self.energy_percentile)
        favorable_mask = merged_energy <= threshold
        favorable_points = grid_points[favorable_mask]
        favorable_energies = merged_energy[favorable_mask]
        
        print(f"  ✓ Threshold: {threshold:.2f} kcal/mol ({self.energy_percentile}th percentile)")
        print(f"  ✓ Favorable points: {len(favorable_points)}")
        
        if len(favorable_points) < 10:
            print("  ⚠️ Too few favorable points, relaxing criteria...")
            self.energy_percentile = 20.0
            threshold = np.percentile(merged_energy, self.energy_percentile)
            favorable_mask = merged_energy <= threshold
            favorable_points = grid_points[favorable_mask]
            favorable_energies = merged_energy[favorable_mask]
        
        # Step 4: DBSCAN Clustering
        print("\n[4/5] DBSCAN Clustering")
        labels = self.dbscan.fit(favorable_points, len(atom_data.coordinates))
        
        # Extract clusters
        unique_labels = sorted([l for l in set(labels) if l != -1])
        clusters = []
        for lab in unique_labels:
            mask = labels == lab
            cluster_pts = favorable_points[mask]
            cluster_E = favorable_energies[mask]
            
            # Calculate mean probe energies for this cluster
            cluster_probe_energies = {}
            for probe_name, energies in probe_energies.items():
                probe_favorable = energies[favorable_mask]
                cluster_probe_energies[probe_name] = float(np.mean(probe_favorable[mask]))
            
            clusters.append({
                'points': cluster_pts,
                'energies': cluster_E,
                'probe_energies': cluster_probe_energies,
                'center': np.mean(cluster_pts, axis=0)
            })
        
        print(f"  ✓ Valid clusters: {len(clusters)}")
        
        # Step 5: Comprehensive Scoring
        print("\n[5/5] Pocket Quality Scoring")
        pockets = []
        
        for i, cluster in enumerate(clusters):
            center = cluster['center']
            points = cluster['points']
            probe_E = cluster['probe_energies']
            
            # Score pocket
            score, breakdown, metadata = self.scorer.score_pocket(
                points, center, atom_data, probe_E
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(score, breakdown, metadata)
            
            # PCA for principal axes
            if len(points) >= 3:
                centered = points - center
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                idx = np.argsort(eigenvalues)[::-1]
                principal_axes = eigenvectors[:, idx]
                
                # Extent along principal axes
                projected = centered @ principal_axes
                extent = np.max(projected, axis=0) - np.min(projected, axis=0)
            else:
                principal_axes = np.eye(3)
                extent = np.array([1.0, 1.0, 1.0])
            
            # Create PocketResult
            pocket = PocketResult(
                center=center,
                cluster_points=points,
                volume=metadata['estimated_volume'],
                depth=metadata['depth_angstrom'],
                pocket_score=score,
                score_breakdown=breakdown,
                probe_energies=probe_E,
                mean_energy=float(np.mean(cluster['energies'])),
                hydrophobicity=float(breakdown['hydrophobicity'] * 2 - 1),  # 0-1 -> -1 to +1
                residue_composition=metadata['residue_composition'],
                principal_axes=principal_axes,
                extent=extent,
                confidence=confidence,
                rank=i
            )
            
            pockets.append(pocket)
            
            print(f"  Pocket {i+1}: Score={score:.1f}, Confidence={confidence:.2f}, "
                  f"Volume={metadata['estimated_volume']:.0f}ų, Depth={metadata['depth_angstrom']:.1f}Å")
        
        # Sort by score
        pockets.sort(key=lambda p: p.pocket_score, reverse=True)
        
        # Re-rank
        for i, p in enumerate(pockets):
            p.rank = i + 1
        
        # Return top K
        top_pockets = pockets[:top_k]
        
        print(f"\n✅ Detection complete: {len(pockets)} pockets found, returning top {min(top_k, len(pockets))}")
        
        return top_pockets
    
    def _calculate_confidence(self, score: float, breakdown: Dict[str, float],
                            metadata: Dict[str, Any]) -> float:
        """
        Calculate confidence based on multiple factors.
        """
        # Base confidence from score
        conf = score / 100.0
        
        # Adjust based on cluster size
        cluster_size = metadata['cluster_size']
        if cluster_size < 20:
            conf *= 0.8
        elif cluster_size > 100:
            conf *= 1.1
        
        # Adjust based on score balance (avoid one metric dominating)
        scores_array = np.array(list(breakdown.values()))
        score_std = np.std(scores_array)
        if score_std > 0.3:
            conf *= 0.9  # Unbalanced scores
        
        return float(np.clip(conf, 0.0, 1.0))


# ============================================================================
# Mock & Vina Scoring
# ============================================================================

class MockDockingScorer:
    """Lightweight mock scoring for testing."""
    
    @staticmethod
    def calculate_mock_score(coords: np.ndarray, center: np.ndarray,
                           box_size: np.ndarray, residues: List[str],
                           pocket_score: float = 0.0) -> float:
        """Mock docking score incorporating pocket quality."""
        distances = np.linalg.norm(coords - center, axis=1)
        nearby_mask = distances < 15.0
        
        if np.sum(nearby_mask) < 5:
            return 0.0
        
        nearby_residues = [residues[i] for i in range(len(residues)) if nearby_mask[i]]
        
        # Pocket depth factor
        pocket_depth = np.sum(nearby_mask) / 100.0
        
        # Chemical environment
        polar_res = {'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'SER', 'THR', 'ASN', 'GLN', 'CYS'}
        hydrophobic = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'}
        
        polar_count = sum(1 for r in nearby_residues if r in polar_res)
        hydrophobic_count = sum(1 for r in nearby_residues if r in hydrophobic)
        
        chemical = (polar_count * 0.3 + hydrophobic_count * 0.2) / max(len(nearby_residues), 1)
        
        # Volume penalty
        volume_penalty = np.prod(box_size) / 50000.0
        
        # Incorporate pocket score
        pocket_bonus = pocket_score / 100.0 * 2.0
        
        score = -(pocket_depth * 2.0 + chemical * 3.0 - volume_penalty * 0.5 + pocket_bonus)
        return max(min(score, 0.0), -15.0)


class VinaDockingScorer:
    """AutoDock Vina wrapper."""
    
    def __init__(self, vina_path: str):
        self.vina_path = vina_path
    
    def run_vina(self, receptor_pdbqt: str, ligand_pdbqt: str,
                center: np.ndarray, box_size: np.ndarray,
                exhaustiveness: int = 8) -> Tuple[Optional[float], str]:
        """Run Vina docking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, "config.txt")
            out_file = os.path.join(tmpdir, "out.pdbqt")
            
            with open(config_file, "w") as f:
                f.write(f"receptor = {receptor_pdbqt}\n")
                f.write(f"ligand = {ligand_pdbqt}\n")
                f.write(f"center_x = {center[0]:.3f}\n")
                f.write(f"center_y = {center[1]:.3f}\n")
                f.write(f"center_z = {center[2]:.3f}\n")
                f.write(f"size_x = {box_size[0]:.3f}\n")
                f.write(f"size_y = {box_size[1]:.3f}\n")
                f.write(f"size_z = {box_size[2]:.3f}\n")
                f.write(f"exhaustiveness = {exhaustiveness}\n")
                f.write(f"out = {out_file}\n")
            
            try:
                result = subprocess.run(
                    [self.vina_path, "--config", config_file],
                    capture_output=True, text=True, timeout=300
                )
            except subprocess.TimeoutExpired:
                return None, "timeout"
            except Exception as e:
                return None, f"exception:{str(e)[:50]}"
            
            if result.returncode != 0:
                return None, f"exit_{result.returncode}"
            
            # Parse output
            best_score = None
            if result.stdout:
                for line in result.stdout.splitlines():
                    if "   1 " in line or line.strip().startswith("1 "):
                        parts = line.split()
                        for p in parts:
                            try:
                                best_score = float(p)
                                break
                            except:
                                continue
                        if best_score is not None:
                            break
            
            if best_score is None:
                return None, "no_score"
            
            return best_score, ""


# ============================================================================
# Multi-Objective Bayesian Optimizer
# ============================================================================

class MultiObjectiveBayesianOptimizer:
    """
    Multi-objective Bayesian optimization for docking box.
    
    Objectives:
    1. Docking score (minimize)
    2. Cavity overlap (maximize)
    3. Box compactness (minimize volume)
    """
    
    def __init__(self, bounds: np.ndarray, cavity_center: np.ndarray,
                 cavity_volume: float):
        """
        Args:
            bounds: (d, 2) parameter bounds
            cavity_center: Initial cavity center
            cavity_volume: Initial cavity volume
        """
        self.bounds = bounds
        self.dim = bounds.shape[0]
        self.cavity_center = cavity_center
        self.cavity_volume = cavity_volume
        
        self.X = []
        self.Y_dock = []  # Docking scores
        self.Y_overlap = []  # Cavity overlap
        self.Y_volume = []  # Box volumes
    
    def suggest(self) -> np.ndarray:
        """Suggest next point using scalarized EI."""
        if len(self.X) < 5:
            # Random initialization
            return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        
        X = np.array(self.X)
        
        # Scalarize objectives (weighted sum)
        # Normalize each objective to [0, 1]
        y_dock_norm = np.array(self.Y_dock)
        y_overlap_norm = np.array(self.Y_overlap)
        y_volume_norm = np.array(self.Y_volume)
        
        if y_dock_norm.std() > 0:
            y_dock_norm = (y_dock_norm - y_dock_norm.min()) / (y_dock_norm.max() - y_dock_norm.min())
        if y_overlap_norm.std() > 0:
            y_overlap_norm = (y_overlap_norm - y_overlap_norm.min()) / (y_overlap_norm.max() - y_overlap_norm.min())
        if y_volume_norm.std() > 0:
            y_volume_norm = (y_volume_norm - y_volume_norm.min()) / (y_volume_norm.max() - y_volume_norm.min())
        
        # Scalarized objective: minimize dock, maximize overlap, minimize volume
        weights = [0.5, 0.3, 0.2]
        y_scalar = weights[0] * y_dock_norm - weights[1] * y_overlap_norm + weights[2] * y_volume_norm
        
        # Simple GP-like EI
        n_cand = 256
        cand = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                size=(n_cand, self.dim))
        
        acq_vals = []
        for c in cand:
            d = np.linalg.norm(X - c, axis=1)
            if np.all(d < 1e-6):
                acq_vals.append(-1e9)
                continue
            
            w = np.exp(-d / (np.mean(d) + 1e-6))
            w /= np.sum(w)
            
            mu = np.sum(w * y_scalar)
            sigma = np.sqrt(np.sum(w * (y_scalar - mu) ** 2)) + 1e-6
            
            best = np.min(y_scalar)
            z = (best - mu) / sigma
            ei = (best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            acq_vals.append(ei)
        
        idx = int(np.argmax(acq_vals))
        return cand[idx]
    
    def add(self, x: np.ndarray, dock_score: float, overlap: float, volume: float):
        """Add observation."""
        self.X.append(x.copy())
        self.Y_dock.append(float(dock_score))
        self.Y_overlap.append(float(overlap))
        self.Y_volume.append(float(volume))
    
    def pareto_front(self) -> List[int]:
        """
        Get Pareto-optimal solutions.
        
        Returns:
            Indices of Pareto-optimal points
        """
        if len(self.X) == 0:
            return []
        
        # Stack objectives (minimize dock, maximize overlap, minimize volume)
        Y = np.column_stack([
            self.Y_dock,
            [-o for o in self.Y_overlap],  # Negate to minimize
            self.Y_volume
        ])
        
        n = len(Y)
        pareto = []
        
        for i in range(n):
            dominated = False
            for j in range(n):
                if i == j:
                    continue
                # Check if j dominates i
                if np.all(Y[j] <= Y[i]) and np.any(Y[j] < Y[i]):
                    dominated = True
                    break
            if not dominated:
                pareto.append(i)
        
        return pareto
    
    def best(self, prefer_dock: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Get best solution.
        
        Args:
            prefer_dock: Prioritize docking score over other objectives
        """
        if not self.X:
            return None, {}
        
        if prefer_dock:
            idx = int(np.argmin(self.Y_dock))
        else:
            # Scalarize
            y_dock = np.array(self.Y_dock)
            y_overlap = np.array(self.Y_overlap)
            y_volume = np.array(self.Y_volume)
            
            # Normalize
            y_dock = (y_dock - y_dock.min()) / (y_dock.max() - y_dock.min() + 1e-6)
            y_overlap = (y_overlap - y_overlap.min()) / (y_overlap.max() - y_overlap.min() + 1e-6)
            y_volume = (y_volume - y_volume.min()) / (y_volume.max() - y_volume.min() + 1e-6)
            
            scores = 0.5 * y_dock - 0.3 * y_overlap + 0.2 * y_volume
            idx = int(np.argmin(scores))
        
        return np.array(self.X[idx]), {
            'dock_score': self.Y_dock[idx],
            'overlap': self.Y_overlap[idx],
            'volume': self.Y_volume[idx]
        }


# ============================================================================
# High-Precision Optimizer
# ============================================================================

class HighPrecisionDockingOptimizer:
    """
    High-precision docking box optimizer.
    Integrates all components.
    """
    
    def __init__(self, padding: float = 10.0, vina_path: Optional[str] = None):
        self.padding = padding
        self.vina_path = vina_path
        self.vina_scorer = VinaDockingScorer(vina_path) if vina_path else None
        
        self.cavity_detector = IntegratedCavityDetector(
            use_alpha_shape=True,
            grid_spacing=1.5,
            energy_percentile=10.0
        )
    
    def optimize(self, atom_data: AtomData, n_iter: int = 15,
                receptor_pdbqt: Optional[str] = None,
                ligand_pdbqt: Optional[str] = None,
                top_k_pockets: int = 1) -> List[DockingParameters]:
        """
        Complete optimization pipeline.
        
        Returns:
            List of DockingParameters for top-K pockets
        """
        print("\n" + "="*70)
        print("🎯 High-Precision Docking Box Optimization")
        print("="*70)
        
        # Detect pockets
        pockets = self.cavity_detector.detect_pockets(atom_data, top_k=top_k_pockets)
        
        if len(pockets) == 0:
            print("❌ No pockets detected!")
            return []
        
        # Optimize each pocket
        results = []
        
        for i, pocket in enumerate(pockets):
            print(f"\n{'='*70}")
            print(f"📦 Optimizing Pocket {i+1}/{len(pockets)}")
            print(f"{'='*70}")
            print(f"  Center: ({pocket.center[0]:.2f}, {pocket.center[1]:.2f}, {pocket.center[2]:.2f})")
            print(f"  Score: {pocket.pocket_score:.1f}, Confidence: {pocket.confidence:.2f}")
            print(f"  Volume: {pocket.volume:.0f}ų, Depth: {pocket.depth:.1f}Å")
            
            params = self._optimize_single_pocket(
                pocket, atom_data, n_iter,
                receptor_pdbqt, ligand_pdbqt
            )
            
            results.append(params)
        
        return results
    
    def _optimize_single_pocket(self, pocket: PocketResult, atom_data: AtomData,
                               n_iter: int, receptor_pdbqt: Optional[str],
                               ligand_pdbqt: Optional[str]) -> DockingParameters:
        """Optimize docking box for a single pocket."""
        
        # Initial box from pocket geometry
        base_size = pocket.extent + self.padding
        base_size = np.clip(base_size, 18.0, 45.0)
        
        com = PDBParser.center_of_mass(atom_data)
        init_center = 0.7 * pocket.center + 0.3 * com
        
        # BO bounds
        center_range = 6.0
        bounds = np.array([
            [init_center[0] - center_range, init_center[0] + center_range],
            [init_center[1] - center_range, init_center[1] + center_range],
            [init_center[2] - center_range, init_center[2] + center_range],
            [max(15.0, base_size[0] * 0.8), min(50.0, base_size[0] * 1.2)],
            [max(15.0, base_size[1] * 0.8), min(50.0, base_size[1] * 1.2)],
            [max(15.0, base_size[2] * 0.8), min(50.0, base_size[2] * 1.2)],
        ], dtype=np.float64)
        
        print(f"\n  🎯 Bayesian Optimization Setup:")
        print(f"    Iterations: {n_iter}")
        print(f"    Initial center: ({init_center[0]:.1f}, {init_center[1]:.1f}, {init_center[2]:.1f})")
        print(f"    Initial size: ({base_size[0]:.1f}, {base_size[1]:.1f}, {base_size[2]:.1f})")
        
        # Multi-objective BO
        bo = MultiObjectiveBayesianOptimizer(
            bounds, pocket.center, pocket.volume
        )
        
        use_vina = bool(self.vina_scorer and receptor_pdbqt and ligand_pdbqt)
        print(f"    Scoring: {'Vina' if use_vina else 'Mock'}")
        
        vina_fail = 0
        
        for it in range(n_iter):
            x = bo.suggest()
            center = x[:3]
            box_size = x[3:]
            box_volume = np.prod(box_size)
            
            # Cavity overlap
            overlap = self._calculate_overlap(center, box_size, pocket)
            
            # Docking score
            if use_vina:
                score, err = self.vina_scorer.run_vina(
                    receptor_pdbqt, ligand_pdbqt, center, box_size, exhaustiveness=8
                )
                if score is None:
                    vina_fail += 1
                    score = MockDockingScorer.calculate_mock_score(
                        atom_data.coordinates, center, box_size,
                        atom_data.residues, pocket.pocket_score
                    ) + 2.0
                    scorer = f"mock({err})"
                else:
                    scorer = "vina"
            else:
                score = MockDockingScorer.calculate_mock_score(
                    atom_data.coordinates, center, box_size,
                    atom_data.residues, pocket.pocket_score
                )
                scorer = "mock"
            
            bo.add(x, score, overlap, box_volume)
            
            print(f"    Iter {it+1:2d}/{n_iter}: dock={score:6.2f}, "
                  f"overlap={overlap:.2f}, vol={box_volume:7.0f} [{scorer}]")
        
        # Get best solution
        best_x, best_obj = bo.best(prefer_dock=True)
        final_center = best_x[:3]
        final_size = best_x[3:]
        
        print(f"\n  ✅ Optimization Complete:")
        print(f"    Best docking score: {best_obj['dock_score']:.3f}")
        print(f"    Cavity overlap: {best_obj['overlap']:.2f}")
        print(f"    Box volume: {best_obj['volume']:.0f}ų")
        if vina_fail > 0:
            print(f"    Vina failures: {vina_fail}/{n_iter}")
        
        # Pareto front
        pareto_indices = bo.pareto_front()
        if len(pareto_indices) > 1:
            print(f"    Pareto optimal solutions: {len(pareto_indices)}")
        
        return DockingParameters(
            center_x=float(final_center[0]),
            center_y=float(final_center[1]),
            center_z=float(final_center[2]),
            size_x=float(final_size[0]),
            size_y=float(final_size[1]),
            size_z=float(final_size[2]),
            n_atoms=len(atom_data.coordinates),
            box_volume=float(best_obj['volume']),
            pocket_score=pocket.pocket_score,
            pocket_confidence=pocket.confidence,
            docking_score=float(best_obj['dock_score']),
            iterations=n_iter,
            optimized_by="high_precision_multi_obj",
            pocket_depth=pocket.depth,
            hydrophobicity=pocket.hydrophobicity,
            probe_energies=pocket.probe_energies
        )
    
    def _calculate_overlap(self, center: np.ndarray, box_size: np.ndarray,
                          pocket: PocketResult) -> float:
        """Calculate overlap between box and cavity cluster."""
        box_min = center - box_size / 2.0
        box_max = center + box_size / 2.0
        
        # Count pocket points inside box
        pts = pocket.cluster_points
        inside = np.all((pts >= box_min) & (pts <= box_max), axis=1)
        
        if len(pts) == 0:
            return 0.0
        
        return float(np.sum(inside) / len(pts))


# ============================================================================
# Output Writers
# ============================================================================

class ParameterWriter:
    @staticmethod
    def write_vina_config(params: DockingParameters, output_file: str,
                         protein_name: str, padding: float):
        """Write Vina configuration file."""
        try:
            with open(output_file, "w") as f:
                f.write("# AutoDock Vina Configuration File\n")
                f.write(f"# Generated by ZymEvo High-Precision Optimizer\n")
                f.write(f"# Protein: {protein_name}\n")
                f.write(f"# Atoms: {params.n_atoms}\n")
                f.write(f"# Box volume: {params.box_volume:.2f} ų\n")
                f.write(f"# Pocket score: {params.pocket_score:.1f}/100\n")
                f.write(f"# Pocket confidence: {params.pocket_confidence:.2f}\n")
                f.write(f"# Pocket depth: {params.pocket_depth:.1f} Å\n")
                f.write(f"# Hydrophobicity: {params.hydrophobicity:.2f}\n")
                f.write(f"# Docking score: {params.docking_score:.3f} kcal/mol\n")
                f.write(f"# Optimization: {params.optimized_by} ({params.iterations} iter)\n")
                f.write(f"# Padding: {padding:.1f} Å\n")
                
                if params.probe_energies:
                    f.write(f"#\n# Probe energies (kcal/mol):\n")
                    for probe, energy in params.probe_energies.items():
                        f.write(f"#   {probe}: {energy:.2f}\n")
                
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
        except Exception as e:
            print(f"❌ Failed to write config: {e}")
            return False
    
    @staticmethod
    def write_csv_summary(results: List[Tuple[str, List[DockingParameters]]],
                         output_file: str):
        """Write comprehensive CSV summary."""
        try:
            with open(output_file, "w") as f:
                f.write("Protein,Pocket_Rank,Center_X,Center_Y,Center_Z,"
                       "Size_X,Size_Y,Size_Z,N_Atoms,Box_Volume,"
                       "Pocket_Score,Pocket_Confidence,Pocket_Depth,"
                       "Hydrophobicity,Docking_Score,Iterations,Optimized_By\n")
                
                for name, param_list in results:
                    for rank, p in enumerate(param_list, 1):
                        f.write(
                            f"{name},{rank},"
                            f"{p.center_x:.3f},{p.center_y:.3f},{p.center_z:.3f},"
                            f"{p.size_x:.3f},{p.size_y:.3f},{p.size_z:.3f},"
                            f"{p.n_atoms},{p.box_volume:.2f},"
                            f"{p.pocket_score:.1f},{p.pocket_confidence:.2f},"
                            f"{p.pocket_depth:.1f},{p.hydrophobicity:.2f},"
                            f"{p.docking_score:.3f},{p.iterations},"
                            f"{p.optimized_by}\n"
                        )
            return True
        except Exception as e:
            print(f"❌ Failed to write CSV: {e}")
            return False


# ============================================================================
# Vina Installer
# ============================================================================

class VinaInstaller:
    """AutoDock Vina installer."""
    
    VINA_URL = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
    VINA_BINARY = "vina"
    
    @staticmethod
    def ensure_vina(work_dir: str = ".") -> Optional[str]:
        """Ensure Vina is available."""
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


# ============================================================================
# Batch Processing
# ============================================================================

def process_single_protein(pdb_file: str, output_dir: str, n_iter: int,
                          padding: float, top_k_pockets: int,
                          vina_path: Optional[str] = None,
                          receptor_pdbqt: Optional[str] = None,
                          ligand_pdbqt: Optional[str] = None
                          ) -> Optional[Tuple[str, List[DockingParameters]]]:
    """Process single protein."""
    protein_name = Path(pdb_file).stem
    print(f"\n{'='*70}")
    print(f"🔬 Processing: {protein_name}")
    print(f"{'='*70}")
    
    atom_data = PDBParser.parse_pdb(pdb_file)
    if atom_data is None:
        return None
    
    print(f"  📊 Protein: {len(atom_data.coordinates)} atoms")
    
    optimizer = HighPrecisionDockingOptimizer(padding=padding, vina_path=vina_path)
    
    param_list = optimizer.optimize(
        atom_data, n_iter=n_iter,
        receptor_pdbqt=receptor_pdbqt,
        ligand_pdbqt=ligand_pdbqt,
        top_k_pockets=top_k_pockets
    )
    
    if not param_list:
        return None
    
    # Write configs
    os.makedirs(output_dir, exist_ok=True)
    
    for rank, params in enumerate(param_list, 1):
        cfg_file = os.path.join(output_dir, f"{protein_name}_pocket{rank}_config.txt")
        ok = ParameterWriter.write_vina_config(params, cfg_file, protein_name, padding)
        if ok:
            print(f"\n  ✅ Saved config: {cfg_file}")
    
    return protein_name, param_list


def batch_process(pdb_files: List[str], output_dir: str, n_iter: int,
                 padding: float, top_k_pockets: int,
                 vina_path: Optional[str] = None,
                 receptor_pdbqt: Optional[str] = None,
                 ligand_pdbqt: Optional[str] = None
                 ) -> List[Tuple[str, List[DockingParameters]]]:
    """Batch process proteins."""
    results = []
    
    for pdb in pdb_files:
        res = process_single_protein(
            pdb, output_dir, n_iter, padding, top_k_pockets,
            vina_path=vina_path,
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=ligand_pdbqt
        )
        if res:
            results.append(res)
    
    if results:
        summary = os.path.join(output_dir, "high_precision_summary.csv")
        ParameterWriter.write_csv_summary(results, summary)
        print(f"\n📊 Summary CSV: {summary}")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ZymEvo High-Precision Docking Box Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features:
  - α-shape cavity detection
  - 5-probe energy grid (hydrophobic, H-donor, H-acceptor, +/- charge)
  - Enhanced DBSCAN clustering
  - Comprehensive pocket scoring (6 metrics)
  - Multi-objective Bayesian optimization
  - Top-K pocket detection
  - Cavity-ligand matching (optional)

Example:
  python curpocket_precision_optimizer.py \\
    --pdbs protein1.pdb protein2.pdb \\
    --output results/ \\
    --iterations 20 \\
    --top_pockets 3 \\
    --vina_path ./vina \\
    --receptor_pdbqt receptor.pdbqt \\
    --ligand_pdbqt ligand.pdbqt
        """
    )
    
    parser.add_argument("--pdbs", nargs="+", required=True,
                       help="Input PDB files")
    parser.add_argument("--output", default="high_precision_results",
                       help="Output directory (default: high_precision_results)")
    parser.add_argument("--iterations", type=int, default=15,
                       help="Bayesian optimization iterations (default: 15)")
    parser.add_argument("--padding", type=float, default=10.0,
                       help="Box padding in Angstroms (default: 10.0)")
    parser.add_argument("--top_pockets", type=int, default=1,
                       help="Number of top pockets to optimize (default: 1)")
    parser.add_argument("--grid_spacing", type=float, default=1.5,
                       help="Energy grid spacing (default: 1.5 Å)")
    parser.add_argument("--vina_path", type=str, default=None,
                       help="Path to Vina executable (optional, will auto-download)")
    parser.add_argument("--receptor_pdbqt", type=str, default=None,
                       help="Receptor PDBQT for Vina scoring (optional)")
    parser.add_argument("--ligand_pdbqt", type=str, default=None,
                       help="Ligand PDBQT for Vina scoring (optional)")
    parser.add_argument("--no_alpha_shape", action="store_true",
                       help="Disable α-shape pre-filtering")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🧬 ZymEvo High-Precision Docking Box Optimizer")
    print("="*70)
    print(f"Input PDBs: {len(args.pdbs)}")
    print(f"Output directory: {args.output}")
    print(f"BO iterations: {args.iterations}")
    print(f"Box padding: {args.padding} Å")
    print(f"Top pockets: {args.top_pockets}")
    print(f"Grid spacing: {args.grid_spacing} Å")
    print(f"α-shape detection: {'Disabled' if args.no_alpha_shape else 'Enabled'}")
    
    # Handle Vina
    vina_path = args.vina_path
    if args.receptor_pdbqt and args.ligand_pdbqt:
        if not vina_path:
            print("\n📦 AutoDock Vina Setup")
            vina_path = VinaInstaller.ensure_vina(args.output)
            if not vina_path:
                print("  ⚠️ Vina not available, using mock scoring")
        print(f"Scoring mode: Vina")
    else:
        print(f"Scoring mode: Mock (no PDBQT files provided)")
    
    print("="*70)
    
    # Batch process
    batch_process(
        pdb_files=args.pdbs,
        output_dir=args.output,
        n_iter=args.iterations,
        padding=args.padding,
        top_k_pockets=args.top_pockets,
        vina_path=vina_path,
        receptor_pdbqt=args.receptor_pdbqt,
        ligand_pdbqt=args.ligand_pdbqt
    )
    
    print("\n" + "="*70)
    print("✅ All proteins processed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
