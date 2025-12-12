#!/usr/bin/env python3
"""
Automatically reduces ligand rotatable bonds (TORSDOF) while preserving chemical validity
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class AtomInfo:
    index: int
    name: str
    atom_type: str  # PDBQT atom type (A, C, N, NA, etc.)
    coords: np.ndarray
    residue: str
    
    
@dataclass
class BranchInfo:
    branch_id: int
    axis_atoms: Tuple[int, int]  # Atom indices defining rotation axis
    axis_atom_types: Tuple[str, str]
    moving_atoms: Set[int]  # Atoms affected by this rotation
    n_affected_atoms: int
    is_terminal: bool
    depth: int  # Nesting level in BRANCH tree
    priority_score: float = 0.0
    freeze_reason: str = ""
    distance_to_pocket: float = 999.0
    

@dataclass
class RigidRegion:
    region_type: str  # 'aromatic', 'amide', 'small_ring', 'conjugated'
    atoms: Set[int]
    bonds: Set[Tuple[int, int]]
    reason: str


@dataclass
class OptimizationReport:
    original_torsdof: int
    optimized_torsdof: int
    chemical_constraints: Dict[str, int]
    flexible_optimization: Dict
    warnings: List[str]
    recommendations: List[str]
    frozen_branches: List[Dict]
    kept_branches: List[Dict]

# RigidStructureDetector - Detect chemically rigid structures

class RigidStructureDetector:
    """
    Detects chemically rigid structures that should NOT be rotated:
    - Aromatic rings (benzene, pyridine, etc.)
    - Amide bonds (peptide bonds)
    - Small rings (3-4 membered)
    - Conjugated systems
    """
    
    # PDBQT aromatic atom types
    AROMATIC_TYPES = {'A', 'NA', 'NS', 'SA', 'OA'}
    
    def __init__(self):
        self.atoms: List[AtomInfo] = []
        self.bonds: Set[Tuple[int, int]] = set()
        self.rigid_regions: List[RigidRegion] = []
        
    def detect_from_pdbqt(self, pdbqt_file: str) -> Dict:
        """
        Main detection pipeline
        
        Returns:
            {
                'aromatic_atoms': Set[int],
                'rigid_rings': List[Set[int]],
                'amide_bonds': Set[Tuple[int, int]],
                'protected_bonds': Set[Tuple[int, int]],
                'rigid_regions': List[RigidRegion]
            }
        """
        # Parse PDBQT file
        self.atoms = self._parse_atoms(pdbqt_file)
        self.bonds = self._build_bond_graph()
        
        result = {
            'aromatic_atoms': set(),
            'rigid_rings': [],
            'amide_bonds': set(),
            'protected_bonds': set(),
            'rigid_regions': []
        }
        
        # 1. Detect aromatic atoms by atom type
        aromatic = self._detect_aromatic_atoms()
        result['aromatic_atoms'] = aromatic
        
        # 2. Detect ring structures
        rings = self._detect_rings()
        
        # 3. Classify rings as rigid or flexible
        for ring in rings:
            if self._is_rigid_ring(ring, aromatic):
                result['rigid_rings'].append(ring)
                
                # Add internal bonds to protected set
                for i, atom1 in enumerate(ring):
                    for atom2 in list(ring)[i+1:]:
                        bond = tuple(sorted([atom1, atom2]))
                        if bond in self.bonds:
                            result['protected_bonds'].add(bond)
                
                # Create rigid region
                region = RigidRegion(
                    region_type='aromatic' if ring.issubset(aromatic) else 'small_ring',
                    atoms=ring,
                    bonds={b for b in self.bonds if b[0] in ring and b[1] in ring},
                    reason='Aromatic ring' if ring.issubset(aromatic) else 'Small ring'
                )
                result['rigid_regions'].append(region)
        
        # 4. Detect amide bonds
        amide_bonds = self._detect_amide_bonds()
        result['amide_bonds'] = amide_bonds
        result['protected_bonds'].update(amide_bonds)
        
        # Create amide rigid regions
        for bond in amide_bonds:
            region = RigidRegion(
                region_type='amide',
                atoms=set(bond),
                bonds={bond},
                reason='Amide/peptide bond planarity'
            )
            result['rigid_regions'].append(region)
        
        self.rigid_regions = result['rigid_regions']
        
        return result
    
    def _parse_atoms(self, pdbqt_file: str) -> List[AtomInfo]:
        """Parse atoms from PDBQT file"""
        atoms = []
        atom_index = 0
        
        with open(pdbqt_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    try:
                        atom_name = line[12:16].strip()
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords = np.array([x, y, z])
                        
                        # PDBQT atom type at columns 77-78
                        atom_type = line[77:79].strip() if len(line) >= 79 else 'C'
                        
                        residue = line[17:20].strip()
                        
                        atom = AtomInfo(
                            index=atom_index,
                            name=atom_name,
                            atom_type=atom_type,
                            coords=coords,
                            residue=residue
                        )
                        atoms.append(atom)
                        atom_index += 1
                        
                    except (ValueError, IndexError):
                        continue
        
        return atoms
    
    def _build_bond_graph(self) -> Set[Tuple[int, int]]:
        """
        Build bond connectivity from distance
        Standard covalent radii: C(1.7Å), N(1.55Å), O(1.52Å), S(1.8Å)
        Bond = distance < sum_of_radii + 0.4Å tolerance
        """
        bonds = set()
        
        COVALENT_RADII = {
            'C': 1.70, 'A': 1.70,  # Aromatic C same as aliphatic
            'N': 1.55, 'NA': 1.55, 'NS': 1.55,
            'O': 1.52, 'OA': 1.52, 'OS': 1.52,
            'S': 1.80, 'SA': 1.80,
            'P': 1.80,
            'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 2.00,
            'H': 1.20, 'HD': 1.20
        }
        
        for i, atom1 in enumerate(self.atoms):
            for atom2 in self.atoms[i+1:]:
                dist = np.linalg.norm(atom1.coords - atom2.coords)
                
                r1 = COVALENT_RADII.get(atom1.atom_type, 1.7)
                r2 = COVALENT_RADII.get(atom2.atom_type, 1.7)
                
                threshold = r1 + r2 + 0.4
                
                if dist < threshold:
                    bond = tuple(sorted([atom1.index, atom2.index]))
                    bonds.add(bond)
        
        return bonds
    
    def _detect_aromatic_atoms(self) -> Set[int]:
        """Detect aromatic atoms by PDBQT atom type"""
        aromatic = set()
        
        for atom in self.atoms:
            if atom.atom_type in self.AROMATIC_TYPES:
                aromatic.add(atom.index)
        
        return aromatic
    
    def _detect_rings(self) -> List[Set[int]]:
        """
        Detect all ring structures using DFS
        Returns list of sets, each set contains atom indices in a ring
        """
        # Build adjacency list
        graph = defaultdict(list)
        for a1, a2 in self.bonds:
            graph[a1].append(a2)
            graph[a2].append(a1)
        
        rings = []
        visited_global = set()
        
        def dfs_find_rings(start):
            """Find all rings containing start node"""
            stack = [(start, [start], {start})]
            
            while stack:
                node, path, visited = stack.pop()
                
                for neighbor in graph[node]:
                    if neighbor == path[-2] if len(path) > 1 else -1:
                        continue  # Don't go back to parent
                    
                    if neighbor in visited:
                        # Found a ring
                        ring_start = path.index(neighbor)
                        ring = set(path[ring_start:])
                        
                        if 3 <= len(ring) <= 20:  # Valid ring size
                            # Check if this ring already exists
                            if not any(ring == existing for existing in rings):
                                rings.append(ring)
                    else:
                        stack.append((neighbor, path + [neighbor], visited | {neighbor}))
        
        # Try each atom as potential ring member
        for atom in self.atoms:
            if atom.index not in visited_global:
                dfs_find_rings(atom.index)
                visited_global.add(atom.index)
        
        return rings
    
    def _is_rigid_ring(self, ring: Set[int], aromatic_atoms: Set[int]) -> bool:
        """
        Determine if a ring is rigid
        
        Criteria:
        1. All atoms are aromatic (aromatic ring)
        2. Ring size <= 4 (small ring, high strain)
        3. Fused ring (shares edge with another ring) - TODO
        """
        # 1. Aromatic ring
        if ring.issubset(aromatic_atoms):
            return True
        
        # 2. Small ring (3-4 membered)
        if len(ring) <= 4:
            return True
        
        # 3. TODO: Detect fused rings
        
        return False
    
    def _detect_amide_bonds(self) -> Set[Tuple[int, int]]:
        """
        Detect amide bonds: C(=O)-N pattern
        
        Steps:
        1. Find all C-N bonds
        2. Check if C has double-bonded O
        3. Verify planarity (optional, approximated by geometry)
        """
        amide_bonds = set()
        
        # Build atom lookup
        atom_dict = {atom.index: atom for atom in self.atoms}
        
        for bond in self.bonds:
            a1_idx, a2_idx = bond
            a1 = atom_dict[a1_idx]
            a2 = atom_dict[a2_idx]
            
            # Check C-N bond (either direction)
            c_atom, n_atom = None, None
            
            if a1.atom_type == 'C' and a2.atom_type == 'N':
                c_atom, n_atom = a1, a2
            elif a1.atom_type == 'N' and a2.atom_type == 'C':
                c_atom, n_atom = a2, a1
            else:
                continue
            
            # Check if C has bonded O
            c_neighbors = [atom_dict[nb] for a, nb in self.bonds 
                          if a == c_atom.index or nb == c_atom.index]
            
            has_carbonyl_o = any(
                nb.atom_type in ['O', 'OA'] and 
                np.linalg.norm(c_atom.coords - nb.coords) < 1.35  # C=O typical ~1.23Å
                for nb in c_neighbors
            )
            
            if has_carbonyl_o:
                amide_bonds.add(tuple(sorted([c_atom.index, n_atom.index])))
        
        return amide_bonds


# ============================================================================
# BranchAnalyzer - Parse and analyze BRANCH records in PDBQT
# ============================================================================

class BranchAnalyzer:
    """
    Parse BRANCH structure from PDBQT file
    Extract rotation axis, affected atoms, and topology
    """
    
    def __init__(self):
        self.branches: List[BranchInfo] = []
        self.atoms: List[AtomInfo] = []
        
    def analyze_pdbqt(self, pdbqt_file: str) -> Dict:
        """
        Main analysis pipeline
        
        Returns:
            {
                'torsdof': int,
                'n_branches': int,
                'branches': List[BranchInfo],
                'atoms': List[AtomInfo],
                'needs_optimization': bool
            }
        """
        # Parse atoms
        detector = RigidStructureDetector()
        self.atoms = detector._parse_atoms(pdbqt_file)
        
        # Parse BRANCH records
        self.branches = self._parse_branches(pdbqt_file)
        
        # Calculate topology properties
        self._calculate_branch_properties()
        
        torsdof = len(self.branches)
        
        return {
            'torsdof': torsdof,
            'n_branches': len(self.branches),
            'branches': self.branches,
            'atoms': self.atoms,
            'needs_optimization': torsdof > 10
        }
    
    def _parse_branches(self, pdbqt_file: str) -> List[BranchInfo]:
        """
        Parse BRANCH records from PDBQT
        
        PDBQT BRANCH format:
        BRANCH   4  11
        (atoms in branch)
        ENDBRANCH   4  11
        """
        branches = []
        branch_stack = []
        current_branch_id = 0
        
        with open(pdbqt_file, 'r') as f:
            lines = f.readlines()
        
        atom_dict = {atom.index: atom for atom in self.atoms}
        line_to_atom = {}  # Map PDBQT line number to atom index
        
        atom_counter = 0
        for i, line in enumerate(lines):
            if line.startswith(('ATOM', 'HETATM')):
                line_to_atom[i] = atom_counter
                atom_counter += 1
        
        current_moving_atoms = set()
        
        for i, line in enumerate(lines):
            if line.startswith('BRANCH'):
                parts = line.split()
                if len(parts) >= 3:
                    axis_atom1 = int(parts[1])
                    axis_atom2 = int(parts[2])
                    
                    # Find atom types
                    a1_type = atom_dict[axis_atom1].atom_type if axis_atom1 in atom_dict else 'C'
                    a2_type = atom_dict[axis_atom2].atom_type if axis_atom2 in atom_dict else 'C'
                    
                    branch = BranchInfo(
                        branch_id=current_branch_id,
                        axis_atoms=(axis_atom1, axis_atom2),
                        axis_atom_types=(a1_type, a2_type),
                        moving_atoms=set(),
                        n_affected_atoms=0,
                        is_terminal=False,
                        depth=len(branch_stack)
                    )
                    
                    branch_stack.append(branch)
                    current_branch_id += 1
                    current_moving_atoms = set()
            
            elif line.startswith('ENDBRANCH'):
                if branch_stack:
                    branch = branch_stack.pop()
                    branch.moving_atoms = current_moving_atoms.copy()
                    branch.n_affected_atoms = len(current_moving_atoms)
                    
                    # Check if terminal (no sub-branches)
                    branch.is_terminal = branch.n_affected_atoms <= 5
                    
                    branches.append(branch)
                    
                    # Add these atoms to parent branch if exists
                    if branch_stack:
                        branch_stack[-1].moving_atoms.update(current_moving_atoms)
            
            elif line.startswith(('ATOM', 'HETATM')):
                if i in line_to_atom and branch_stack:
                    atom_idx = line_to_atom[i]
                    current_moving_atoms.add(atom_idx)
        
        return branches
    
    def _calculate_branch_properties(self):
        """Calculate additional properties for each branch"""
        for branch in self.branches:
            # Recalculate is_terminal based on whether it contains other branches
            sub_branches = sum(1 for b in self.branches 
                             if b.depth > branch.depth and 
                             b.axis_atoms[0] in branch.moving_atoms)
            
            branch.is_terminal = (sub_branches == 0) and (branch.n_affected_atoms <= 5)


# ============================================================================
# PocketDetector - Detect active site center
# ============================================================================

class PocketDetector:
    """
    Detect active site center from receptor
    Methods: geometric center, largest cavity, user-specified
    """
    
    @staticmethod
    def detect_from_receptor(receptor_pdbqt: str, 
                            method: str = 'geometric') -> np.ndarray:
        """
        Detect pocket center
        
        Args:
            receptor_pdbqt: Path to receptor PDBQT file
            method: 'geometric', 'mass_center', 'user' (returns None)
        
        Returns:
            np.ndarray of [x, y, z] or None
        """
        if method == 'user':
            return None
        
        # Parse receptor atoms
        coords = []
        masses = []
        
        ATOM_MASSES = {
            'C': 12.01, 'A': 12.01,
            'N': 14.01, 'NA': 14.01, 'NS': 14.01,
            'O': 16.00, 'OA': 16.00, 'OS': 16.00,
            'S': 32.06, 'SA': 32.06,
            'P': 30.97,
            'H': 1.008, 'HD': 1.008
        }
        
        with open(receptor_pdbqt, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
                        
                        atom_type = line[77:79].strip() if len(line) >= 79 else 'C'
                        mass = ATOM_MASSES.get(atom_type, 12.01)
                        masses.append(mass)
                        
                    except (ValueError, IndexError):
                        continue
        
        if not coords:
            return None
        
        coords = np.array(coords)
        masses = np.array(masses)
        
        if method == 'geometric':
            return np.mean(coords, axis=0)
        
        elif method == 'mass_center':
            total_mass = np.sum(masses)
            weighted_coords = coords * masses[:, np.newaxis]
            return np.sum(weighted_coords, axis=0) / total_mass
        
        else:
            return np.mean(coords, axis=0)
    
    @staticmethod
    def detect_from_ligand(ligand_pdbqt: str) -> np.ndarray:
        """Get ligand geometric center as pocket estimate"""
        coords = []
        
        with open(ligand_pdbqt, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
                    except (ValueError, IndexError):
                        continue
        
        if coords:
            return np.mean(coords, axis=0)
        return None


# ============================================================================
# BranchPrioritizer - Score branches for freeze/keep decision
# ============================================================================

class BranchPrioritizer:
    """
    Multi-dimensional scoring system for branch importance
    Higher score = keep rotation, Lower score = freeze rotation
    """
    
    def __init__(self, rigid_detector: RigidStructureDetector):
        self.rigid_detector = rigid_detector
        self.rigid_info = None
        
    def score_branches(self, 
                      pdbqt_file: str,
                      branches: List[BranchInfo],
                      atoms: List[AtomInfo],
                      pocket_center: Optional[np.ndarray] = None) -> List[BranchInfo]:
        """
        Score all branches
        
        Scoring dimensions:
        A. Position factors (30 points)
        B. Affected atoms (25 points)
        C. Distance to pocket (25 points, if pocket provided)
        D. Chemical functionality (20 points)
        
        Chemical hard constraints:
        - Aromatic ring internal: -1000 (force freeze)
        - Amide bond: -500 (force freeze)
        """
        # Detect rigid structures
        self.rigid_info = self.rigid_detector.detect_from_pdbqt(pdbqt_file)
        
        aromatic_atoms = self.rigid_info['aromatic_atoms']
        amide_bonds = self.rigid_info['amide_bonds']
        protected_bonds = self.rigid_info['protected_bonds']
        
        atom_dict = {atom.index: atom for atom in atoms}
        
        scored_branches = []
        
        for branch in branches:
            score = 50  # Neutral baseline
            
            axis_bond = tuple(sorted(branch.axis_atoms))
            
            # ========== CHEMICAL HARD CONSTRAINTS ==========
            
            # Check 1: Aromatic ring internal bond
            if self._both_in_aromatic(branch.axis_atoms, aromatic_atoms):
                score = -1000
                branch.freeze_reason = "Aromatic ring internal bond (chemically forbidden)"
                branch.priority_score = score
                scored_branches.append(branch)
                continue
            
            # Check 2: Amide bond
            if axis_bond in amide_bonds:
                score = -500
                branch.freeze_reason = "Amide bond planarity (chemically forbidden)"
                branch.priority_score = score
                scored_branches.append(branch)
                continue
            
            # Check 3: Other protected bonds
            if axis_bond in protected_bonds:
                score = -300
                branch.freeze_reason = "Small ring internal bond (high strain)"
                branch.priority_score = score
                scored_branches.append(branch)
                continue
            
            # ========== FLEXIBLE SCORING (non-rigid bonds) ==========
            
            # Dimension A: Position factors (30 points)
            if branch.is_terminal:
                score += 30  # Terminal rotations are important
            elif branch.depth == 1:
                score += 20  # First-level side chains
            elif branch.depth > 2:
                score -= 20  # Deep nested rotations
            
            # Check if backbone internal rotation
            if self._is_backbone_internal(branch):
                score -= 20
            
            # Check if hydrophobic chain
            if self._is_hydrophobic_chain(branch, atom_dict):
                score -= 15
            
            # Dimension B: Affected atoms (25 points)
            n_affected = branch.n_affected_atoms
            if n_affected <= 5:
                score += 25  # Micro-adjustment
            elif n_affected <= 10:
                score += 10
            elif n_affected <= 15:
                score -= 10
            else:
                score -= 40  # Conformational explosion culprit
            
            # Dimension C: Distance to pocket (25 points)
            if pocket_center is not None:
                distance = self._calc_distance_to_pocket(branch, atom_dict, pocket_center)
                branch.distance_to_pocket = distance
                
                if distance < 5.0:
                    score += 25  # Directly involved in binding
                elif distance < 8.0:
                    score += 10  # Second shell interaction
                elif distance < 12.0:
                    score -= 15  # Peripheral region
                else:
                    score -= 30  # Far from binding site
            
            # Dimension D: Chemical functionality (20 points)
            if self._one_in_aromatic(branch.axis_atoms, aromatic_atoms):
                score += 15  # Aromatic ring orientation adjustment
            
            if self._connects_pharmacophores(branch, atom_dict):
                score += 15  # Linker between functional groups
            
            if self._is_hbond_terminal(branch, atom_dict):
                score += 20  # H-bond donor/acceptor orientation
            
            branch.priority_score = score
            scored_branches.append(branch)
        
        # Sort by score (highest first = keep)
        return sorted(scored_branches, key=lambda x: x.priority_score, reverse=True)
    
    def _both_in_aromatic(self, axis_atoms: Tuple[int, int], 
                         aromatic_set: Set[int]) -> bool:
        """Both axis atoms in aromatic ring"""
        return all(a in aromatic_set for a in axis_atoms)
    
    def _one_in_aromatic(self, axis_atoms: Tuple[int, int], 
                        aromatic_set: Set[int]) -> bool:
        """Exactly one axis atom in aromatic ring (connecting bond)"""
        return sum(a in aromatic_set for a in axis_atoms) == 1
    
    def _is_backbone_internal(self, branch: BranchInfo) -> bool:
        """
        Internal backbone rotation
        Characteristics: not terminal, depth > 1, affects many atoms
        """
        return (not branch.is_terminal and 
                branch.depth > 1 and 
                branch.n_affected_atoms > 8)
    
    def _is_hydrophobic_chain(self, branch: BranchInfo, 
                             atom_dict: Dict[int, AtomInfo]) -> bool:
        """
        Hydrophobic alkyl chain internal rotation
        Check if axis is C-C and most moving atoms are C/H
        """
        axis_types = branch.axis_atom_types
        
        # Both axis atoms are carbon
        if not all(t in ['C', 'A'] for t in axis_types):
            return False
        
        # Check moving atoms
        if not branch.moving_atoms:
            return False
        
        moving_types = [atom_dict[a].atom_type for a in branch.moving_atoms 
                       if a in atom_dict]
        
        if not moving_types:
            return False
        
        # Count C and H
        c_h_count = sum(1 for t in moving_types if t in ['C', 'A', 'H', 'HD'])
        c_h_ratio = c_h_count / len(moving_types)
        
        return c_h_ratio > 0.8 and not branch.is_terminal
    
    def _calc_distance_to_pocket(self, branch: BranchInfo, 
                                 atom_dict: Dict[int, AtomInfo],
                                 pocket_center: np.ndarray) -> float:
        """Calculate minimum distance from branch atoms to pocket center"""
        if not branch.moving_atoms:
            return 999.0
        
        distances = []
        for atom_idx in branch.moving_atoms:
            if atom_idx in atom_dict:
                atom_coords = atom_dict[atom_idx].coords
                dist = np.linalg.norm(atom_coords - pocket_center)
                distances.append(dist)
        
        return min(distances) if distances else 999.0
    
    def _connects_pharmacophores(self, branch: BranchInfo,
                                 atom_dict: Dict[int, AtomInfo]) -> bool:
        """
        Check if rotation connects two pharmacophoric groups
        Simplified: check if axis atoms are different types
        """
        if len(branch.axis_atom_types) != 2:
            return False
        
        t1, t2 = branch.axis_atom_types
        
        # Different atom types suggest functional connection
        return t1 != t2
    
    def _is_hbond_terminal(self, branch: BranchInfo,
                          atom_dict: Dict[int, AtomInfo]) -> bool:
        """
        Check if terminal rotation with H-bond capable atoms (O, N)
        """
        if not branch.is_terminal:
            return False
        
        # Check moving atoms for O/N
        for atom_idx in branch.moving_atoms:
            if atom_idx in atom_dict:
                atom_type = atom_dict[atom_idx].atom_type
                if atom_type in ['O', 'OA', 'OS', 'N', 'NA', 'NS']:
                    return True
        
        return False


# ============================================================================
# LigandOptimizer - Execute optimization
# ============================================================================

class LigandOptimizer:
    """
    Main optimizer: reduce TORSDOF to target while preserving chemistry
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.rigid_detector = RigidStructureDetector()
        self.branch_analyzer = BranchAnalyzer()
        self.prioritizer = BranchPrioritizer(self.rigid_detector)
        
    def optimize_to_target_torsdof(self,
                                   pdbqt_file: str,
                                   target_torsdof: int = 10,
                                   pocket_center: Optional[np.ndarray] = None,
                                   output_file: Optional[str] = None) -> Tuple[str, OptimizationReport]:
        """
        Main optimization pipeline
        
        Args:
            pdbqt_file: Input PDBQT file path
            target_torsdof: Target rotatable bonds (default 10)
            pocket_center: Optional [x, y, z] active site center
            output_file: Output path (default: input_optimized.pdbqt)
        
        Returns:
            (optimized_pdbqt_path, OptimizationReport)
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  Ligand Flexibility Optimizer")
            print(f"{'='*70}")
            print(f"Input: {Path(pdbqt_file).name}")
            print(f"Target TORSDOF: {target_torsdof}")
            if pocket_center is not None:
                print(f"Pocket center: [{pocket_center[0]:.2f}, {pocket_center[1]:.2f}, {pocket_center[2]:.2f}]")
            print(f"{'='*70}\n")
        
        # Step 1: Analyze ligand
        if self.verbose:
            print("[1/5] Analyzing ligand structure...")
        
        analysis = self.branch_analyzer.analyze_pdbqt(pdbqt_file)
        original_torsdof = analysis['torsdof']
        
        if self.verbose:
            print(f"      Original TORSDOF: {original_torsdof}")
        
        if not analysis['needs_optimization']:
            if self.verbose:
                print(f"      ✓ Already within target (≤10), no optimization needed\n")
            
            report = OptimizationReport(
                original_torsdof=original_torsdof,
                optimized_torsdof=original_torsdof,
                chemical_constraints={},
                flexible_optimization={},
                warnings=[],
                recommendations=["Already optimized, no changes made"],
                frozen_branches=[],
                kept_branches=[]
            )
            
            return pdbqt_file, report
        
        # Step 2: Detect rigid structures
        if self.verbose:
            print("[2/5] Detecting rigid structures...")
        
        rigid_info = self.rigid_detector.detect_from_pdbqt(pdbqt_file)
        
        n_aromatic = len([r for r in rigid_info['rigid_regions'] if r.region_type == 'aromatic'])
        n_amide = len([r for r in rigid_info['rigid_regions'] if r.region_type == 'amide'])
        n_small_ring = len([r for r in rigid_info['rigid_regions'] if r.region_type == 'small_ring'])
        
        if self.verbose:
            print(f"      Aromatic rings: {n_aromatic}")
            print(f"      Amide bonds: {n_amide}")
            print(f"      Small rings: {n_small_ring}")
        
        # Step 3: Score branches
        if self.verbose:
            print("[3/5] Scoring rotatable bonds...")
        
        scored_branches = self.prioritizer.score_branches(
            pdbqt_file,
            analysis['branches'],
            analysis['atoms'],
            pocket_center
        )
        
        # Count chemical constraints
        chemical_frozen = [b for b in scored_branches if b.priority_score < 0]
        flexible_branches = [b for b in scored_branches if b.priority_score >= 0]
        
        if self.verbose:
            print(f"      Chemical constraints: {len(chemical_frozen)} bonds")
            print(f"      Flexible bonds: {len(flexible_branches)} bonds")
        
        # Step 4: Determine freeze list
        if self.verbose:
            print("[4/5] Determining optimization strategy...")
        
        current_flexible = len(flexible_branches)
        n_to_freeze = max(0, original_torsdof - target_torsdof)
        
        warnings = []
        recommendations = []
        
        # Check if achievable
        if len(chemical_frozen) >= target_torsdof:
            warnings.append(
                f"Chemical constraints already use {len(chemical_frozen)} bonds, "
                f"target {target_torsdof} may be too aggressive"
            )
            recommendations.append(f"Consider increasing target to {len(chemical_frozen) + 5}")
        
        # Determine actual branches to freeze
        if n_to_freeze > len(flexible_branches):
            warnings.append(
                f"Cannot freeze {n_to_freeze} bonds (only {len(flexible_branches)} flexible bonds available)"
            )
            n_to_freeze = len(flexible_branches)
            recommendations.append("Using maximum possible optimization")
        
        # Take lowest-scoring flexible branches
        branches_to_freeze = flexible_branches[-n_to_freeze:] if n_to_freeze > 0 else []
        branches_to_keep = flexible_branches[:-n_to_freeze] if n_to_freeze > 0 else flexible_branches
        
        # Add chemical constraints to frozen list
        all_frozen = chemical_frozen + branches_to_freeze
        
        final_torsdof = original_torsdof - len(all_frozen)
        
        if self.verbose:
            print(f"      Will freeze: {len(all_frozen)} bonds")
            print(f"      Will keep: {len(branches_to_keep)} bonds")
            print(f"      Final TORSDOF: {final_torsdof}")
        
        # Check if all terminals are frozen
        terminal_frozen = sum(1 for b in branches_to_freeze if b.is_terminal)
        if terminal_frozen > 0 and len(branches_to_keep) > 0:
            warnings.append(f"{terminal_frozen} terminal rotations frozen (usually should be kept)")
        
        # Step 5: Generate optimized PDBQT
        if self.verbose:
            print("[5/5] Generating optimized PDBQT...")
        
        if output_file is None:
            base = Path(pdbqt_file).stem
            output_file = str(Path(pdbqt_file).parent / f"{base}_optimized.pdbqt")
        
        self._freeze_branches(pdbqt_file, all_frozen, output_file)
        
        if self.verbose:
            print(f"      ✓ Saved: {Path(output_file).name}\n")
        
        # Generate recommendations
        if final_torsdof <= 10:
            recommendations.append("Optimization successful, ready for docking")
        
        if pocket_center is None:
            recommendations.append("Consider providing pocket center for better optimization")
        
        if len(branches_to_keep) > 0:
            kept_terminal = sum(1 for b in branches_to_keep if b.is_terminal)
            recommendations.append(f"Preserved {kept_terminal} terminal rotations for pharmacophore adjustment")
        
        # Build report
        report = OptimizationReport(
            original_torsdof=original_torsdof,
            optimized_torsdof=final_torsdof,
            chemical_constraints={
                'aromatic_rings': n_aromatic,
                'amide_bonds': n_amide,
                'small_rings': n_small_ring,
                'total_frozen': len(chemical_frozen)
            },
            flexible_optimization={
                'frozen_by_score': len(branches_to_freeze),
                'kept_branches': len(branches_to_keep),
                'terminal_kept': sum(1 for b in branches_to_keep if b.is_terminal)
            },
            warnings=warnings,
            recommendations=recommendations,
            frozen_branches=[
                {
                    'branch_id': b.branch_id,
                    'axis_atoms': b.axis_atoms,
                    'reason': b.freeze_reason if b.freeze_reason else self._get_freeze_reason(b),
                    'score': b.priority_score,
                    'affected_atoms': b.n_affected_atoms,
                    'distance_to_pocket': b.distance_to_pocket
                }
                for b in all_frozen
            ],
            kept_branches=[
                {
                    'branch_id': b.branch_id,
                    'axis_atoms': b.axis_atoms,
                    'score': b.priority_score,
                    'is_terminal': b.is_terminal,
                    'affected_atoms': b.n_affected_atoms
                }
                for b in branches_to_keep
            ]
        )
        
        return output_file, report
    
    def _freeze_branches(self, 
                        pdbqt_file: str,
                        branches_to_freeze: List[BranchInfo],
                        output_file: str):
        """
        Modify PDBQT file by removing specified BRANCH/ENDBRANCH blocks
        """
        with open(pdbqt_file, 'r') as f:
            lines = f.readlines()
        
        freeze_ids = {b.branch_id for b in branches_to_freeze}
        
        new_lines = []
        skip_depth = 0
        current_branch_id = -1
        
        for line in lines:
            if line.startswith('BRANCH'):
                current_branch_id += 1
                
                if current_branch_id in freeze_ids:
                    skip_depth = 1
                    continue
                
                if skip_depth > 0:
                    skip_depth += 1
                    continue
            
            if skip_depth > 0:
                if line.startswith('BRANCH'):
                    skip_depth += 1
                elif line.startswith('ENDBRANCH'):
                    skip_depth -= 1
                continue
            
            if line.startswith('TORSDOF'):
                new_torsdof = len([b for b in branches_to_freeze if b.branch_id <= current_branch_id])
                final_torsdof = max(0, current_branch_id + 1 - new_torsdof)
                new_lines.append(f"TORSDOF {final_torsdof}\n")
            else:
                new_lines.append(line)
        
        with open(output_file, 'w') as f:
            f.writelines(new_lines)
    
    def _get_freeze_reason(self, branch: BranchInfo) -> str:
        """Generate readable freeze reason"""
        if branch.freeze_reason:
            return branch.freeze_reason
        
        reasons = []
        
        if branch.is_terminal:
            reasons.append("terminal rotation")
        
        if branch.n_affected_atoms > 15:
            reasons.append("affects many atoms")
        elif branch.n_affected_atoms < 5:
            reasons.append("affects few atoms")
        
        if branch.distance_to_pocket > 10:
            reasons.append("far from active site")
        elif branch.distance_to_pocket < 5:
            reasons.append("close to active site")
        
        if branch.depth > 2:
            reasons.append("deeply nested")
        
        return ", ".join(reasons) if reasons else "low priority score"


# ============================================================================
# OptimizationReporter - Generate comprehensive reports
# ============================================================================

class OptimizationReporter:
    """
    Generate human-readable and machine-readable reports
    """
    
    @staticmethod
    def print_summary(report: OptimizationReport, ligand_name: str = "ligand"):
        """Print formatted summary to console"""
        print(f"\n{'='*70}")
        print(f"  Optimization Report - {ligand_name}")
        print(f"{'='*70}")
        print(f"Original TORSDOF:     {report.original_torsdof}")
        print(f"Optimized TORSDOF:    {report.optimized_torsdof}  "
              f"(-{report.original_torsdof - report.optimized_torsdof})")
        print(f"{'='*70}")
        
        # Chemical constraints
        if report.chemical_constraints:
            print(f"Chemical Constraints Frozen:")
            cc = report.chemical_constraints
            if cc.get('aromatic_rings', 0) > 0:
                print(f"  ✓ Aromatic ring bonds:  {cc['aromatic_rings']}")
            if cc.get('amide_bonds', 0) > 0:
                print(f"  ✓ Amide bond planes:    {cc['amide_bonds']}")
            if cc.get('small_rings', 0) > 0:
                print(f"  ✓ Small ring bonds:     {cc['small_rings']}")
            print(f"{'='*70}")
        
        # Flexible optimization
        if report.flexible_optimization:
            fo = report.flexible_optimization
            print(f"Smart Optimization:")
            print(f"  ✓ Frozen by scoring:    {fo.get('frozen_by_score', 0)}")
            print(f"  ✓ Kept flexible:        {fo.get('kept_branches', 0)}")
            print(f"  ✓ Terminal preserved:   {fo.get('terminal_kept', 0)}")
            print(f"{'='*70}")
        
        # Warnings
        if report.warnings:
            print(f"⚠️  Warnings:")
            for warning in report.warnings:
                print(f"  - {warning}")
            print(f"{'='*70}")
        
        # Recommendations
        if report.recommendations:
            print(f"✅ Recommendations:")
            for rec in report.recommendations:
                print(f"  - {rec}")
            print(f"{'='*70}\n")
    
    @staticmethod
    def save_json_report(report: OptimizationReport, output_file: str):
        """Save detailed JSON report"""
        report_dict = asdict(report)
        
        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
    
    @staticmethod
    def save_detailed_log(report: OptimizationReport, 
                         branches_frozen: List[BranchInfo],
                         output_file: str):
        """Save detailed text log with all branch information"""
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DETAILED OPTIMIZATION LOG\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Original TORSDOF: {report.original_torsdof}\n")
            f.write(f"Optimized TORSDOF: {report.optimized_torsdof}\n")
            f.write(f"Reduction: {report.original_torsdof - report.optimized_torsdof}\n\n")
            
            f.write("-"*70 + "\n")
            f.write("FROZEN BRANCHES:\n")
            f.write("-"*70 + "\n")
            
            for fb in report.frozen_branches:
                f.write(f"\nBranch ID: {fb['branch_id']}\n")
                f.write(f"  Axis atoms: {fb['axis_atoms']}\n")
                f.write(f"  Reason: {fb['reason']}\n")
                f.write(f"  Score: {fb['score']:.1f}\n")
                f.write(f"  Affected atoms: {fb['affected_atoms']}\n")
                if fb['distance_to_pocket'] < 999:
                    f.write(f"  Distance to pocket: {fb['distance_to_pocket']:.2f} Å\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("KEPT BRANCHES:\n")
            f.write("-"*70 + "\n")
            
            for kb in report.kept_branches:
                f.write(f"\nBranch ID: {kb['branch_id']}\n")
                f.write(f"  Axis atoms: {kb['axis_atoms']}\n")
                f.write(f"  Score: {kb['score']:.1f}\n")
                f.write(f"  Terminal: {kb['is_terminal']}\n")
                f.write(f"  Affected atoms: {kb['affected_atoms']}\n")


# ============================================================================
# Main Function - Command Line Interface
# ============================================================================

def main():
    """Command-line interface for standalone usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='AutoPre Ligand Advanced - Intelligent Ligand Flexibility Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic optimization (auto-detect, target TORSDOF=10)
  python autopre_ligand_advanced.py ligand.pdbqt
  
  # Custom target TORSDOF
  python autopre_ligand_advanced.py ligand.pdbqt --target 8
  
  # With pocket center
  python autopre_ligand_advanced.py ligand.pdbqt --pocket 15.2 -8.3 22.1
  
  # Auto-detect pocket from receptor
  python autopre_ligand_advanced.py ligand.pdbqt --receptor protein.pdbqt
  
  # Batch processing
  python autopre_ligand_advanced.py *.pdbqt --output optimized/
        """
    )
    
    parser.add_argument('input', nargs='+', help='Input PDBQT file(s)')
    parser.add_argument('--target', type=int, default=10,
                       help='Target TORSDOF (default: 10)')
    parser.add_argument('--pocket', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                       help='Pocket center coordinates')
    parser.add_argument('--receptor', help='Receptor PDBQT for pocket detection')
    parser.add_argument('--output', help='Output directory (default: same as input)')
    parser.add_argument('--report', choices=['summary', 'json', 'detailed', 'all'],
                       default='summary', help='Report format (default: summary)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Determine pocket center
    pocket_center = None
    if args.pocket:
        pocket_center = np.array(args.pocket)
        print(f"Using specified pocket center: {pocket_center}")
    elif args.receptor:
        pocket_center = PocketDetector.detect_from_receptor(args.receptor)
        if pocket_center is not None:
            print(f"Detected pocket center from receptor: {pocket_center}")
    
    # Setup output directory
    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each ligand
    optimizer = LigandOptimizer(verbose=not args.quiet)
    reporter = OptimizationReporter()
    
    for input_file in args.input:
        input_path = Path(input_file)
        
        if not input_path.exists():
            print(f"Warning: {input_file} not found, skipping")
            continue
        
        # Determine output path
        if output_dir:
            output_file = str(output_dir / f"{input_path.stem}_optimized.pdbqt")
        else:
            output_file = str(input_path.parent / f"{input_path.stem}_optimized.pdbqt")
        
        # Optimize
        try:
            optimized_pdbqt, report = optimizer.optimize_to_target_torsdof(
                str(input_path),
                target_torsdof=args.target,
                pocket_center=pocket_center,
                output_file=output_file
            )
            
            # Generate reports
            if args.report in ['summary', 'all']:
                reporter.print_summary(report, input_path.stem)
            
            if args.report in ['json', 'all']:
                json_file = str(Path(output_file).with_suffix('.json'))
                reporter.save_json_report(report, json_file)
                if not args.quiet:
                    print(f"JSON report saved: {json_file}")
            
            if args.report in ['detailed', 'all']:
                log_file = str(Path(output_file).with_suffix('.log'))
                reporter.save_detailed_log(report, report.frozen_branches, log_file)
                if not args.quiet:
                    print(f"Detailed log saved: {log_file}")
        
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
