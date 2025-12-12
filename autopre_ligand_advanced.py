#!/usr/bin/env python3
"""
AutoPre Ligand Advanced - Intelligent Ligand Flexibility Optimizer (FIXED VERSION)
Automatically reduces ligand rotatable bonds (TORSDOF) while preserving chemical validity
Author: ZymEvo Development Team
Version: 2.0 (Bug fixes + Better logic)
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AtomInfo:
    """Atom information from PDBQT"""
    index: int
    name: str
    atom_type: str  # PDBQT atom type (A, C, N, NA, etc.)
    coords: np.ndarray
    residue: str
    
    
@dataclass
class BranchInfo:
    """Branch (rotatable bond) information"""
    branch_id: int
    axis_atoms: Tuple[int, int]  # Atom indices defining rotation axis
    axis_atom_types: Tuple[str, str]
    moving_atoms: Set[int]  # Atoms affected by this rotation
    n_affected_atoms: int
    is_terminal: bool
    depth: int  # Nesting level in BRANCH tree
    line_start: int = 0  # NEW: Line number where BRANCH starts
    line_end: int = 0    # NEW: Line number where ENDBRANCH ends
    priority_score: float = 0.0
    freeze_reason: str = ""
    distance_to_pocket: float = 999.0
    

@dataclass
class RigidRegion:
    """Rigid structural region"""
    region_type: str  # 'aromatic', 'amide', 'small_ring', 'conjugated'
    atoms: Set[int]
    bonds: Set[Tuple[int, int]]
    reason: str


@dataclass
class OptimizationReport:
    """Comprehensive optimization report"""
    original_torsdof: int
    optimized_torsdof: int
    chemical_constraints: Dict[str, int]
    flexible_optimization: Dict
    warnings: List[str]
    recommendations: List[str]
    frozen_branches: List[Dict]
    kept_branches: List[Dict]


# ============================================================================
# RigidStructureDetector - Detect chemically rigid structures
# ============================================================================

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
        """Build bond connectivity from distance"""
        bonds = set()
        
        COVALENT_RADII = {
            'C': 1.70, 'A': 1.70,
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
        """Detect all ring structures using DFS"""
        graph = defaultdict(list)
        for a1, a2 in self.bonds:
            graph[a1].append(a2)
            graph[a2].append(a1)
        
        rings = []
        visited_global = set()
        
        def dfs_find_rings(start):
            stack = [(start, [start], {start})]
            
            while stack:
                node, path, visited = stack.pop()
                
                for neighbor in graph[node]:
                    if neighbor == path[-2] if len(path) > 1 else -1:
                        continue
                    
                    if neighbor in visited:
                        ring_start = path.index(neighbor)
                        ring = set(path[ring_start:])
                        
                        if 3 <= len(ring) <= 20:
                            if not any(ring == existing for existing in rings):
                                rings.append(ring)
                    else:
                        stack.append((neighbor, path + [neighbor], visited | {neighbor}))
        
        for atom in self.atoms:
            if atom.index not in visited_global:
                dfs_find_rings(atom.index)
                visited_global.add(atom.index)
        
        return rings
    
    def _is_rigid_ring(self, ring: Set[int], aromatic_atoms: Set[int]) -> bool:
        """Determine if a ring is rigid"""
        # 1. Aromatic ring
        if ring.issubset(aromatic_atoms):
            return True
        
        # 2. Small ring (3-4 membered)
        if len(ring) <= 4:
            return True
        
        return False
    
    def _detect_amide_bonds(self) -> Set[Tuple[int, int]]:
        """Detect amide bonds: C(=O)-N pattern"""
        amide_bonds = set()
        
        atom_dict = {atom.index: atom for atom in self.atoms}
        
        for bond in self.bonds:
            a1_idx, a2_idx = bond
            a1 = atom_dict[a1_idx]
            a2 = atom_dict[a2_idx]
            
            c_atom, n_atom = None, None
            
            if a1.atom_type == 'C' and a2.atom_type == 'N':
                c_atom, n_atom = a1, a2
            elif a1.atom_type == 'N' and a2.atom_type == 'C':
                c_atom, n_atom = a2, a1
            else:
                continue
            
            c_neighbors = [atom_dict[nb] for a, nb in self.bonds 
                          if a == c_atom.index or nb == c_atom.index]
            
            has_carbonyl_o = any(
                nb.atom_type in ['O', 'OA'] and 
                np.linalg.norm(c_atom.coords - nb.coords) < 1.35
                for nb in c_neighbors
            )
            
            if has_carbonyl_o:
                amide_bonds.add(tuple(sorted([c_atom.index, n_atom.index])))
        
        return amide_bonds


# ============================================================================
# BranchAnalyzer - Parse and analyze BRANCH records in PDBQT
# ============================================================================

class BranchAnalyzer:
    """Parse BRANCH structure from PDBQT file - FIXED VERSION"""
    
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
        detector = RigidStructureDetector()
        self.atoms = detector._parse_atoms(pdbqt_file)
        
        # Parse BRANCH records with line numbers
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
        Parse BRANCH records - FIXED to track line numbers
        """
        with open(pdbqt_file, 'r') as f:
            lines = f.readlines()
        
        branches = []
        branch_stack = []
        current_branch_id = -1
        
        atom_dict = {atom.index: atom for atom in self.atoms}
        
        # Map line numbers to atom indices
        line_to_atom = {}
        atom_counter = 0
        for i, line in enumerate(lines):
            if line.startswith(('ATOM', 'HETATM')):
                line_to_atom[i] = atom_counter
                atom_counter += 1
        
        for line_num, line in enumerate(lines):
            if line.startswith('BRANCH'):
                current_branch_id += 1
                parts = line.split()
                
                if len(parts) >= 3:
                    try:
                        axis_atom1 = int(parts[1])
                        axis_atom2 = int(parts[2])
                    except ValueError:
                        continue
                    
                    a1_type = atom_dict[axis_atom1].atom_type if axis_atom1 in atom_dict else 'C'
                    a2_type = atom_dict[axis_atom2].atom_type if axis_atom2 in atom_dict else 'C'
                    
                    branch = BranchInfo(
                        branch_id=current_branch_id,
                        axis_atoms=(axis_atom1, axis_atom2),
                        axis_atom_types=(a1_type, a2_type),
                        moving_atoms=set(),
                        n_affected_atoms=0,
                        is_terminal=False,
                        depth=len(branch_stack),
                        line_start=line_num  # Record start line
                    )
                    
                    branch_stack.append(branch)
            
            elif line.startswith('ENDBRANCH'):
                if branch_stack:
                    branch = branch_stack.pop()
                    branch.line_end = line_num  # Record end line
                    
                    # Collect atoms in this branch
                    moving_atoms = set()
                    for i in range(branch.line_start + 1, branch.line_end):
                        if i in line_to_atom:
                            moving_atoms.add(line_to_atom[i])
                    
                    branch.moving_atoms = moving_atoms
                    branch.n_affected_atoms = len(moving_atoms)
                    branch.is_terminal = len(moving_atoms) <= 5
                    
                    branches.append(branch)
                    
                    # Add to parent's moving atoms
                    if branch_stack:
                        branch_stack[-1].moving_atoms.update(moving_atoms)
        
        return branches
    
    def _calculate_branch_properties(self):
        """Calculate additional properties"""
        for branch in self.branches:
            sub_branches = sum(1 for b in self.branches 
                             if b.depth > branch.depth and 
                             b.axis_atoms[0] in branch.moving_atoms)
            
            branch.is_terminal = (sub_branches == 0) and (branch.n_affected_atoms <= 5)


# ============================================================================
# PocketDetector - Detect active site center
# ============================================================================

class PocketDetector:
    """Detect active site center from receptor"""
    
    @staticmethod
    def detect_from_receptor(receptor_pdbqt: str, 
                            method: str = 'geometric') -> np.ndarray:
        """Detect pocket center"""
        if method == 'user':
            return None
        
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
        """Get ligand geometric center"""
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
    """Multi-dimensional scoring system - IMPROVED"""
    
    def __init__(self, rigid_detector: RigidStructureDetector):
        self.rigid_detector = rigid_detector
        self.rigid_info = None
        
    def score_branches(self, 
                      pdbqt_file: str,
                      branches: List[BranchInfo],
                      atoms: List[AtomInfo],
                      pocket_center: Optional[np.ndarray] = None) -> List[BranchInfo]:
        """Score all branches"""
        
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
            
            if self._both_in_aromatic(branch.axis_atoms, aromatic_atoms):
                score = -1000
                branch.freeze_reason = "Aromatic ring internal bond (chemically forbidden)"
                branch.priority_score = score
                scored_branches.append(branch)
                continue
            
            if axis_bond in amide_bonds:
                score = -500
                branch.freeze_reason = "Amide bond planarity (chemically forbidden)"
                branch.priority_score = score
                scored_branches.append(branch)
                continue
            
            if axis_bond in protected_bonds:
                score = -300
                branch.freeze_reason = "Small ring internal bond (high strain)"
                branch.priority_score = score
                scored_branches.append(branch)
                continue
            
            # ========== FLEXIBLE SCORING ==========
            
            if branch.is_terminal:
                score += 30
            elif branch.depth == 1:
                score += 20
            elif branch.depth > 2:
                score -= 20
            
            if self._is_backbone_internal(branch):
                score -= 20
            
            if self._is_hydrophobic_chain(branch, atom_dict):
                score -= 15
            
            n_affected = branch.n_affected_atoms
            if n_affected <= 5:
                score += 25
            elif n_affected <= 10:
                score += 10
            elif n_affected <= 15:
                score -= 10
            else:
                score -= 40
            
            if pocket_center is not None:
                distance = self._calc_distance_to_pocket(branch, atom_dict, pocket_center)
                branch.distance_to_pocket = distance
                
                if distance < 5.0:
                    score += 25
                elif distance < 8.0:
                    score += 10
                elif distance < 12.0:
                    score -= 15
                else:
                    score -= 30
            
            if self._one_in_aromatic(branch.axis_atoms, aromatic_atoms):
                score += 15
            
            if self._connects_pharmacophores(branch, atom_dict):
                score += 15
            
            if self._is_hbond_terminal(branch, atom_dict):
                score += 20
            
            branch.priority_score = score
            scored_branches.append(branch)
        
        return sorted(scored_branches, key=lambda x: x.priority_score, reverse=True)
    
    def _both_in_aromatic(self, axis_atoms: Tuple[int, int], aromatic_set: Set[int]) -> bool:
        return all(a in aromatic_set for a in axis_atoms)
    
    def _one_in_aromatic(self, axis_atoms: Tuple[int, int], aromatic_set: Set[int]) -> bool:
        return sum(a in aromatic_set for a in axis_atoms) == 1
    
    def _is_backbone_internal(self, branch: BranchInfo) -> bool:
        return (not branch.is_terminal and branch.depth > 1 and branch.n_affected_atoms > 8)
    
    def _is_hydrophobic_chain(self, branch: BranchInfo, atom_dict: Dict[int, AtomInfo]) -> bool:
        axis_types = branch.axis_atom_types
        
        if not all(t in ['C', 'A'] for t in axis_types):
            return False
        
        if not branch.moving_atoms:
            return False
        
        moving_types = [atom_dict[a].atom_type for a in branch.moving_atoms if a in atom_dict]
        
        if not moving_types:
            return False
        
        c_h_count = sum(1 for t in moving_types if t in ['C', 'A', 'H', 'HD'])
        c_h_ratio = c_h_count / len(moving_types)
        
        return c_h_ratio > 0.8 and not branch.is_terminal
    
    def _calc_distance_to_pocket(self, branch: BranchInfo, 
                                 atom_dict: Dict[int, AtomInfo],
                                 pocket_center: np.ndarray) -> float:
        if not branch.moving_atoms:
            return 999.0
        
        distances = []
        for atom_idx in branch.moving_atoms:
            if atom_idx in atom_dict:
                atom_coords = atom_dict[atom_idx].coords
                dist = np.linalg.norm(atom_coords - pocket_center)
                distances.append(dist)
        
        return min(distances) if distances else 999.0
    
    def _connects_pharmacophores(self, branch: BranchInfo, atom_dict: Dict[int, AtomInfo]) -> bool:
        if len(branch.axis_atom_types) != 2:
            return False
        
        t1, t2 = branch.axis_atom_types
        return t1 != t2
    
    def _is_hbond_terminal(self, branch: BranchInfo, atom_dict: Dict[int, AtomInfo]) -> bool:
        if not branch.is_terminal:
            return False
        
        for atom_idx in branch.moving_atoms:
            if atom_idx in atom_dict:
                atom_type = atom_dict[atom_idx].atom_type
                if atom_type in ['O', 'OA', 'OS', 'N', 'NA', 'NS']:
                    return True
        
        return False


# ============================================================================
# LigandOptimizer - Execute optimization - COMPLETELY REWRITTEN
# ============================================================================

class LigandOptimizer:
    """Main optimizer - FIXED VERSION with proper BRANCH removal"""
    
    # Minimum flexibility preservation
    MIN_FLEXIBLE_BONDS = 5
    MIN_FLEXIBLE_RATIO = 0.20  # Keep at least 20% of flexible bonds
    
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
        """Main optimization pipeline - IMPROVED LOGIC"""
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  Ligand Flexibility Optimizer v2.0")
            print(f"{'='*70}")
            print(f"Input: {Path(pdbqt_file).name}")
            print(f"Target TORSDOF: {target_torsdof}")
            if pocket_center is not None:
                print(f"Pocket center: [{pocket_center[0]:.2f}, {pocket_center[1]:.2f}, {pocket_center[2]:.2f}]")
            print(f"{'='*70}\n")
        
        # Step 1: Analyze
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
        
        # Separate chemical constraints from flexible bonds
        chemical_frozen = [b for b in scored_branches if b.priority_score < 0]
        flexible_branches = [b for b in scored_branches if b.priority_score >= 0]
        
        if self.verbose:
            print(f"      Chemical constraints: {len(chemical_frozen)} bonds")
            print(f"      Flexible bonds: {len(flexible_branches)} bonds")
            print(f"      DEBUG: Total scored branches = {len(scored_branches)}")
            
            # Print distribution
            if len(chemical_frozen) > 0:
                print(f"      DEBUG: Chemical frozen reasons:")
                reason_counts = {}
                for b in chemical_frozen[:5]:  # Show first 5
                    reason = b.freeze_reason if b.freeze_reason else "Unknown"
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    print(f"        Branch {b.branch_id}: {reason} (score={b.priority_score})")
                if len(chemical_frozen) > 5:
                    print(f"        ... and {len(chemical_frozen) - 5} more")
                    
            if len(flexible_branches) > 0:
                print(f"      DEBUG: Flexible branches score range: "
                      f"{flexible_branches[-1].priority_score} to {flexible_branches[0].priority_score}")
        
        # Step 4: Smart freeze decision - IMPROVED
        if self.verbose:
            print("[4/5] Determining optimization strategy...")
        
        warnings = []
        recommendations = []
        
        # Calculate how many we need to freeze
        n_to_freeze_total = original_torsdof - target_torsdof
        n_chemical_frozen = len(chemical_frozen)
        n_flexible_available = len(flexible_branches)
        
        if self.verbose:
            print(f"      DEBUG: n_to_freeze_total = {original_torsdof} - {target_torsdof} = {n_to_freeze_total}")
            print(f"      DEBUG: n_chemical_frozen = {n_chemical_frozen}")
            print(f"      DEBUG: n_flexible_available = {n_flexible_available}")
        
        # Calculate minimum flexible bonds to keep
        min_keep = max(
            self.MIN_FLEXIBLE_BONDS,
            int(n_flexible_available * self.MIN_FLEXIBLE_RATIO)
        )
        
        # Maximum we can freeze from flexible bonds
        max_freezable = max(0, n_flexible_available - min_keep)
        
        # How many flexible bonds we actually need to freeze
        n_flexible_to_freeze = n_to_freeze_total - n_chemical_frozen
        
        if self.verbose:
            print(f"      需要冻结总数: {n_to_freeze_total}")
            print(f"      化学约束已冻结: {n_chemical_frozen}")
            print(f"      需要从柔性中冻结: {n_flexible_to_freeze}")
            print(f"      柔性分支最少保留: {min_keep} ({int(self.MIN_FLEXIBLE_RATIO*100)}% 规则)")
            print(f"      最多可冻结柔性: {max_freezable}")
            
        # Adjust if we can't reach target
        if n_flexible_to_freeze > max_freezable:
            warnings.append(
                f"Cannot reach target TORSDOF={target_torsdof} while preserving minimum flexibility"
            )
            warnings.append(
                f"Will keep at least {min_keep} flexible bonds ({int(self.MIN_FLEXIBLE_RATIO*100)}% minimum)"
            )
            n_flexible_to_freeze = max_freezable
            
            final_torsdof = original_torsdof - n_chemical_frozen - n_flexible_to_freeze
            recommendations.append(
                f"Recommended target for this ligand: {final_torsdof} (preserves key flexibility)"
            )
            
            if self.verbose:
                print(f"      ⚠️  无法达到目标 {target_torsdof}，实际将达到 {final_torsdof}")
        
        # Select branches to freeze
        if n_flexible_to_freeze > 0:
            branches_to_freeze = flexible_branches[-n_flexible_to_freeze:]
            branches_to_keep = flexible_branches[:-n_flexible_to_freeze]
            
            if self.verbose:
                print(f"      DEBUG: 切片逻辑:")
                print(f"        flexible_branches总数: {len(flexible_branches)}")
                print(f"        取最后{n_flexible_to_freeze}个作为to_freeze")
                print(f"        branches_to_freeze数量: {len(branches_to_freeze)}")
                print(f"        branches_to_keep数量: {len(branches_to_keep)}")
        else:
            branches_to_freeze = []
            branches_to_keep = flexible_branches
        
        all_frozen = chemical_frozen + branches_to_freeze
        final_torsdof = original_torsdof - len(all_frozen)
        
        if self.verbose:
            print(f"      Chemical frozen: {len(chemical_frozen)} bonds")
            print(f"      Flexible frozen: {len(branches_to_freeze)} bonds")
            print(f"      Kept flexible: {len(branches_to_keep)} bonds")
            print(f"      Final TORSDOF: {final_torsdof}")
        
        # Step 5: Generate optimized PDBQT - FIXED
        if self.verbose:
            print("[5/5] Generating optimized PDBQT...")
        
        if output_file is None:
            base = Path(pdbqt_file).stem
            output_file = str(Path(pdbqt_file).parent / f"{base}_optimized.pdbqt")
        
        actual_final_torsdof = self._freeze_branches_fixed(pdbqt_file, all_frozen, output_file)
        
        if self.verbose:
            print(f"      ✓ Saved: {Path(output_file).name}\n")
        
        # Generate recommendations
        if final_torsdof <= target_torsdof:
            recommendations.append("Optimization successful, ready for docking")
        
        if pocket_center is None:
            recommendations.append("Consider providing pocket center for better optimization")
        
        if len(branches_to_keep) > 0:
            kept_terminal = sum(1 for b in branches_to_keep if b.is_terminal)
            recommendations.append(f"Preserved {kept_terminal} terminal rotations for pharmacophore adjustment")
        
        # Build report
        report = OptimizationReport(
            original_torsdof=original_torsdof,
            optimized_torsdof=actual_final_torsdof,
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

    def _freeze_branches_fixed(self, 
                           pdbqt_file: str,
                           branches_to_freeze: List[BranchInfo],
                           output_file: str) -> int:
        """
        Freeze branches by removing their BRANCH/ENDBRANCH blocks
        FINAL FIX: Implement branch promotion for nested kept branches
        
        Strategy:
        1. Identify branches to freeze and keep
        2. For kept branches with frozen parents: promote them to ROOT level
        3. Remove only the branches that should be frozen (no children kept)
        """
        with open(pdbqt_file, 'r') as f:
            lines = f.readlines()
        
        # Get sets of branch_ids
        freeze_ids = {b.branch_id for b in branches_to_freeze}
        all_branch_ids = set(range(len(self.branch_analyzer.branches)))
        keep_ids = all_branch_ids - freeze_ids
        
        if self.verbose:
            print(f"      DEBUG: Freeze IDs: {sorted(freeze_ids)}")
            print(f"      DEBUG: Keep IDs: {sorted(keep_ids)}")
        
        # Build parent-child relationships
        parent_map = {}  # child_id -> parent_id
        children_map = {}  # parent_id -> [child_ids]
        
        branch_stack = []
        current_id = -1
        
        for line_num, line in enumerate(lines):
            if line.startswith('BRANCH'):
                current_id += 1
                parent_id = branch_stack[-1] if branch_stack else None
                parent_map[current_id] = parent_id
                
                if parent_id is not None:
                    if parent_id not in children_map:
                        children_map[parent_id] = []
                    children_map[parent_id].append(current_id)
                
                branch_stack.append(current_id)
            elif line.startswith('ENDBRANCH'):
                if branch_stack:
                    branch_stack.pop()
        
        # Find branches that need promotion
        # A kept branch needs promotion if ANY ancestor is frozen
        def has_frozen_ancestor(branch_id):
            current = parent_map.get(branch_id)
            while current is not None:
                if current in freeze_ids:
                    return True
                current = parent_map.get(current)
            return False
        
        promote_ids = {bid for bid in keep_ids if has_frozen_ancestor(bid)}
        
        if self.verbose:
            print(f"      DEBUG: Branches needing promotion: {sorted(promote_ids)}")
        
        # Determine final action for each branch:
        # - REMOVE: branch is frozen AND no kept descendants
        # - PROMOTE: branch is kept but has frozen ancestor
        # - KEEP: branch is kept and no frozen ancestor
        
        def has_kept_descendants(branch_id):
            """Check if this branch has any kept descendants"""
            if branch_id in keep_ids:
                return True
            for child_id in children_map.get(branch_id, []):
                if has_kept_descendants(child_id):
                    return True
            return False
        
        branches_to_remove = {
            bid for bid in freeze_ids 
            if not has_kept_descendants(bid)
        }
        
        if self.verbose:
            print(f"      DEBUG: Branches to actually remove: {sorted(branches_to_remove)}")
            print(f"      DEBUG: Branches to promote: {sorted(promote_ids)}")
        
        # Process file
        new_lines = []
        skip_depth = 0
        branch_counter = -1
        promoted_branches_content = {}  # branch_id -> lines
        currently_promoting = None
        
        for line_num, line in enumerate(lines):
            if line.startswith('BRANCH'):
                branch_counter += 1
                
                # Should we remove this branch?
                if branch_counter in branches_to_remove:
                    skip_depth = 1
                    continue
                
                # Are we inside a skip zone?
                if skip_depth > 0:
                    # Check if this is a branch we want to promote
                    if branch_counter in promote_ids:
                        currently_promoting = branch_counter
                        promoted_branches_content[branch_counter] = []
                        skip_depth = 0  # Stop skipping, start capturing
                    else:
                        skip_depth += 1
                        continue
            
            elif line.startswith('ENDBRANCH'):
                if skip_depth > 0:
                    skip_depth -= 1
                    continue
                
                # Finish promoting this branch
                if currently_promoting is not None:
                    # Don't add yet, will add at ROOT level later
                    currently_promoting = None
                    continue
            
            # TORSDOF line
            if line.startswith('TORSDOF'):
                # Will rewrite later
                continue
            
            # Regular lines
            if skip_depth == 0:
                if currently_promoting is not None:
                    # Capture content for promoted branch
                    promoted_branches_content[currently_promoting].append(line)
                else:
                    new_lines.append(line)
        
        # Add promoted branches at ROOT level (after ROOT/ENDROOT)
        # Find where ROOT ends
        root_end = None
        for i, line in enumerate(new_lines):
            if line.startswith('ENDROOT'):
                root_end = i
                break
        
        if root_end is not None and promoted_branches_content:
            # Insert promoted branches after ENDROOT
            insert_pos = root_end + 1
            for branch_id in sorted(promote_ids):
                if branch_id in promoted_branches_content:
                    content = promoted_branches_content[branch_id]
                    # Wrap in BRANCH/ENDBRANCH
                    branch_obj = self.branch_analyzer.branches[branch_id]
                    axis = branch_obj.axis_atoms
                    new_lines.insert(insert_pos, f"BRANCH {axis[0]:4d} {axis[1]:4d}\n")
                    insert_pos += 1
                    for content_line in content:
                        new_lines.insert(insert_pos, content_line)
                        insert_pos += 1
                    new_lines.insert(insert_pos, f"ENDBRANCH {axis[0]:4d} {axis[1]:4d}\n")
                    insert_pos += 1
        
        # Calculate and insert TORSDOF
        final_torsdof = sum(1 for l in new_lines if l.startswith('BRANCH'))
        
        # Find TORSDOF line position (usually near end, before END)
        torsdof_pos = None
        for i in range(len(new_lines) - 1, -1, -1):
            if new_lines[i].startswith('TORSDOF') or i == len(new_lines) - 1:
                torsdof_pos = i
                break
        
        if torsdof_pos is not None:
            if new_lines[torsdof_pos].startswith('TORSDOF'):
                new_lines[torsdof_pos] = f"TORSDOF {final_torsdof}\n"
            else:
                new_lines.insert(torsdof_pos, f"TORSDOF {final_torsdof}\n")
        else:
            new_lines.append(f"TORSDOF {final_torsdof}\n")
        
        # Write output
        with open(output_file, 'w') as f:
            f.writelines(new_lines)
        
        # Verify
        n_branch = sum(1 for l in new_lines if l.startswith('BRANCH'))
        n_endbranch = sum(1 for l in new_lines if l.startswith('ENDBRANCH'))
        
        if self.verbose:
            print(f"      DEBUG: Final TORSDOF: {final_torsdof}")
            print(f"      DEBUG: BRANCH/ENDBRANCH: {n_branch}/{n_endbranch}")
            
            if n_branch != n_endbranch:
                print(f"      ⚠️  BRANCH/ENDBRANCH mismatch!")
        
        return final_torsdof
        
        # Write output file
        with open(output_file, 'w') as f:
            f.writelines(new_lines)
        
        # Verify final TORSDOF
        final_torsdof = sum(1 for l in new_lines if l.startswith('BRANCH'))
        
        # Verify BRANCH/ENDBRANCH balance
        n_endbranch = sum(1 for l in new_lines if l.startswith('ENDBRANCH'))
        
        if self.verbose:
            print(f"      DEBUG: Final TORSDOF in file: {final_torsdof}")
            print(f"      DEBUG: BRANCH count: {final_torsdof}")
            print(f"      DEBUG: ENDBRANCH count: {n_endbranch}")
            
            if final_torsdof != n_endbranch:
                print(f"      ⚠️  WARNING: BRANCH/ENDBRANCH mismatch!")
        
        # Double-check by reading the file back
        with open(output_file, 'r') as f:
            verify_lines = f.readlines()
        
        actual_torsdof = sum(1 for l in verify_lines if l.startswith('BRANCH'))
        declared_torsdof = None
        
        for line in verify_lines:
            if line.startswith('TORSDOF'):
                declared_torsdof = int(line.split()[1])
                break
        
        if self.verbose and declared_torsdof is not None:
            print(f"      DEBUG: Declared TORSDOF in file: {declared_torsdof}")
            
            if actual_torsdof != declared_torsdof:
                print(f"      ⚠️  WARNING: TORSDOF mismatch! "
                      f"Actual BRANCH={actual_torsdof}, Declared={declared_torsdof}")
        
        return actual_torsdof  # Return the actual count

    
    def _get_freeze_reason(self, branch: BranchInfo) -> str:
        """Generate readable freeze reason"""
        if branch.freeze_reason:
            return branch.freeze_reason
        
        reasons = []
        
        if branch.n_affected_atoms > 15:
            reasons.append("affects many atoms")
        elif branch.n_affected_atoms < 5:
            reasons.append("small terminal group")
        
        if branch.distance_to_pocket > 10:
            reasons.append("far from active site")
        
        if branch.depth > 2:
            reasons.append("deeply nested")
        
        return ", ".join(reasons) if reasons else "low priority score"

# ============================================================================
# OptimizationReporter - Generate reports
# ============================================================================

class OptimizationReporter:
    """Generate reports"""
    
    @staticmethod
    def print_summary(report: OptimizationReport, ligand_name: str = "ligand"):
        """Print formatted summary"""
        print(f"\n{'='*70}")
        print(f"  Optimization Report - {ligand_name}")
        print(f"{'='*70}")
        print(f"Original TORSDOF:     {report.original_torsdof}")
        print(f"Optimized TORSDOF:    {report.optimized_torsdof}  "
              f"(-{report.original_torsdof - report.optimized_torsdof})")
        print(f"{'='*70}")
        
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
        
        if report.flexible_optimization:
            fo = report.flexible_optimization
            print(f"Smart Optimization:")
            print(f"  ✓ Frozen by scoring:    {fo.get('frozen_by_score', 0)}")
            print(f"  ✓ Kept flexible:        {fo.get('kept_branches', 0)}")
            print(f"  ✓ Terminal preserved:   {fo.get('terminal_kept', 0)}")
            print(f"{'='*70}")
        
        if report.warnings:
            print(f"⚠️  Warnings:")
            for warning in report.warnings:
                print(f"  - {warning}")
            print(f"{'='*70}")
        
        if report.recommendations:
            print(f"✅ Recommendations:")
            for rec in report.recommendations:
                print(f"  - {rec}")
            print(f"{'='*70}\n")
    
    @staticmethod
    def save_json_report(report: OptimizationReport, output_file: str):
        """Save JSON report"""
        report_dict = asdict(report)
        
        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2)


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='AutoPre Ligand Advanced v2.0 - Fixed and Improved',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python autopre_ligand_advanced.py ligand.pdbqt
  python autopre_ligand_advanced.py ligand.pdbqt --target 8
  python autopre_ligand_advanced.py *.pdbqt --output optimized/
        """
    )
    
    parser.add_argument('input', nargs='+', help='Input PDBQT file(s)')
    parser.add_argument('--target', type=int, default=10, help='Target TORSDOF (default: 10)')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer = LigandOptimizer(verbose=not args.quiet)
    reporter = OptimizationReporter()
    
    for input_file in args.input:
        input_path = Path(input_file)
        
        if not input_path.exists():
            print(f"Warning: {input_file} not found, skipping")
            continue
        
        if output_dir:
            output_file = str(output_dir / f"{input_path.stem}_optimized.pdbqt")
        else:
            output_file = str(input_path.parent / f"{input_path.stem}_optimized.pdbqt")
        
        try:
            optimized_pdbqt, report = optimizer.optimize_to_target_torsdof(
                str(input_path),
                target_torsdof=args.target,
                pocket_center=None,
                output_file=output_file
            )
            
            if not args.quiet:
                reporter.print_summary(report, input_path.stem)
        
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
