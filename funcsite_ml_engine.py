#!/usr/bin/env python3
"""
FuncSite-ML Engine: Enzyme Functional Site Prediction
======================================================

Dual ML models for identifying:
1. Catalytic hotspots (activity-critical residues)
2. Specificity sites (selectivity-determining residues)

"""

import os
import re
import sys
import warnings
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, roc_curve, auc,
                             confusion_matrix, classification_report)
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

CHARGED_POS = {'LYS', 'ARG', 'HIS'}
CHARGED_NEG = {'ASP', 'GLU'}
CHARGED_ALL = CHARGED_POS | CHARGED_NEG

HYDROPHOBIC_RES = {
    'ALA', 'VAL', 'LEU', 'ILE', 'MET',
    'PHE', 'TRP', 'PRO', 'TYR', 'CYS',
}

HALOGENS = {'F', 'CL', 'BR', 'I'}

POLAR_RECEPTOR       = set('NOS')             
POLAR_LIGAND         = set('NOFS')          
HYDROPHOBIC_RECEPTOR = set('CS')            
HYDROPHOBIC_LIGAND   = {'C', 'S', 'F',        
                         'CL', 'BR', 'I'}

HBOND_CUTOFF       = 3.5   # polar / H-bond
IONIC_CUTOFF       = 4.0   # salt bridge
HYDROPHOBIC_CUTOFF = 4.5   # hydrophobic stacking
HALOGEN_CUTOFF     = 4.0   # halogen bond / halogen–polar


class FuncSiteMLEngine:

    def __init__(self, output_dir='funcsite_results', verbose=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.verbose = verbose

        self.docking_results = []
        self.interaction_data = None
        self.residue_features = None
        self.ml_models = {}

        self._setup_plot_style()

        if self.verbose:
            print("✓ FuncSite-ML Engine initialized")

    def _setup_plot_style(self):
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 10,
            'figure.dpi': 300,
            'axes.linewidth': 1.0,
            'savefig.bbox': 'tight',
        })

    def _print(self, message):
        if self.verbose:
            print(message)

    @staticmethod
    def _get_element(atom_name: str) -> str:
        """
        Robustly extract the element symbol from a PDB/PDBQT atom name.

        Rules
        -----
        1. Strip leading/trailing whitespace and embedded digits.
        2. Check for two-character element symbols that appear in
           biological structures (Cl, Br, Mg, Zn, Ca, Fe, Mn …).
        3. Fall back to the first remaining character (upper-cased).

        Returns 'X' when the name is empty or unparseable.
        """
        TWO_CHAR = {'CL', 'BR', 'MG', 'ZN', 'CA', 'FE', 'MN',
                    'NA', 'CU', 'NI', 'CO', 'SE', 'SI'}

        name = re.sub(r'\d', '', atom_name.strip()).strip().upper()
        if not name:
            return 'X'
        if len(name) >= 2 and name[:2] in TWO_CHAR:
            return name[:2]
        return name[0]

    def parse_docking_results(self, pdbqt_files):
        self._print("\n[1/6] 📊 Parsing docking results...")

        results = []

        if isinstance(pdbqt_files, dict):
            for filename, content in pdbqt_files.items():
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                results.extend(self._parse_pdbqt_content(filename, content))
        else:
            for file_path in pdbqt_files:
                with open(file_path, 'r') as f:
                    content = f.read()
                results.extend(
                    self._parse_pdbqt_content(Path(file_path).name, content)
                )

        if not results:
            raise ValueError("No valid docking results found")

        df = pd.DataFrame(results)
        self.docking_results = df
        self._print(f"   ✓ Parsed {len(df)} docking poses")
        return df

    def _parse_pdbqt_content(self, filename, content):
        results = []
        current_model = None

        for line in content.split('\n'):
            if line.startswith('MODEL'):
                try:
                    current_model = int(line.split()[1])
                except (IndexError, ValueError):
                    current_model = None

            elif line.startswith('REMARK VINA RESULT:'):
                try:
                    parts = line.strip().split()
                    results.append({
                        'file':    filename,
                        'model':   current_model,
                        'score':   float(parts[3]),
                        'rmsd_lb': float(parts[4]),
                        'rmsd_ub': float(parts[5]),
                    })
                except (IndexError, ValueError):
                    continue

        return results

    def extract_residue_interactions(self, pdbqt_files, receptor_file):
        self._print("\n[2/6] 🔬 Extracting residue interactions...")

        if isinstance(receptor_file, (str, Path)):
            with open(receptor_file, 'r') as f:
                receptor_content = f.read()
        else:
            receptor_content = (receptor_file.decode('utf-8')
                                if isinstance(receptor_file, bytes)
                                else receptor_file)

        receptor_residues = self._parse_receptor(receptor_content)
        self._print(f"   ✓ Extracted {len(receptor_residues)} receptor residues")

        interactions = []

        if isinstance(pdbqt_files, dict):
            file_list = pdbqt_files.items()
        else:
            file_list = [(Path(f).name, open(f).read()) for f in pdbqt_files]

        for filename, content in file_list:
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            interactions.extend(
                self._analyze_docking_content(filename, content, receptor_residues)
            )

        if not interactions:
            raise ValueError("No interactions found")

        df = pd.DataFrame(interactions)
        self.interaction_data = df

        output_path = self.output_dir / 'residue_interactions.csv'
        df.to_csv(output_path, index=False)

        self._print(f"   ✓ Found {len(df)} residue interactions")
        self._print(f"   ✓ Saved to {output_path}")
        return df

    def _parse_receptor(self, content):
        residues = {}

        for line in content.split('\n'):
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    atom_name = line[12:16].strip()
                    res_name  = line[17:20].strip()
                    res_num   = line[22:26].strip()
                    chain     = line[21:22].strip() or 'A'

                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())

                    res_id = f"{res_name}{res_num}{chain}"

                    if res_id not in residues:
                        residues[res_id] = {
                            'res_name': res_name,
                            'res_num':  res_num,
                            'chain':    chain,
                            'atoms':    {},
                        }
                    residues[res_id]['atoms'][atom_name] = [x, y, z]

                except (IndexError, ValueError):
                    continue

        return residues

    def _analyze_docking_content(self, filename, content, receptor_residues,
                                 cutoff=4.0):
        interactions = []
        lines = content.split('\n')
        current_model = None
        model_score   = None
        ligand_atoms  = []
        reading_model = False

        for line in lines:
            if line.startswith('MODEL'):
                try:
                    current_model = int(line.split()[1])
                    ligand_atoms  = []
                    model_score   = None
                    reading_model = True
                except (IndexError, ValueError):
                    reading_model = False

            elif line.startswith('REMARK VINA RESULT:') and reading_model:
                try:
                    model_score = float(line.split()[3])
                except (IndexError, ValueError):
                    pass

            elif line.startswith('HETATM') and reading_model:
                try:
                    atom_name = line[12:16].strip()
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    atom_type = line[77:79].strip() if len(line) >= 79 else ''

                    ligand_atoms.append({
                        'name':   atom_name,
                        'coords': [x, y, z],
                        'type':   atom_type,
                    })
                except (IndexError, ValueError):
                    continue

            elif line.startswith('ENDMDL') and reading_model:
                if current_model and model_score and ligand_atoms:
                    for res_id, residue in receptor_residues.items():
                        interaction = self._calculate_interaction(
                            residue, res_id, ligand_atoms, cutoff
                        )
                        if interaction['has_contact']:
                            interactions.append({
                                'file':              filename,
                                'model':             current_model,
                                'score':             model_score,
                                'residue_id':        res_id,
                                'res_name':          residue['res_name'],
                                'res_num':           residue['res_num'],
                                'chain':             residue['chain'],
                                'min_distance':      interaction['min_distance'],
                                'contact_count':     interaction['contact_count'],
                                'h_bond_count':      interaction['h_bond_count'],
                                'ionic_count':       interaction['ionic_count'],
                                'hydrophobic_count': interaction['hydrophobic_count'],
                                'halogen_count':     interaction['halogen_count'],
                                'energy_contribution': interaction['energy_contribution'],
                            })
                reading_model = False

        return interactions

    def _calculate_interaction(self, residue, res_id, ligand_atoms, cutoff=4.5):

        res_name   = residue['res_name'].upper()
        is_charged = res_name in CHARGED_ALL

        min_distance      = float('inf')
        contact_count     = 0
        polar_count       = 0
        ionic_count       = 0
        hydrophobic_count = 0
        halogen_count     = 0
        energy_contribution = 0.0

        for res_atom_name, res_coords in residue['atoms'].items():
            elem_r = self._get_element(res_atom_name)

            for lig_atom in ligand_atoms:
                raw_type = lig_atom['type'] if lig_atom['type'] else lig_atom['name']
                elem_l   = self._get_element(raw_type)

                rc, lc = res_coords, lig_atom['coords']
                dist = np.sqrt(
                    (rc[0] - lc[0]) ** 2 +
                    (rc[1] - lc[1]) ** 2 +
                    (rc[2] - lc[2]) ** 2
                )
                if dist < min_distance:
                    min_distance = dist

                # Each atom pair is classified into at most one contact type
                # (priority: ionic > polar > halogen > hydrophobic)
                # to avoid double-counting contact_count.
                classified  = False
                is_polar_pair = False

                # ── Ionic contacts (salt bridge) ──────────────────────────
                if not classified and dist <= IONIC_CUTOFF and is_charged:
                    if (res_name in CHARGED_POS and elem_l in {'O', 'S'}) or \
                       (res_name in CHARGED_NEG and elem_l == 'N'):
                        ionic_count  += 1
                        contact_count += 1
                        classified    = True
                        is_polar_pair = True   # ionic pairs get energy boost

                # ── Polar / H-bond contacts ───────────────────────────────
                if not classified and dist <= HBOND_CUTOFF:
                    if elem_r in POLAR_RECEPTOR and elem_l in POLAR_LIGAND:
                        polar_count   += 1
                        contact_count += 1
                        classified    = True
                        is_polar_pair = True

                # ── Halogen contacts ──────────────────────────────────────
                if not classified and dist <= HALOGEN_CUTOFF:
                    if elem_l in HALOGENS and elem_r in POLAR_RECEPTOR:
                        halogen_count += 1
                        contact_count += 1
                        classified    = True
                        is_polar_pair = True   # polar-like energy scaling

                # ── Hydrophobic contacts ──────────────────────────────────
                if not classified and dist <= HYDROPHOBIC_CUTOFF:
                    if (res_name in HYDROPHOBIC_RES
                            and elem_r in HYDROPHOBIC_RECEPTOR
                            and elem_l in HYDROPHOBIC_LIGAND):
                        hydrophobic_count += 1
                        contact_count     += 1
                        classified         = True

                # ── Energy contribution ───────────────────────────────────
                if dist <= HYDROPHOBIC_CUTOFF:           # widest cutoff
                    if dist < 2.0:
                        energy_contribution += 5.0       # steric clash
                    else:
                        contrib = -1.0 / (dist + 0.5)
                        if is_polar_pair or is_charged:  # polar / ionic boost
                            contrib *= 1.5
                        energy_contribution += contrib

        if min_distance == float('inf'):
            min_distance = 999.0

        return {
            'has_contact':         contact_count > 0,
            'min_distance':        min_distance,
            'contact_count':       contact_count,
            'h_bond_count':        polar_count,
            'ionic_count':         ionic_count,
            'hydrophobic_count':   hydrophobic_count,
            'halogen_count':       halogen_count,
            'energy_contribution': energy_contribution,
        }

    def prepare_ml_features(self, interaction_df=None):
        if interaction_df is None:
            interaction_df = self.interaction_data
        if interaction_df is None or len(interaction_df) == 0:
            raise ValueError("No interaction data available")

        self._print("\n[3/6] 🧪 Engineering ML features...")

        features = interaction_df.groupby('residue_id').agg({
            'min_distance':       'mean',
            'contact_count':      'sum',
            'h_bond_count':       'sum',
            'ionic_count':        'sum',
            'hydrophobic_count':  'sum',
            'halogen_count':      'sum',
            'energy_contribution':'sum',
            'score':              'mean',
            'model':              'count',
        }).reset_index()

        features = features.rename(columns={'model': 'frequency'})

        features['contact_ratio']     = features['contact_count']     / features['frequency']
        features['h_bond_ratio']      = features['h_bond_count']      / features['frequency']
        features['ionic_ratio']       = features['ionic_count']        / features['frequency']
        features['hydrophobic_ratio'] = features['hydrophobic_count'] / features['frequency']
        features['halogen_ratio']     = features['halogen_count']     / features['frequency']
        features['energy_per_contact']= (features['energy_contribution']
                                         / features['contact_count'].replace(0, 1))

        features['res_type'] = features['residue_id'].str[:3]

        features['is_hydrophobic']  = features['res_type'].isin(
            ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'])
        features['is_polar']        = features['res_type'].isin(
            ['SER', 'THR', 'CYS', 'ASN', 'GLN', 'TYR'])
        features['is_charged_pos']  = features['res_type'].isin(['LYS', 'ARG', 'HIS'])
        features['is_charged_neg']  = features['res_type'].isin(['ASP', 'GLU'])

        features['rel_frequency'] = features['frequency'] / features['frequency'].max()

        e_min = features['energy_contribution'].min()
        e_max = features['energy_contribution'].max()
        e_rng = e_max - e_min
        features['rel_energy'] = (
            (features['energy_contribution'] - e_min) / e_rng if e_rng > 0 else 0
        )

        self.residue_features = features
        self._print(f"   ✓ Generated features for {len(features)} residues")
        return features

    def calculate_specificity_scores(self, interaction_df=None,
                                     residue_features=None):
        if interaction_df is None:
            interaction_df = self.interaction_data
        if residue_features is None:
            residue_features = self.residue_features

        res_model_energy = (interaction_df
                            .groupby(['residue_id', 'file', 'model'])['energy_contribution']
                            .mean()
                            .reset_index())

        res_counts    = res_model_energy.groupby('residue_id').size()
        valid_residues = res_counts[res_counts >= 2].index
        res_model_energy = res_model_energy[
            res_model_energy['residue_id'].isin(valid_residues)
        ]

        specificity = (res_model_energy
                       .groupby('residue_id')
                       .agg(energy_std=('energy_contribution', 'std'),
                            energy_iqr=('energy_contribution',
                                        lambda x: x.quantile(0.75) - x.quantile(0.25)))
                       .reset_index()
                       .fillna(0))

        specificity['variation'] = (0.5 * specificity['energy_std']
                                    + 0.5 * specificity['energy_iqr'])

        res_freq = (res_model_energy
                    .groupby('residue_id').size()
                    .reset_index(name='model_count'))
        specificity = pd.merge(specificity, res_freq, on='residue_id')

        cnt_min = specificity['model_count'].min()
        cnt_max = specificity['model_count'].max()
        specificity['model_count_norm'] = (
            (specificity['model_count'] - cnt_min) / (cnt_max - cnt_min + 1e-6)
        )

        max_var = specificity['variation'].max()
        if max_var > 0:
            specificity['variation_norm']    = specificity['variation'] / max_var
            specificity['specificity_score'] = (0.8 * specificity['variation_norm']
                                                + 0.2 * specificity['model_count_norm'])
        else:
            specificity['specificity_score'] = 0.0

        spec_scores = pd.Series(
            index=residue_features['residue_id'], data=0.0
        )
        for _, row in specificity.iterrows():
            if row['residue_id'] in spec_scores.index:
                spec_scores[row['residue_id']] = row['specificity_score']

        return spec_scores.values

    def train_models(self, residue_features=None):
        if residue_features is None:
            residue_features = self.residue_features
        if residue_features is None or len(residue_features) < 10:
            raise ValueError("Insufficient data for model training")

        self._print("\n[4/6] 🤖 Training dual ML models...")

        feature_cols = [
            'frequency', 'min_distance', 'contact_count', 'h_bond_count',
            'ionic_count', 'hydrophobic_count', 'halogen_count',
            'energy_contribution',
            'contact_ratio', 'h_bond_ratio', 'ionic_ratio', 'hydrophobic_ratio',
            'halogen_ratio', 'energy_per_contact',
            'is_hydrophobic', 'is_polar',
            'is_charged_pos', 'is_charged_neg', 'rel_frequency', 'rel_energy',
        ]

        X        = residue_features[feature_cols].values
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        energy_threshold = np.percentile(residue_features['energy_contribution'], 15)
        freq_threshold   = np.percentile(residue_features['frequency'], 85)

        y_key = (
            (residue_features['energy_contribution'] <= energy_threshold) |
            (residue_features['frequency'] >= freq_threshold)
        ).astype(int)

        specificity_scores = self.calculate_specificity_scores(
            self.interaction_data, residue_features
        )
        residue_features['specificity_score'] = specificity_scores

        spec_threshold = np.percentile(specificity_scores, 85)
        y_spec = (specificity_scores >= spec_threshold).astype(int)

        rf_params = dict(
            n_estimators=80,
            max_depth=4,
            min_samples_split=8,
            random_state=42,
            class_weight='balanced',
        )
        cv_folds = min(10, len(X_scaled) // 2)

        self._print("   ↪ Training catalytic hotspot model...")
        rf_key = RandomForestClassifier(**rf_params)
        rf_key.fit(X_scaled, y_key)
        cv_key = cross_val_score(rf_key, X_scaled, y_key, cv=cv_folds)
        self._print(f"      ✓ CV Accuracy: {cv_key.mean():.3f} ± {cv_key.std():.3f}")

        self._print("   ↪ Training specificity model...")
        rf_spec = RandomForestClassifier(**rf_params)
        rf_spec.fit(X_scaled, y_spec)
        cv_spec = cross_val_score(rf_spec, X_scaled, y_spec, cv=cv_folds)
        self._print(f"      ✓ CV Accuracy: {cv_spec.mean():.3f} ± {cv_spec.std():.3f}")

        residue_features['catalytic_prob']   = rf_key.predict_proba(X_scaled)[:, 1]
        residue_features['specificity_prob'] = rf_spec.predict_proba(X_scaled)[:, 1]

        self.ml_models = {
            'catalytic':    rf_key,
            'specificity':  rf_spec,
            'scaler':       scaler,
            'feature_cols': feature_cols,
        }
        self.residue_features = residue_features

        self._print("   ✓ Model training complete")
        return self.ml_models, residue_features

    def generate_reports(self, residue_features=None):
        if residue_features is None:
            residue_features = self.residue_features

        self._print("\n[5/6] 📋 Generating reports...")

        catalytic    = residue_features.sort_values('catalytic_prob',   ascending=False)
        specificity  = residue_features.sort_values('specificity_prob', ascending=False)
        dual_function = residue_features[
            (residue_features['catalytic_prob']   > 0.7) &
            (residue_features['specificity_prob'] > 0.7)
        ].sort_values('catalytic_prob', ascending=False)

        catalytic.to_csv(self.output_dir / 'catalytic_hotspots.csv',  index=False)
        specificity.to_csv(self.output_dir / 'specificity_sites.csv', index=False)
        if len(dual_function) > 0:
            dual_function.to_csv(self.output_dir / 'dual_function_residues.csv',
                                 index=False)

        self._generate_text_report(catalytic, specificity, dual_function)

        self._print("   ✓ Reports saved")
        self._print(f"      • {self.output_dir}/catalytic_hotspots.csv")
        self._print(f"      • {self.output_dir}/specificity_sites.csv")
        if len(dual_function) > 0:
            self._print(f"      • {self.output_dir}/dual_function_residues.csv")
        self._print(f"      • {self.output_dir}/analysis_report.txt")

    def _generate_text_report(self, catalytic, specificity, dual_function):
        report_path = self.output_dir / 'analysis_report.txt'

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FUNCSITE-ML ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("INTERACTION CLASSIFICATION (Level-2 revision)\n")
            f.write("-" * 40 + "\n")
            f.write("  Polar    : receptor N/O/S – ligand N/O/S/F        cutoff 3.5 Å\n")
            f.write("  Ionic    : Lys/Arg/His(+)+lig-O/S | Asp/Glu(-)+lig-N\n")
            f.write("             (mutually exclusive with polar)         cutoff 4.0 Å\n")
            f.write("  Halogen  : ligand F/Cl/Br/I – receptor N/O/S      cutoff 4.0 Å\n")
            f.write("  Hydropho : hydrophobic-res C/S – ligand C/S/F/Cl/Br/I\n")
            f.write("             (residue-name filtered)                 cutoff 4.5 Å\n")
            f.write("  Priority : ionic > polar > halogen > hydrophobic\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total residues analyzed        : {len(catalytic)}\n")
            f.write(f"Catalytic hotspots (prob>0.70) : "
                    f"{len(catalytic[catalytic['catalytic_prob'] > 0.7])}\n")
            f.write(f"Specificity sites  (prob>0.70) : "
                    f"{len(specificity[specificity['specificity_prob'] > 0.7])}\n")
            f.write(f"Dual-function residues         : {len(dual_function)}\n\n")

            f.write("TOP 10 CATALYTIC HOTSPOTS\n")
            f.write("-" * 40 + "\n")
            for i, (_, row) in enumerate(catalytic.head(10).iterrows(), 1):
                f.write(f"{i:2d}. {row['residue_id']:>8}  "
                        f"Prob: {row['catalytic_prob']:.3f}  "
                        f"Energy: {row['energy_contribution']:7.2f}\n")

            f.write("\nTOP 10 SPECIFICITY SITES\n")
            f.write("-" * 40 + "\n")
            for i, (_, row) in enumerate(specificity.head(10).iterrows(), 1):
                f.write(f"{i:2d}. {row['residue_id']:>8}  "
                        f"Prob: {row['specificity_prob']:.3f}  "
                        f"Score: {row['specificity_score']:.3f}\n")

            if len(dual_function) > 0:
                f.write("\nDUAL-FUNCTION RESIDUES\n")
                f.write("-" * 40 + "\n")
                for i, (_, row) in enumerate(dual_function.iterrows(), 1):
                    f.write(f"{i:2d}. {row['residue_id']:>8}  "
                            f"Cat: {row['catalytic_prob']:.3f}  "
                            f"Spec: {row['specificity_prob']:.3f}\n")

            f.write("\n" + "=" * 80 + "\n")

    def package_results(self, output_zip='funcsite_results.zip'):
        self._print("\n[6/6] 📦 Packaging results...")

        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.output_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.output_dir.parent)
                    zipf.write(file_path, arcname)

        self._print(f"   ✓ Results packaged: {output_zip}")
        return output_zip

if __name__ == "__main__":
    print("FuncSite-ML Engine")
    print("This module is designed to be imported, not run directly.")
    print("Use from Google Colab or Python script.")
