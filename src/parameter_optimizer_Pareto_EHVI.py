"""
Requires Python >=3.9, numpy, scipy, scikit-learn and an external Vina CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Sequence
from uuid import uuid4

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

PARAMETERS = ('center_x', 'center_y', 'center_z', 'size_x', 'size_y',
              'size_z', 'exhaustiveness')


def pareto_mask(values: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if y.ndim != 2 or not np.isfinite(y).all():
        raise ValueError('Expected finite 2D objectives')
    return np.array([not np.any(np.all(y <= row, axis=1) &
                               np.any(y < row, axis=1)) for row in y])


def hypervolume(values: np.ndarray, reference: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)
    if ref.ndim != 1 or not len(ref) or not np.isfinite(ref).all():
        raise ValueError('Invalid reference')
    y = np.asarray(values, dtype=float).reshape(-1, len(ref))
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite objectives')
    y = y[np.all(y < ref, axis=1)]
    if not len(y):
        return 0.0
    y = np.unique(y[pareto_mask(y)], axis=0)
    if len(ref) == 1:
        return float(ref[0] - y[:, 0].min())
    edges = np.r_[np.unique(y[:, 0]), ref[0]]
    return float(sum((right-left) * hypervolume(y[y[:, 0] <= left, 1:], ref[1:])
                     for left, right in zip(edges[:-1], edges[1:])))


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
    """Analytic EI for minimization."""
    mu, sigma = np.asarray(mu), np.asarray(sigma)
    delta = best - mu
    safe = np.maximum(sigma, 1e-15)
    z = delta / safe
    ei = delta * ndtr(z) + safe * np.exp(-0.5*z*z) / np.sqrt(2*np.pi)
    return np.where(sigma > 1e-15, ei, np.maximum(delta, 0.0))


def sampled_ehvi(mu: np.ndarray, sigma: np.ndarray, front: np.ndarray,
                 reference: np.ndarray, normal_samples: np.ndarray) -> np.ndarray:
    base = hypervolume(front, reference)
    return np.array([np.mean([
        max(0.0, hypervolume(np.vstack((front, draw)), reference)-base)
        for draw in mean + normal_samples*std])
        for mean, std in zip(mu, sigma)])


def read_coordinates(path: Path) -> np.ndarray:
    coords = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.startswith(('ATOM  ', 'HETATM')):
            try:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError as exc:
                raise ValueError(f'Malformed coordinates: {path}') from exc
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or not len(arr) or not np.isfinite(arr).all():
        raise ValueError(f'No finite atoms: {path}')
    return arr


def parse_vina_pose(path: Path) -> float:
    text = path.read_text(encoding='utf-8', errors='replace')
    scores = []
    for block in re.split(r'^MODEL\s+.*$', text, flags=re.M):
        match = re.search(r'^REMARK VINA RESULT:\s*([-+\d.eE]+)', block, re.M)
        if not match:
            continue
        value = float(match.group(1))
        atoms = [l for l in block.splitlines() if l.startswith(('ATOM  ', 'HETATM'))]
        if not atoms or not math.isfinite(value):
            raise ValueError('Scored pose lacks atoms or finite affinity')
        for line in atoms:
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            if not np.isfinite(xyz).all():
                raise ValueError('Nonfinite pose coordinates')
        scores.append(value)
    if not scores:
        raise ValueError('No scored Vina models')
    return min(scores)


class DockingParameterOptimizer:

    def __init__(self, receptor_file: str, ligand_file: Optional[str] = None,
                 initial_center: Optional[Sequence[float]] = None,
                 initial_size: Optional[Sequence[float]] = None,
                 output_dir: Optional[str] = None, *,
                 ligand_files: Optional[Sequence[str]] = None,
                 vina_executable: str = 'vina', seed: int = 42,
                 cpu_per_task: int = 1, ligand_workers: int = 1,
                 timeout: float = 300, num_modes: int = 9, energy_range: float = 3,
                 center_radius: float = 15, size_factors: Sequence[float] = (0.5, 1.5),
                 exhaustiveness_bounds: Sequence[int] = (8, 32),
                 search_bounds: Optional[dict] = None,
                 reference_point: Optional[Sequence[float]] = None,
                 allow_global_box: bool = False, gp_noise: float = 0.01,
                 candidate_count: int = 64, ehvi_samples: int = 32,
                 center_provenance: str = 'explicit pocket coordinates'):
        files = list(ligand_files or [])
        if ligand_file:
            files.insert(0, ligand_file)
        if not files:
            raise ValueError('At least one ligand is required')
        self.receptor = Path(receptor_file).resolve()
        self.ligands = [Path(f).resolve() for f in files]
        if len(set(self.ligands)) != len(self.ligands):
            raise ValueError('Duplicate ligand paths')
        for path in [self.receptor, *self.ligands]:
            if not path.is_file() or path.suffix.lower() != '.pdbqt':
                raise ValueError(f'Require prepared PDBQT: {path}')
            read_coordinates(path)
        if (not 0 < seed < 2**31 or cpu_per_task < 1 or ligand_workers < 1 or
                not np.isfinite([timeout, energy_range, gp_noise]).all() or
                timeout <= 0 or num_modes < 1 or energy_range <= 0 or gp_noise <= 0 or
                candidate_count < 2 or ehvi_samples < 2):
            raise ValueError('Invalid seed, execution budget, noise or sampling setting')
        if cpu_per_task * ligand_workers > (os.cpu_count() or 1):
            raise ValueError('Parallel CPU budget exceeds logical CPU count')
        self.seed, self.rng = seed, np.random.default_rng(seed)
        self.cpu_per_task, self.workers = cpu_per_task, ligand_workers
        self.timeout, self.num_modes, self.energy_range = timeout, num_modes, energy_range
        self.gp_noise, self.candidate_count, self.ehvi_samples = gp_noise, candidate_count, ehvi_samples
        self.vina_executable, self.version = vina_executable, None
        if initial_center is None or initial_size is None:
            if not allow_global_box or initial_center is not None or initial_size is not None:
                raise ValueError('Provide both pocket center and size; global fallback requires opt-in')
            xyz = read_coordinates(self.receptor)
            initial_center = (xyz.min(0) + xyz.max(0)) / 2
            initial_size = np.maximum(xyz.max(0) - xyz.min(0) + 10, 10)
            center_provenance = 'explicit receptor-only global box'
        self.center, self.size = np.asarray(initial_center, float), np.asarray(initial_size, float)
        if (self.center.shape != (3,) or self.size.shape != (3,) or
                not np.isfinite(np.r_[self.center, self.size]).all() or np.any(self.size <= 0)):
            raise ValueError('Need three finite center/size coordinates and positive sizes')
        if center_radius <= 0 or len(size_factors) != 2 or not 0 < size_factors[0] < size_factors[1]:
            raise ValueError('Invalid center radius or size factors')
        bounds = {k: (float(v-center_radius), float(v+center_radius))
                  for k, v in zip(PARAMETERS[:3], self.center)}
        bounds.update({k: (max(1.0, float(v*size_factors[0])), float(v*size_factors[1]))
                       for k, v in zip(PARAMETERS[3:6], self.size)})
        bounds['exhaustiveness'] = tuple(exhaustiveness_bounds)
        if search_bounds:
            if set(search_bounds) - set(PARAMETERS):
                raise ValueError('Only seven documented parameters may be searched')
            bounds.update(search_bounds)
        self.bounds = np.asarray([bounds[k] for k in PARAMETERS], float)
        if self.bounds.shape != (7, 2) or not np.isfinite(self.bounds).all() or np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError('Finite lower < upper required for each parameter')
        if np.any(self.bounds[3:, 0] <= 0) or np.any(self.bounds[6] != np.round(self.bounds[6])):
            raise ValueError('Need positive sizes and integer exhaustiveness bounds')
        self.search_space = dict(zip(PARAMETERS, self.bounds.tolist()))
        self.user_reference = None if reference_point is None else np.asarray(reference_point, float)
        if self.user_reference is not None and (self.user_reference.shape != (len(files),) or not np.isfinite(self.user_reference).all()):
            raise ValueError('Reference needs one finite affinity per ligand')
        root = Path(output_dir or 'parameter_optimization').resolve()
        self.output_dir = root / ('run_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8])
        self.output_dir.mkdir(parents=True)
        self.optimization_history: list[dict[str, Any]] = []
        self.cache: dict[tuple, dict] = {}
        self.best_parameters = self.best_score = None
        self.reference = self.offset = self.scale = None
        import scipy
        import sklearn
        self._write_json('manifest.json', {
            'implementation': 'reconstructed GP-EI / Sobol sampled EHVI, not historical reproduction',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'receptor': str(self.receptor), 'ligands': [str(p) for p in self.ligands],
            'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [self.receptor, *self.ligands]},
            'seed': seed, 'bounds': self.search_space, 'initial_center': self.center.tolist(),
            'initial_size': self.size.tolist(), 'center_provenance': center_provenance,
            'cpu_per_task': cpu_per_task, 'ligand_workers': ligand_workers, 'timeout': timeout,
            'num_modes': num_modes, 'energy_range': energy_range, 'gp_noise': gp_noise,
            'candidate_count': candidate_count, 'ehvi_samples': ehvi_samples,
            'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
            'reference_point': None if self.user_reference is None else self.user_reference.tolist(),
            'objective': 'minimum reported Vina affinity per ligand, all objectives minimized',
            'failure_policy': 'exclude incomplete vectors; preserve failures; no synthetic penalty'})

    def _write_json(self, name: str, value: Any) -> None:
        path = self.output_dir / name
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(path)

    def _check_vina(self) -> None:
        if self.version is not None:
            return
        executable = shutil.which(self.vina_executable)
        if executable is None:
            raise FileNotFoundError(f'Vina executable not found: {self.vina_executable}')
        self.vina_executable = executable
        result = subprocess.run([executable, '--version'], capture_output=True, text=True,
                                errors='replace', timeout=15, check=True)
        self.version = (result.stdout + result.stderr).strip()
        self._write_json('vina_version.json', {'executable': executable, 'version': self.version})

    def _canonical(self, parameters: dict) -> dict:
        values = np.array([parameters[k] for k in PARAMETERS], float)
        if not np.isfinite(values).all() or np.any(values < self.bounds[:, 0]) or np.any(values > self.bounds[:, 1]):
            raise ValueError('Candidate outside configured bounds')
        result = {k: float(v) for k, v in zip(PARAMETERS, values)}
        result['exhaustiveness'] = int(np.rint(values[6]))
        return result

    def _dock_one(self, index: int, parameters: dict, trial_dir: Path) -> dict:
        folder = trial_dir / f'ligand_{index:03d}'
        folder.mkdir()
        config, pose = folder / 'config.txt', folder / 'poses.pdbqt'
        config.write_text('\n'.join(f'{k} = {v}' for k, v in parameters.items()) +
                          f'\nnum_modes = {self.num_modes}\nenergy_range = {self.energy_range}\n', encoding='utf-8')
        seed = 1 + ((self.seed + index * 104729) % (2**31 - 2))
        cmd = [self.vina_executable, '--receptor', str(self.receptor), '--ligand',
               str(self.ligands[index]), '--config', str(config), '--out', str(pose),
               '--cpu', str(self.cpu_per_task), '--seed', str(seed)]
        record = {'ligand': str(self.ligands[index]), 'command': cmd, 'seed': seed,
                  'pose_file': str(pose), 'status': 'failed'}
        start = time.monotonic()
        try:
            with (folder / 'stdout.log').open('w', encoding='utf-8') as stdout, (folder / 'stderr.log').open('w', encoding='utf-8') as stderr:
                result = subprocess.run(cmd, stdout=stdout, stderr=stderr, timeout=self.timeout, check=False)
            record['returncode'] = result.returncode
            if result.returncode:
                raise RuntimeError(f'Vina exit code {result.returncode}')
            record.update(score=parse_vina_pose(pose), status='success')
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        record['elapsed_seconds'] = time.monotonic() - start
        (folder / 'result.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        return record

    def evaluate(self, parameters: dict) -> dict:
        """Real evaluation with caching and persistent complete/failed records."""
        self._check_vina()
        params = self._canonical(parameters)
        key = tuple(params[k] for k in PARAMETERS)
        if key in self.cache:
            return self.cache[key]
        folder = self.output_dir / f'trial_{len(self.optimization_history):04d}'
        folder.mkdir()
        with ThreadPoolExecutor(max_workers=min(self.workers, len(self.ligands))) as executor:
            results = list(executor.map(lambda i: self._dock_one(i, params, folder), range(len(self.ligands))))
        success = all(r['status'] == 'success' for r in results)
        record = {'trial': len(self.optimization_history), 'parameters': params,
                  'status': 'success' if success else 'failed', 'ligand_results': results,
                  'scores': [r['score'] for r in results] if success else None}
        self.optimization_history.append(record)
        self.cache[key] = record
        with (self.output_dir / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        return record

    def latin_hypercube_sampling(self, n_samples: int) -> list[dict]:
        if n_samples < 1:
            raise ValueError('n_samples must be positive')
        unit = qmc.LatinHypercube(7, seed=int(self.rng.integers(2**31))).random(n_samples)
        values = qmc.scale(unit, self.bounds[:, 0], self.bounds[:, 1])
        return [self._canonical(dict(zip(PARAMETERS, row))) for row in values]

    def _observations(self) -> tuple[np.ndarray, np.ndarray]:
        valid = [r for r in self.optimization_history if r['status'] == 'success']
        x = np.array([[r['parameters'][k] for k in PARAMETERS] for r in valid])
        y = np.array([r['scores'] for r in valid])
        return (x-self.bounds[:, 0]) / np.diff(self.bounds, axis=1).ravel(), y

    def _freeze_normalization(self, y: np.ndarray) -> None:
        self.offset = y.min(axis=0)
        self.scale = np.maximum(np.ptp(y, axis=0), 1.0)
        raw_ref = self.user_reference if self.user_reference is not None else y.max(0) + 0.1*self.scale
        if np.any(raw_ref <= y.min(0)):
            raise ValueError('Reference must be worse than initial ideal in every objective')
        self.reference = (raw_ref-self.offset) / self.scale
        self._write_json('objective_scaling.json', {'offset': self.offset.tolist(),
            'scale': self.scale.tolist(), 'raw_reference': raw_ref.tolist(),
            'policy': 'frozen after successful initialization'})

    def bayesian_optimization(self, n_iterations: int = 20, n_initial_points: int = 10) -> tuple[dict, Any]:
        """Sequential BO; evaluation budgets include failed attempts."""
        if self.optimization_history:
            raise RuntimeError('Use a fresh optimizer for an independent run')
        if n_iterations < 0 or n_initial_points < 2:
            raise ValueError('Require iterations >=0 and initial points >=2')
        self._check_vina()
        initial = self.latin_hypercube_sampling(n_initial_points)
        baseline = np.clip([*self.center, *self.size, self.bounds[6, 0]],
                           self.bounds[:, 0], self.bounds[:, 1])
        initial[0] = self._canonical(dict(zip(PARAMETERS, baseline)))
        for candidate in initial:
            self.evaluate(candidate)
        if sum(r['status'] == 'success' for r in self.optimization_history) < 2:
            raise RuntimeError('Fewer than two successful complete initial trials; inspect logs')
        _, y = self._observations()
        self._freeze_normalization(y)
        self._write_json('budget.json', {'initial_attempts': n_initial_points, 'iterations': n_iterations})
        for iteration in range(n_iterations):
            x, y = self._observations()
            normalized = (y-self.offset) / self.scale
            models = []
            for column in normalized.T:
                gp = GaussianProcessRegressor(kernel=ConstantKernel(1, (0.01, 100))*Matern(
                    length_scale=np.ones(7), length_scale_bounds=(0.02, 20), nu=2.5),
                    alpha=self.gp_noise, normalize_y=False, random_state=self.seed,
                    n_restarts_optimizer=1)
                gp.fit(x, column)
                models.append(gp)
            candidates = self.latin_hypercube_sampling(self.candidate_count)
            candidates = [p for p in candidates if tuple(p[k] for k in PARAMETERS) not in self.cache]
            if not candidates:
                raise RuntimeError('Candidate pool exhausted')
            cx = np.array([[p[k] for k in PARAMETERS] for p in candidates])
            cx = (cx-self.bounds[:, 0]) / np.diff(self.bounds, axis=1).ravel()
            predictions = [gp.predict(cx, return_std=True) for gp in models]
            mu = np.column_stack([p[0] for p in predictions])
            sigma = np.column_stack([p[1] for p in predictions])
            if len(self.ligands) == 1:
                acquisition = expected_improvement(mu[:, 0], sigma[:, 0], normalized[:, 0].min())
            else:
                sobol = qmc.Sobol(len(self.ligands), scramble=True, seed=self.seed+iteration)
                u = sobol.random_base2(int(math.ceil(math.log2(self.ehvi_samples))))[:self.ehvi_samples]
                normals = ndtri(np.clip(u, 1e-12, 1-1e-12))
                acquisition = sampled_ehvi(mu, sigma, normalized[pareto_mask(normalized)],
                                           self.reference, normals)
            chosen = int(np.argmax(acquisition))
            record = self.evaluate(candidates[chosen])
            with (self.output_dir / 'acquisition.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'iteration': iteration, 'trial': record['trial'],
                    'acquisition': float(acquisition[chosen]), 'candidate_count': len(candidates)}) + '\n')
            print(f'Iteration {iteration+1}/{n_iterations}: {record["status"]}', flush=True)
        self._select_result()
        self.generate_optimized_config()
        self.generate_optimization_report()
        return self.best_parameters, self.best_score

    def _select_result(self) -> None:
        valid = [r for r in self.optimization_history if r['status'] == 'success']
        y = np.array([r['scores'] for r in valid])
        if self.offset is None:
            self._freeze_normalization(y)
        mask = pareto_mask(y)
        indices = np.flatnonzero(mask)
        distances = (y[mask] - y.min(0)) / self.scale
        order = np.lexsort((distances.mean(1), distances.max(1)))
        selected = valid[int(indices[order[0]])]
        self.best_parameters = selected['parameters']
        self.best_score = selected['scores'][0] if len(self.ligands) == 1 else selected['scores']
        self._write_json('pareto_front.json', [valid[int(i)] for i in indices])
        self._write_json('selected_result.json', {
            'selection': 'normalized Chebyshev to observed ideal, mean tie-break',
            'result': selected,
            'hypervolume_normalized': hypervolume((y[mask]-self.offset)/self.scale, self.reference)})

    def generate_optimized_config(self) -> str:
        if self.best_parameters is None:
            raise RuntimeError('No successful optimization')
        path = self.output_dir / 'optimized_config.conf'
        path.write_text('# Shared box; dock ligands separately\n' +
            '\n'.join(f'{k} = {v}' for k, v in self.best_parameters.items()) +
            f'\nnum_modes = {self.num_modes}\nenergy_range = {self.energy_range}\n', encoding='utf-8')
        return str(path)

    def generate_optimization_report(self) -> str:
        path = self.output_dir / 'optimization_report.json'
        baseline = self.optimization_history[0] if self.optimization_history else None
        baseline_scores = baseline['scores'] if baseline else None
        # Positive values mean lower affinity than the explicit first trial.
        improvement = (np.asarray(baseline_scores) - np.atleast_1d(self.best_score)).tolist() if baseline_scores is not None and self.best_score is not None else None
        self._write_json(path.name, {'best_parameters': self.best_parameters, 'best_score': self.best_score,
            'first_trial_scores': baseline_scores, 'affinity_decrease_vs_first_trial': improvement,
            'trials': len(self.optimization_history),
            'failed_trials': sum(r['status'] != 'success' for r in self.optimization_history),
            'acquisition': 'GP-EI' if len(self.ligands) == 1 else 'GP-QMC-EHVI (independent objectives)',
            'limitations': ['Reconstructed code, historical results not reproduced',
                'Vina scores are not catalytic activity or measured binding free energies',
                'Failed vectors excluded; feasibility is not modeled',
                'Candidate-pool acquisition search; approximate EHVI; fixed-seed evaluations',
                'Pocket detection, DBSCAN and chemical preparation are external inputs']})
        return str(path)

    def visualize_optimization(self) -> str:
        """Plot observed affinities; failed trials remain gaps in trial indices."""
        import matplotlib.pyplot as plt
        valid = [r for r in self.optimization_history if r['status'] == 'success']
        if not valid:
            raise RuntimeError('No successful observations')
        fig, ax = plt.subplots()
        for i, ligand in enumerate(self.ligands):
            ax.plot([r['trial'] for r in valid], [r['scores'][i] for r in valid],
                    'o', label=f'{i}: {ligand.stem}')
        ax.set(xlabel='Trial index (failed trials omitted)', ylabel='Vina affinity (kcal/mol)')
        ax.legend()
        fig.tight_layout()
        path = self.output_dir / 'optimization_analysis.png'
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return str(path)

    def grid_search(self, n_points_per_dim: int = 2, max_evaluations: int = 256) -> tuple[dict, Any]:
        """Budget-guarded real grid over the same seven parameters."""
        if n_points_per_dim < 2 or n_points_per_dim**7 > max_evaluations:
            raise ValueError('Invalid grid size or evaluation budget exceeded')
        if self.optimization_history:
            raise RuntimeError('Use a fresh optimizer for grid comparison')
        for row in itertools.product(*[np.linspace(lo, hi, n_points_per_dim) for lo, hi in self.bounds]):
            self.evaluate(dict(zip(PARAMETERS, row)))
        if not any(r['status'] == 'success' for r in self.optimization_history):
            raise RuntimeError('All grid trials failed')
        self._select_result()
        self.generate_optimized_config()
        self.generate_optimization_report()
        return self.best_parameters, self.best_score


def optimize_docking_parameters(receptor_file: str, ligand_file: Optional[str] = None,
                               method: str = 'bayesian', n_iterations: int = 20,
                               output_dir: Optional[str] = None, *, n_initial_points: int = 10,
                               **kwargs: Any) -> dict:
    """Compatibility wrapper forwarding pocket inputs and execution settings."""
    optimizer = None
    try:
        optimizer = DockingParameterOptimizer(receptor_file, ligand_file,
                                             output_dir=output_dir, **kwargs)
        if method == 'bayesian':
            params, score = optimizer.bayesian_optimization(n_iterations, n_initial_points)
        elif method == 'grid':
            params, score = optimizer.grid_search()
        else:
            raise ValueError(f'Unknown method: {method}')
        return {'success': True, 'best_parameters': params, 'best_score': score,
                'config_path': optimizer.generate_optimized_config(),
                'report_path': optimizer.generate_optimization_report(), 'optimizer': optimizer}
    except Exception as exc:
        return {'success': False, 'error': f'{type(exc).__name__}: {exc}',
                'output_dir': str(optimizer.output_dir) if optimizer else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receptor', required=True)
    parser.add_argument('--ligands', nargs='+', required=True)
    parser.add_argument('--center', nargs=3, type=float, required=True)
    parser.add_argument('--size', nargs=3, type=float, required=True)
    parser.add_argument('--vina', default='vina')
    parser.add_argument('--output', default='parameter_optimization')
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--initial-points', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--reference', nargs='+', type=float)
    parser.add_argument('--candidates', type=int, default=64)
    parser.add_argument('--ehvi-samples', type=int, default=32)
    parser.add_argument('--num-modes', type=int, default=9)
    parser.add_argument('--energy-range', type=float, default=3)
    parser.add_argument('--center-radius', type=float, default=15)
    parser.add_argument('--gp-noise', type=float, default=0.01)
    parser.add_argument('--exhaustiveness-bounds', nargs=2, type=int, default=(8, 32))
    args = parser.parse_args()
    result = optimize_docking_parameters(args.receptor, ligand_files=args.ligands,
        initial_center=args.center, initial_size=args.size, output_dir=args.output,
        vina_executable=args.vina, n_iterations=args.iterations, n_initial_points=args.initial_points,
        seed=args.seed, cpu_per_task=args.cpu, ligand_workers=args.workers, timeout=args.timeout,
        reference_point=args.reference, candidate_count=args.candidates, ehvi_samples=args.ehvi_samples,
        num_modes=args.num_modes, energy_range=args.energy_range, center_radius=args.center_radius,
        gp_noise=args.gp_noise, exhaustiveness_bounds=args.exhaustiveness_bounds)
    print(json.dumps({k: v for k, v in result.items() if k != 'optimizer'}, indent=2))
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
