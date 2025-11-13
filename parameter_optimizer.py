import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import concurrent.futures
import time
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt

from .environment import get_environment

logger = logging.getLogger(__name__)


class DockingParameterOptimizer:
    """Advanced molecular docking parameter optimizer"""
    
    def __init__(self, receptor_file: str, ligand_file: str = None,
                 initial_center: Optional[Tuple[float, float, float]] = None,
                 initial_size: Optional[Tuple[float, float, float]] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize the parameter optimizer
        
        Args:
            receptor_file: Path to receptor PDBQT file
            ligand_file: Path to ligand PDBQT file
            initial_center: Initial center coordinates (x, y, z)
            initial_size: Initial box size (size_x, size_y, size_z)
            output_dir: Output directory for optimization results
        """
        self.env = get_environment()
        self.output_dir = Path(output_dir) if output_dir else self.env.get_work_dir() / "parameter_optimization"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.receptor_file = receptor_file
        self.ligand_file = ligand_file
        self.optimization_history = []
        self.best_parameters = None
        self.best_score = None
        
        # Parse molecular files
        self.receptor_atoms = self._parse_pdb(receptor_file)
        if ligand_file:
            self.ligand_atoms = self._parse_pdb(ligand_file)
        else:
            self.ligand_atoms = []
        
        # Calculate initial parameters
        if initial_center is None or initial_size is None:
            self._calculate_initial_parameters()
        else:
            self.center = initial_center
            self.size = initial_size
        
        # Define search space
        self._define_search_space()
        
        logger.info(f"🎯 Parameter optimizer initialized")
        logger.info(f"   Receptor: {len(self.receptor_atoms)} atoms")
        logger.info(f"   Ligand: {len(self.ligand_atoms)} atoms")
        logger.info(f"   Initial center: {self.center}")
        logger.info(f"   Initial size: {self.size}")
    
    def _parse_pdb(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse PDB/PDBQT file and extract atomic information"""
        atoms = []
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        try:
                            atom_info = {
                                'atom_name': line[12:16].strip(),
                                'residue_name': line[17:20].strip(),
                                'x': float(line[30:38].strip()),
                                'y': float(line[38:46].strip()),
                                'z': float(line[46:54].strip())
                            }
                            atoms.append(atom_info)
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            logger.warning(f"Error parsing file {file_path}: {str(e)}")
        
        return atoms
    
    def _calculate_initial_parameters(self):
        """Calculate initial docking box parameters from molecular coordinates"""
        if not self.receptor_atoms:
            logger.warning("No receptor atoms found, using default parameters")
            self.center = (0, 0, 0)
            self.size = (20, 20, 20)
            return
        
        try:
            # Extract coordinates
            receptor_coords = np.array([
                [atom['x'], atom['y'], atom['z']] for atom in self.receptor_atoms
            ])
            
            if self.ligand_atoms:
                ligand_coords = np.array([
                    [atom['x'], atom['y'], atom['z']] for atom in self.ligand_atoms
                ])
                all_coords = np.vstack((receptor_coords, ligand_coords))
            else:
                all_coords = receptor_coords
            
            # Calculate center and size
            min_coords = np.min(all_coords, axis=0)
            max_coords = np.max(all_coords, axis=0)
            
            center = (min_coords + max_coords) / 2
            size = max_coords - min_coords + 10  # 5Å padding on each side
            
            self.center = tuple(center)
            self.size = tuple(np.maximum(size, 10.0))  # Minimum 10Å
            
        except Exception as e:
            logger.warning(f"Error calculating initial parameters: {str(e)}")
            self.center = (0, 0, 0)
            self.size = (20, 20, 20)
    
    def _define_search_space(self):
        """Define parameter search space boundaries"""
        center_x, center_y, center_z = self.center
        size_x, size_y, size_z = self.size
        
        padding = 15.0  # Search radius around initial center
        
        self.search_space = {
            # Center coordinates
            'center_x': (center_x - padding, center_x + padding),
            'center_y': (center_y - padding, center_y + padding),
            'center_z': (center_z - padding, center_z + padding),
            
            # Box dimensions
            'size_x': (max(10, size_x * 0.5), size_x * 1.5),
            'size_y': (max(10, size_y * 0.5), size_y * 1.5),
            'size_z': (max(10, size_z * 0.5), size_z * 1.5),
            
            # Algorithm parameters
            'exhaustiveness': (8, 32),
            'num_modes': (5, 20),
            'energy_range': (2, 5)
        }
    
    def mock_docking_score(self, parameters: Dict[str, float]) -> float:
        """
        Simulate docking score calculation
        Replace this with actual Vina docking in production
        """
        # Extract parameters
        center_x = parameters.get('center_x', self.center[0])
        center_y = parameters.get('center_y', self.center[1])
        center_z = parameters.get('center_z', self.center[2])
        size_x = parameters.get('size_x', self.size[0])
        size_y = parameters.get('size_y', self.size[1])
        size_z = parameters.get('size_z', self.size[2])
        exhaustiveness = parameters.get('exhaustiveness', 8)
        
        # Simulate scoring based on realistic factors
        
        # 1. Distance from original center (penalty for moving too far)
        original_center = np.array(self.center)
        current_center = np.array([center_x, center_y, center_z])
        distance_penalty = 0.1 * np.linalg.norm(current_center - original_center)
        
        # 2. Box volume (penalty for extremely large or small boxes)
        volume = size_x * size_y * size_z
        if volume < 1000:
            volume_penalty = 2.0 * (1.0 - volume / 1000.0)
        elif volume > 27000:
            volume_penalty = 0.5 * (volume - 27000) / 10000.0
        else:
            volume_penalty = 0.0
        
        # 3. Exhaustiveness bonus (diminishing returns)
        exhaustiveness_bonus = -0.2 * np.log(exhaustiveness / 8.0)
        
        # 4. Random noise to simulate experimental variation
        noise = np.random.normal(0, 0.3)
        
        # Base score (typical good binding energy)
        base_score = -8.5
        
        # Calculate final mock score
        mock_score = (base_score + 
                     distance_penalty + 
                     volume_penalty + 
                     exhaustiveness_bonus + 
                     noise)
        
        # Store in optimization history
        self.optimization_history.append({
            'parameters': parameters.copy(),
            'score': mock_score
        })
        
        return mock_score
    
    def latin_hypercube_sampling(self, n_samples: int) -> List[Dict[str, float]]:
        """Generate initial parameter samples using Latin Hypercube Sampling"""
        param_names = list(self.search_space.keys())
        n_params = len(param_names)
        
        # Generate LHS samples in [0,1]^n
        samples = np.zeros((n_samples, n_params))
        for i in range(n_params):
            samples[:, i] = (np.random.permutation(n_samples) + np.random.random(n_samples)) / n_samples
        
        # Scale to actual parameter ranges
        parameter_sets = []
        for sample in samples:
            params = {}
            for i, param_name in enumerate(param_names):
                lower, upper = self.search_space[param_name]
                params[param_name] = lower + (upper - lower) * sample[i]
            parameter_sets.append(params)
        
        return parameter_sets
    
    def bayesian_optimization(self, n_iterations: int = 20, 
                            n_initial_points: int = 5) -> Tuple[Dict[str, float], float]:
        """
        Perform Bayesian optimization to find optimal parameters
        
        Args:
            n_iterations: Number of optimization iterations
            n_initial_points: Number of initial sampling points
            
        Returns:
            Tuple of (best_parameters, best_score)
        """
        logger.info(f"🔍 Starting Bayesian optimization...")
        logger.info(f"   Iterations: {n_iterations}")
        logger.info(f"   Initial points: {n_initial_points}")
        
        # Parameter setup
        param_names = list(self.search_space.keys())
        bounds = [self.search_space[param] for param in param_names]
        
        # Generate initial samples
        initial_params = self.latin_hypercube_sampling(n_initial_points)
        
        # Evaluate initial points
        logger.info("📊 Evaluating initial points...")
        X_observed = []
        Y_observed = []
        
        for i, params in enumerate(initial_params):
            score = self.mock_docking_score(params)
            
            # Convert parameters to array
            x = [params[param] for param in param_names]
            X_observed.append(x)
            Y_observed.append(score)
            
            logger.info(f"   Point {i+1}/{n_initial_points}: Score = {score:.3f}")
        
        X_observed = np.array(X_observed)
        Y_observed = np.array(Y_observed)
        
        # Track best solution
        best_idx = np.argmin(Y_observed)
        best_params = initial_params[best_idx]
        best_score = Y_observed[best_idx]
        
        logger.info(f"   Initial best score: {best_score:.3f}")
        
        # Gaussian Process setup
        kernel = Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=3
        )
        
        # Bayesian optimization loop
        for iteration in range(n_iterations):
            logger.info(f"🔄 Iteration {iteration + 1}/{n_iterations}")
            
            # Fit Gaussian Process
            gp.fit(X_observed, Y_observed)
            
            # Acquisition function (Expected Improvement)
            def expected_improvement(x):
                x = x.reshape(1, -1)
                mu, sigma = gp.predict(x, return_std=True)
                mu = mu[0]
                sigma = sigma[0]
                
                if sigma < 1e-6:
                    return 0.0
                
                # Calculate improvement
                improvement = best_score - mu
                z = improvement / sigma
                
                from scipy.stats import norm
                ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei  # Negative because we minimize
            
            # Optimize acquisition function
            best_x = None
            best_ei = float('inf')
            
            # Try multiple random starting points
            for _ in range(10):
                x0 = np.random.uniform(
                    low=[b[0] for b in bounds],
                    high=[b[1] for b in bounds]
                )
                
                try:
                    result = minimize(
                        expected_improvement,
                        x0,
                        bounds=bounds,
                        method='L-BFGS-B'
                    )
                    
                    if result.success and result.fun < best_ei:
                        best_ei = result.fun
                        best_x = result.x
                except:
                    continue
            
            if best_x is None:
                logger.warning("   Acquisition optimization failed, using random point")
                best_x = np.random.uniform(
                    low=[b[0] for b in bounds],
                    high=[b[1] for b in bounds]
                )
            
            # Evaluate new point
            next_params = {param: best_x[i] for i, param in enumerate(param_names)}
            next_score = self.mock_docking_score(next_params)
            
            # Update observations
            X_observed = np.vstack([X_observed, best_x])
            Y_observed = np.append(Y_observed, next_score)
            
            # Update best solution
            if next_score < best_score:
                best_score = next_score
                best_params = next_params
                logger.info(f"   🎉 New best score: {best_score:.3f}")
            else:
                logger.info(f"   Score: {next_score:.3f} (best: {best_score:.3f})")
        
        # Store results
        self.best_parameters = best_params
        self.best_score = best_score
        
        logger.info(f"✅ Optimization complete!")
        logger.info(f"   Best score: {best_score:.3f}")
        logger.info(f"   Total evaluations: {len(self.optimization_history)}")
        
        return best_params, best_score
    
    def grid_search(self, n_points_per_dim: int = 3) -> Tuple[Dict[str, float], float]:
        """
        Perform grid search optimization
        
        Args:
            n_points_per_dim: Number of grid points per dimension
            
        Returns:
            Tuple of (best_parameters, best_score)
        """
        logger.info(f"🔍 Starting grid search optimization...")
        logger.info(f"   Grid points per dimension: {n_points_per_dim}")
        
        # Focus on most important parameters
        important_params = ['center_x', 'center_y', 'center_z', 'size_x', 'size_y', 'size_z']
        
        # Create grid
        param_grids = {}
        for param in important_params:
            lower, upper = self.search_space[param]
            param_grids[param] = np.linspace(lower, upper, n_points_per_dim)
        
        # Generate all combinations
        import itertools
        grid_combinations = list(itertools.product(*param_grids.values()))
        total_combinations = len(grid_combinations)
        
        logger.info(f"   Total combinations: {total_combinations}")
        
        best_params = None
        best_score = float('inf')
        
        # Evaluate all combinations
        for i, combination in enumerate(grid_combinations):
            # Create parameter dictionary
            params = dict(zip(important_params, combination))
            
            # Add default values for other parameters
            for param, (lower, upper) in self.search_space.items():
                if param not in params:
                    params[param] = (lower + upper) / 2
            
            # Evaluate
            score = self.mock_docking_score(params)
            
            if score < best_score:
                best_score = score
                best_params = params
            
            if (i + 1) % max(1, total_combinations // 10) == 0:
                progress = (i + 1) / total_combinations * 100
                logger.info(f"   Progress: {progress:.1f}% (best: {best_score:.3f})")
        
        # Store results
        self.best_parameters = best_params
        self.best_score = best_score
        
        logger.info(f"✅ Grid search complete!")
        logger.info(f"   Best score: {best_score:.3f}")
        
        return best_params, best_score
    
    def visualize_optimization(self):
        """Create optimization progress visualization"""
        if not self.optimization_history:
            logger.warning("No optimization history to visualize")
            return
        
        # Extract data
        scores = [entry['score'] for entry in self.optimization_history]
        iterations = list(range(len(scores)))
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Optimization progress
        plt.subplot(2, 2, 1)
        plt.plot(iterations, scores, 'b-', alpha=0.7, label='All scores')
        
        # Running best
        running_best = []
        current_best = float('inf')
        for score in scores:
            if score < current_best:
                current_best = score
            running_best.append(current_best)
        
        plt.plot(iterations, running_best, 'r-', linewidth=2, label='Best score')
        plt.xlabel('Iteration')
        plt.ylabel('Docking Score (kcal/mol)')
        plt.title('Optimization Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Score distribution
        plt.subplot(2, 2, 2)
        plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(self.best_score, color='red', linestyle='--', 
                   label=f'Best: {self.best_score:.3f}')
        plt.xlabel('Docking Score (kcal/mol)')
        plt.ylabel('Frequency')
        plt.title('Score Distribution')
        plt.legend()
        
        # Subplot 3: Parameter correlation (center_x vs score)
        plt.subplot(2, 2, 3)
        center_x_values = [entry['parameters'].get('center_x', 0) for entry in self.optimization_history]
        plt.scatter(center_x_values, scores, alpha=0.6)
        plt.xlabel('Center X')
        plt.ylabel('Docking Score (kcal/mol)')
        plt.title('Center X vs Score')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Box volume vs score
        plt.subplot(2, 2, 4)
        volumes = []
        for entry in self.optimization_history:
            params = entry['parameters']
            volume = (params.get('size_x', 20) * 
                     params.get('size_y', 20) * 
                     params.get('size_z', 20))
            volumes.append(volume)
        
        plt.scatter(volumes, scores, alpha=0.6)
        plt.xlabel('Box Volume (Ų)')
        plt.ylabel('Docking Score (kcal/mol)')
        plt.title('Box Volume vs Score')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'optimization_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Optimization visualization saved to {plot_path}")
    
    def generate_optimized_config(self) -> Optional[str]:
        """Generate optimized configuration file"""
        if self.best_parameters is None:
            logger.error("No optimized parameters available")
            return None
        
        config_path = self.output_dir / 'optimized_config.conf'
        
        try:
            with open(config_path, 'w') as f:
                f.write(f"# Optimized Docking Configuration\n")
                f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Best Score: {self.best_score:.4f} kcal/mol\n")
                f.write(f"# Receptor: {self.receptor_file}\n")
                if self.ligand_file:
                    f.write(f"# Ligand: {self.ligand_file}\n")
                f.write(f"\n")
                
                # Center coordinates
                f.write(f"center_x = {self.best_parameters['center_x']:.4f}\n")
                f.write(f"center_y = {self.best_parameters['center_y']:.4f}\n")
                f.write(f"center_z = {self.best_parameters['center_z']:.4f}\n")
                f.write(f"\n")
                
                # Box size
                f.write(f"size_x = {self.best_parameters['size_x']:.4f}\n")
                f.write(f"size_y = {self.best_parameters['size_y']:.4f}\n")
                f.write(f"size_z = {self.best_parameters['size_z']:.4f}\n")
                f.write(f"\n")
                
                # Algorithm parameters
                f.write(f"exhaustiveness = {int(self.best_parameters['exhaustiveness'])}\n")
                if 'num_modes' in self.best_parameters:
                    f.write(f"num_modes = {int(self.best_parameters['num_modes'])}\n")
                if 'energy_range' in self.best_parameters:
                    f.write(f"energy_range = {self.best_parameters['energy_range']:.2f}\n")
            
            logger.info(f"✓ Optimized configuration saved to {config_path}")
            return str(config_path)
            
        except Exception as e:
            logger.error(f"Error generating config file: {str(e)}")
            return None
    
    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report"""
        report_path = self.output_dir / 'optimization_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("🎯 DOCKING PARAMETER OPTIMIZATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Input information
            f.write("📋 INPUT INFORMATION\n")
            f.write("-" * 30 + "\n")
            f.write(f"Receptor file: {self.receptor_file}\n")
            f.write(f"Receptor atoms: {len(self.receptor_atoms)}\n")
            if self.ligand_file:
                f.write(f"Ligand file: {self.ligand_file}\n")
                f.write(f"Ligand atoms: {len(self.ligand_atoms)}\n")
            f.write(f"Initial center: {self.center}\n")
            f.write(f"Initial size: {self.size}\n\n")
            
            # Optimization results
            if self.best_parameters:
                f.write("🏆 OPTIMIZATION RESULTS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Best score: {self.best_score:.4f} kcal/mol\n")
                f.write(f"Total evaluations: {len(self.optimization_history)}\n\n")
                
                f.write("Optimized parameters:\n")
                for param, value in self.best_parameters.items():
                    if isinstance(value, float):
                        f.write(f"  {param}: {value:.4f}\n")
                    else:
                        f.write(f"  {param}: {value}\n")
                f.write("\n")
                
                # Parameter changes
                f.write("Parameter changes from initial:\n")
                initial_params = {
                    'center_x': self.center[0], 'center_y': self.center[1], 'center_z': self.center[2],
                    'size_x': self.size[0], 'size_y': self.size[1], 'size_z': self.size[2]
                }
                
                for param in initial_params:
                    if param in self.best_parameters:
                        initial = initial_params[param]
                        optimized = self.best_parameters[param]
                        change = optimized - initial
                        f.write(f"  {param}: {initial:.3f} → {optimized:.3f} (Δ{change:+.3f})\n")
                f.write("\n")
            
            # Statistics
            if self.optimization_history:
                scores = [entry['score'] for entry in self.optimization_history]
                f.write("📊 OPTIMIZATION STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Best score: {min(scores):.4f} kcal/mol\n")
                f.write(f"Worst score: {max(scores):.4f} kcal/mol\n")
                f.write(f"Mean score: {np.mean(scores):.4f} kcal/mol\n")
                f.write(f"Standard deviation: {np.std(scores):.4f} kcal/mol\n")
                f.write(f"Improvement: {max(scores) - min(scores):.4f} kcal/mol\n\n")
            
            # Recommendations
            f.write("💡 RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            f.write("1. Use the optimized configuration for production docking\n")
            f.write("2. Validate results with experimental binding data\n")
            f.write("3. Consider multiple conformations for flexible ligands\n")
            f.write("4. Monitor for local minima in optimization landscape\n")
            f.write("5. Perform multiple independent optimization runs\n\n")
            
            f.write("📁 OUTPUT FILES\n")
            f.write("-" * 30 + "\n")
            f.write("• optimized_config.conf - Optimized Vina configuration\n")
            f.write("• optimization_analysis.png - Optimization progress plots\n")
            f.write("• optimization_report.txt - This detailed report\n")
        
        logger.info(f"✓ Optimization report saved to {report_path}")
        return str(report_path)


def optimize_docking_parameters(receptor_file: str, ligand_file: str = None,
                               method: str = 'bayesian', n_iterations: int = 20,
                               output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for docking parameter optimization
    
    Args:
        receptor_file: Path to receptor PDBQT file
        ligand_file: Path to ligand PDBQT file
        method: Optimization method ('bayesian' or 'grid')
        n_iterations: Number of optimization iterations
        output_dir: Output directory for results
        
    Returns:
        Dictionary with optimization results
    """
    optimizer = DockingParameterOptimizer(receptor_file, ligand_file, output_dir=output_dir)
    
    try:
        # Run optimization
        if method == 'bayesian':
            best_params, best_score = optimizer.bayesian_optimization(n_iterations)
        elif method == 'grid':
            best_params, best_score = optimizer.grid_search()
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Generate outputs
        optimizer.visualize_optimization()
        config_path = optimizer.generate_optimized_config()
        report_path = optimizer.generate_optimization_report()
        
        return {
            'success': True,
            'best_parameters': best_params,
            'best_score': best_score,
            'config_path': config_path,
            'report_path': report_path,
            'optimizer': optimizer
        }
        
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        return {'success': False, 'error': str(e)}
