#!/usr/bin/env python3
"""
Tunnel Characteristics Analysis Tool
Extracts and visualizes tunnel properties from CAVER output
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class TunnelAnalyzer:
    """Analyze and visualize tunnel characteristics from CAVER output"""
    
    def __init__(self, output_dir: str = "/content/protein_analysis/output"):
        self.output_dir = Path(output_dir)
        self.caver_output = self.output_dir / "caver"
        
    def parse_tunnel_characteristics(self, pdb_name: str) -> Optional[pd.DataFrame]:
        """
        Parse tunnel_characteristics.csv for a specific protein
        
        Args:
            pdb_name: Name of the protein (without .pdb extension)
            
        Returns:
            DataFrame with tunnel characteristics, or None if not found
        """
        csv_path = self.caver_output / pdb_name / "analysis" / "tunnel_characteristics.csv"
        
        if not csv_path.exists():
            print(f"⚠️ Tunnel characteristics not found: {csv_path}")
            return None
        
        try:
            # Read with proper space handling
            df = pd.read_csv(csv_path, skipinitialspace=True)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Add protein name column
            df['Protein'] = pdb_name
            
            print(f"✅ Loaded {len(df)} tunnels for {pdb_name}")
            return df
            
        except Exception as e:
            print(f"❌ Error parsing {csv_path}: {e}")
            return None
    
    def parse_all_tunnels(self) -> pd.DataFrame:
        """
        Parse tunnel characteristics for all proteins
        
        Returns:
            Combined DataFrame with all tunnel data
        """
        all_dfs = []
        
        if not self.caver_output.exists():
            print(f"❌ CAVER output directory not found: {self.caver_output}")
            return pd.DataFrame()
        
        # Find all protein subdirectories
        for protein_dir in self.caver_output.iterdir():
            if protein_dir.is_dir():
                pdb_name = protein_dir.name
                df = self.parse_tunnel_characteristics(pdb_name)
                if df is not None:
                    all_dfs.append(df)
        
        if not all_dfs:
            print("⚠️ No tunnel data found")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n✅ Total: {len(combined_df)} tunnels across {len(all_dfs)} proteins")
        
        return combined_df
    
    def get_tunnel_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for tunnel characteristics
        
        Args:
            df: DataFrame with tunnel data
            
        Returns:
            Summary DataFrame with statistics per protein
        """
        if df.empty:
            return pd.DataFrame()
        
        # Numeric columns to summarize
        numeric_cols = []
        for col in ['Throughput', 'Cost', 'Bottleneck radius', 'Length', 'Curvature']:
            if col in df.columns:
                numeric_cols.append(col)
        
        summary_stats = []
        
        for protein in df['Protein'].unique():
            protein_df = df[df['Protein'] == protein]
            
            stats = {
                'Protein': protein,
                'Tunnel_Count': len(protein_df)
            }
            
            for col in numeric_cols:
                stats[f'{col}_Mean'] = protein_df[col].mean()
                stats[f'{col}_Std'] = protein_df[col].std()
                stats[f'{col}_Min'] = protein_df[col].min()
                stats[f'{col}_Max'] = protein_df[col].max()
            
            summary_stats.append(stats)
        
        return pd.DataFrame(summary_stats)
    
    def plot_tunnel_comparison(self, df: pd.DataFrame, save_path: Optional[Path] = None):
        """
        Create comparative plots of tunnel characteristics
        
        Args:
            df: DataFrame with tunnel data
            save_path: Optional path to save the figure
        """
        if df.empty:
            print("⚠️ No data to plot")
            return
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (16, 12)
        
        # Create subplot layout
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Tunnel Characteristics Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Throughput distribution
        if 'Throughput' in df.columns:
            ax = axes[0, 0]
            df.boxplot(column='Throughput', by='Protein', ax=ax)
            ax.set_title('Throughput by Protein')
            ax.set_xlabel('Protein')
            ax.set_ylabel('Throughput')
            plt.sca(ax)
            plt.xticks(rotation=45, ha='right')
        
        # Plot 2: Bottleneck radius distribution
        if 'Bottleneck radius' in df.columns:
            ax = axes[0, 1]
            df.boxplot(column='Bottleneck radius', by='Protein', ax=ax)
            ax.set_title('Bottleneck Radius by Protein')
            ax.set_xlabel('Protein')
            ax.set_ylabel('Bottleneck Radius (Å)')
            plt.sca(ax)
            plt.xticks(rotation=45, ha='right')
        
        # Plot 3: Length distribution
        if 'Length' in df.columns:
            ax = axes[1, 0]
            df.boxplot(column='Length', by='Protein', ax=ax)
            ax.set_title('Tunnel Length by Protein')
            ax.set_xlabel('Protein')
            ax.set_ylabel('Length (Å)')
            plt.sca(ax)
            plt.xticks(rotation=45, ha='right')
        
        # Plot 4: Curvature distribution
        if 'Curvature' in df.columns:
            ax = axes[1, 1]
            df.boxplot(column='Curvature', by='Protein', ax=ax)
            ax.set_title('Tunnel Curvature by Protein')
            ax.set_xlabel('Protein')
            ax.set_ylabel('Curvature')
            plt.sca(ax)
            plt.xticks(rotation=45, ha='right')
        
        # Plot 5: Throughput vs Bottleneck radius scatter
        if 'Throughput' in df.columns and 'Bottleneck radius' in df.columns:
            ax = axes[2, 0]
            for protein in df['Protein'].unique():
                protein_df = df[df['Protein'] == protein]
                ax.scatter(protein_df['Bottleneck radius'], protein_df['Throughput'],
                          label=protein, alpha=0.6, s=100)
            ax.set_xlabel('Bottleneck Radius (Å)')
            ax.set_ylabel('Throughput')
            ax.set_title('Throughput vs Bottleneck Radius')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 6: Length vs Curvature scatter
        if 'Length' in df.columns and 'Curvature' in df.columns:
            ax = axes[2, 1]
            for protein in df['Protein'].unique():
                protein_df = df[df['Protein'] == protein]
                ax.scatter(protein_df['Length'], protein_df['Curvature'],
                          label=protein, alpha=0.6, s=100)
            ax.set_xlabel('Length (Å)')
            ax.set_ylabel('Curvature')
            ax.set_title('Tunnel Length vs Curvature')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_tunnel_profiles(self, df: pd.DataFrame, save_path: Optional[Path] = None):
        """
        Create radar/spider plots for tunnel profiles
        
        Args:
            df: DataFrame with tunnel data
            save_path: Optional path to save the figure
        """
        if df.empty:
            print("⚠️ No data to plot")
            return
        
        # Normalize features for radar plot
        features = []
        for col in ['Throughput', 'Bottleneck radius', 'Length', 'Curvature']:
            if col in df.columns:
                features.append(col)
        
        if len(features) < 3:
            print("⚠️ Not enough features for profile plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(1, len(df['Protein'].unique()), 
                                figsize=(6*len(df['Protein'].unique()), 6),
                                subplot_kw=dict(projection='polar'))
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        fig.suptitle('Tunnel Profiles (Normalized)', fontsize=16, fontweight='bold')
        
        # Normalize data
        df_norm = df.copy()
        for col in features:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0.5
        
        # Plot each protein
        for idx, (ax, protein) in enumerate(zip(axes, df['Protein'].unique())):
            protein_df = df_norm[df_norm['Protein'] == protein]
            
            # Calculate mean values
            mean_values = [protein_df[feat].mean() for feat in features]
            
            # Plot
            angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
            mean_values += mean_values[:1]
            angles += angles[:1]
            
            ax.plot(angles, mean_values, 'o-', linewidth=2, label='Mean')
            ax.fill(angles, mean_values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(features)
            ax.set_ylim(0, 1)
            ax.set_title(f'{protein}\n({len(protein_df)} tunnels)', 
                        fontweight='bold', pad=20)
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Profile plot saved to: {save_path}")
        
        plt.show()
    
    def export_comparison_report(self, df: pd.DataFrame, output_path: Path):
        """
        Export comprehensive comparison report
        
        Args:
            df: DataFrame with tunnel data
            output_path: Path to save the report
        """
        if df.empty:
            print("⚠️ No data to export")
            return
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Raw data
            df.to_excel(writer, sheet_name='Raw_Data', index=False)
            
            # Summary statistics
            summary_df = self.get_tunnel_summary(df)
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            # Per-protein details
            for protein in df['Protein'].unique():
                protein_df = df[df['Protein'] == protein]
                sheet_name = protein[:31]  # Excel sheet name limit
                protein_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✅ Comparison report exported to: {output_path}")


def main():
    """Example usage"""
    analyzer = TunnelAnalyzer()
    
    # Parse all tunnel data
    df = analyzer.parse_all_tunnels()
    
    if not df.empty:
        # Generate summary
        summary = analyzer.get_tunnel_summary(df)
        print("\n📊 Summary Statistics:")
        print(summary.to_string(index=False))
        
        # Create visualizations
        plot_path = Path("/content/protein_analysis/output/tunnel_comparison.png")
        analyzer.plot_tunnel_comparison(df, save_path=plot_path)
        
        profile_path = Path("/content/protein_analysis/output/tunnel_profiles.png")
        analyzer.plot_tunnel_profiles(df, save_path=profile_path)
        
        # Export comprehensive report
        report_path = Path("/content/protein_analysis/output/tunnel_comparison_report.xlsx")
        analyzer.export_comparison_report(df, report_path)


if __name__ == "__main__":
    main()
