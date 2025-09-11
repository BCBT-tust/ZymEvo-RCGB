# ZymEvo: An end-to-end machine learning framework for the evolution of automated enzyme molecules

[![Platform](https://img.shields.io/badge/Platform-Google_Colab-yellow)](https://colab.research.google.com/drive/1yr9DkpHFfzAHju6eyEMkuKQHHxfmNqYr?usp=sharing)
[![License](https://img.shields.io/badge/License-Research_Free-orange)](LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0-blue)](https://github.com/your-repo/ezdock)

**ZymEvo** is a comprehensive, automated molecular docking platform designed for enzyme engineering research. It transforms complex multi-software workflows into simple, automated processes that run entirely in Google Colab.

## 🚀 Quick Start

**Try ZymEvoDock now:** [Google Colab Link](https://colab.research.google.com/drive/1yr9DkpHFfzAHju6eyEMkuKQHHxfmNqYr?usp=sharing)

No installation required - runs entirely in your browser.

## 📊 Key Features

### Complete Workflow Coverage
- **Automated preprocessing** of receptors and ligands (multiple formats)
- **Parallel molecular docking** with AutoDock Vina
- **Dual ML models** for catalytic hotspot and specificity prediction
- **Professional visualizations** and engineering reports

### Format Support
- **Receptors**: PDB → PDBQT
- **Ligands**: SDF, MOL, MOL2, XML, XYZ, PDB → PDBQT
- **Automatic conversion** with built-in and OpenBabel support

### Machine Learning Analysis
- **Catalytic hotspot identification** using Random Forest
- **Specificity site prediction** with interaction variability analysis
- **Dual-function residue detection** for comprehensive engineering guidance

## 🔬 Scientific Workflow

### 1. Environment Setup (2-3 min)
```python
# Automated installation of AutoDock Vina and MGLTools
# No manual configuration required
```

### 2. File Preprocessing (1-10 min)
```python
# Batch receptor processing: PDB → PDBQT with hydrogens
# Multi-format ligand conversion: SDF/MOL/MOL2/XML/XYZ → PDBQT
# Parameter extraction: Automated binding box calculation
```

### 3. Parallel Docking (5-60 min)
```python
# Multi-core parallel execution
# Progress tracking with ETA
# Automatic result validation
```

### 4. ML-Powered Analysis (2-10 min)
```python
# Dual ML models for enzyme engineering
# Feature engineering with 18+ molecular descriptors
# Publication-ready visualizations
```

## 📈 Performance Metrics

| Metric | Traditional Approach | ZymEvoDock |
|--------|---------------------|--------|
| Setup Time | Hours | **2-3 minutes** |
| File Processing | Manual, sequential | **Automated batch** |
| Docking Execution | Single-threaded | **Multi-core parallel** |
| Result Analysis | Separate software | **Integrated ML** |
| Hardware Requirements | High-end workstation | **Any browser** |

## 🧬 Enzyme Engineering Analysis

### Catalytic Hotspots
- **Definition**: Residues critical for catalytic activity
- **Identification**: Energy contribution + contact frequency
- **ML Features**: 18+ descriptors including interaction types, physicochemical properties

### Specificity Sites
- **Definition**: Residues determining substrate selectivity
- **Identification**: Interaction variability across binding poses
- **Analysis**: t-SNE clustering, probability distributions

### Dual-Function Residues
- **Definition**: Important for both activity AND specificity
- **Priority**: High-confidence predictions (>0.9 probability)
- **Impact**: Complex mutation effects on multiple properties

## 📊 Output Files

### Analysis Reports
- `enzyme_engineering_report.txt` - Comprehensive analysis
- `engineering_summary.txt` - Quick overview
- `catalytic_hotspots_complete.csv` - Ranked hotspots
- `specificity_sites_complete.csv` - Ranked specificity sites

### Visualizations
- ROC curves and confusion matrices
- Feature importance analysis
- t-SNE clustering plots
- Interaction heatmaps
- Probability distributions

## 🔧 Technical Details

### Dependencies
- **AutoDock Vina 1.2.5** - Molecular docking engine
- **MGLTools** - Molecular preparation
- **OpenBabel** - Format conversion (optional)
- **scikit-learn** - Machine learning models
- **Standard Python** - pandas, numpy, matplotlib, seaborn

### System Requirements
- **Platform**: Google Colab (free tier sufficient)
- **Memory**: 12+ GB RAM (automatically managed)
- **CPU**: Multi-core parallel processing
- **Storage**: Results automatically packaged for download

### ML Model Performance
- **Catalytic Hotspot Model**: Random Forest (n_estimators=80, max_depth=4)
- **Specificity Model**: Random Forest with class balancing
- **Cross-validation**: 10-fold CV with balanced accuracy
- **Feature Selection**: 18+ molecular descriptors

## 📚 Usage Example

```python
# 1. Run environment setup
# 2. Upload receptor files (PDB format)
# 3. Upload ligand files (SDF/MOL/MOL2/PDB)
# 4. Extract docking parameters
# 5. Run parallel docking
# 6. Analyze with ML models
# 7. Download comprehensive results
```

## 🎯 Target Applications

- **Enzyme Engineering**: Rational design guidance
- **Drug Discovery**: Virtual screening campaigns
- **Academic Research**: Teaching and learning molecular docking
- **Industrial Applications**: High-throughput analysis

## 📄 Citation

If you use ZymEvoDock in your research, please cite:

## 🏛️ Institution

**Tianjin University of Science and Technology**  
Research Center for Green BioManufacturing

## 📞 Contact

For questions and support:
- Open an issue on GitHub
- Email: [zhoucr2023@163.com]

## 🔄 Updates

- **v1.0** (2025.08): Initial release with complete workflow
- Dual ML models for enzyme engineering
- Multi-format support and parallel processing

---

**Ready to start molecular docking analysis?** [Open in Google Colab](https://colab.research.google.com/drive/1yr9DkpHFfzAHju6eyEMkuKQHHxfmNqYr?usp=sharing)
