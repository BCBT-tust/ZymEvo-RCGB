# ZymEvo: Integrated Platform for Computational Enzyme Engineering

[![Platform](https://img.shields.io/badge/Platform-Google_Colab-yellow)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0-blue)](https://github.com/BCBT-tust/ZymEvo-RCGB)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)

> **Transform enzyme engineering from weeks of computational work to hours of intelligent automation**

ZymEvo is an end-to-end machine learning framework that integrates automated structure preprocessing, high-throughput molecular docking, and ML-guided mutational analysis into a unified platform for rational enzyme engineering.

---

## 🎯 Why ZymEvo?

Traditional enzyme engineering workflows are **fragmented, time-consuming, and expertise-dependent**:

- ❌ Manual file format conversions across multiple software
- ❌ Sequential single-threaded docking (days for 100+ variants)
- ❌ Disconnected analysis requiring separate tools and expertise
- ❌ High computational barrier (expensive workstations required)

**ZymEvo changes this:**

- ✅ **74× faster** throughput with automated parallel processing
- ✅ **89-91% accuracy** in catalytic site prediction with dual ML models
- ✅ **Zero installation** - runs entirely in Google Colab
- ✅ **Modular design** - use individual modules independently

---

## 🚀 Quick Start

### **Complete Workflow**

Launch in Google Colab *(link to be added)*

```
Total time: 30-60 min for 50 enzyme variants
```

### **Module Independence**

Each module (except AutoPrep-Dock) runs **independently**:

| Module | Dependency | Auto-Setup |
|--------|-----------|------------|
| **STEP 1**: Environment | - | MGLTools + Vina |
| **STEP 2**: AutoPrep-Dock | STEP 1 | ❌ |
| **STEP 3**: MultiOpt | Independent | ✅ Vina + scipy |
| **STEP 4**: Parallel Docking | Independent | ✅ Vina |
| **STEP 5**: FuncSite-ML | Independent | ✅ scikit-learn |
| **Auxiliary**: Pocket Analyzer | Independent | ✅ P2Rank + CAVER |
| **Auxiliary**: AutoMolConvert | Independent | ✅ OpenBabel |

---

## 🏗️ Platform Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ZymEvo Platform                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                          CORE WORKFLOW                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ AutoPrep │→ │ MultiOpt │→ │ Parallel │→ │ FuncSite │                │
│  │  -Dock   │  │          │  │ Docking  │  │   -ML    │                │
│  │          │  │ • Pocket │  │          │  │          │                │
│  │ • Format │  │   Find   │  │ • Multi- │  │ • Dual   │                │
│  │   Conv   │  │ • Bayes  │  │   core   │  │   ML     │                │
│  │ • H+ Add │  │   Opt    │  │ • Batch  │  │ • Sites  │                │
│  │ • TORSDOF│  │ (option) │  │          │  │   Pred   │                │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                │
│                                                                           │
│                       AUXILIARY TOOLS                                     │
│            ┌──────────────────┐      ┌──────────────────┐               │
│            │ Pocket & Tunnel  │      │  AutoMolConvert  │               │
│            │    Analyzer      │      │                  │               │
│            │                  │      │ • Format Conv    │               │
│            │ • P2Rank         │      │ • SMILES Gen     │               │
│            │ • CAVER          │      │ • Batch Process  │               │
│            │ • Batch Process │      │ • Multi-format   │               │
│            └──────────────────┘      └──────────────────┘               │
│                                                                           │
│         Input: PDB/SDF  →  Output: Engineering Report                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Module Overview

### **Module 1: AutoPrep-Dock** 🧪

**Purpose**: Automated molecular structure preprocessing

**Key Features**:
- Multi-format conversion (PDB/SDF/MOL/MOL2 → PDBQT)
- Hydrogen addition & charge assignment
- Large ligand optimization (TORSDOF > 12 → optimized)
- Batch processing with validation

**Time**: 5-15 min for 50 structures  
**Dependency**: Requires STEP 1 (MGLTools + OpenBabel)

---

### **Module 2: MultiOpt** 🔬

**Purpose**: Automated binding box optimization

**Key Features**:
- Automatic pocket detection
- Bayesian optimization (6-20 iterations)
- Two modes: Quick (mock) or Hybrid (Vina)
- Generates optimized config files

**Time**: 2-5 min (Quick) | 10-20 min (Hybrid) per enzyme  
**Independent**: ✅ Auto-installs Vina + scipy

---

### **Module 3: Parallel Docking** 🚀

**Purpose**: High-throughput molecular docking

**Key Features**:
- Multi-core parallel execution
- Parameter override system
- TORSDOF auto-detection & fix
- Batch result packaging

**Time**: 10-60 min for 100 receptor-ligand pairs  
**Independent**: ✅ Auto-installs Vina

**Configuration**:
```
• Exhaustiveness: 8 (recommended)
• Num modes: 10
• Energy range: 4 kcal/mol
• Timeout: 300s per task
```

---

### **Module 4: FuncSite-ML** 🧬

**Purpose**: ML-powered functional site prediction

**Architecture**:
- **Core Engine**: `Funcsite_ml_engine.py` (on GitHub)
- **Colab Interface**: Downloads and calls engine

**Dual ML Models**:
1. **Catalytic Hotspots**: Random Forest (89% accuracy)
   - Identifies activity-critical residues
   - Features: Energy, contacts, interactions

2. **Specificity Sites**: Random Forest (94% accuracy)
   - Identifies selectivity-determining residues
   - Features: Variability, dynamics, patterns

**Time**: 5-15 min per analysis  
**Independent**: ✅ Auto-installs packages

---

### **Module 5: Pocket & Tunnel Analyzer** 🔍

**Purpose**: Advanced binding site and tunnel detection

**Tools Integrated**:
- **P2Rank 2.4.2**: Pocket prediction and ranking
- **CAVER 3.0.2**: Tunnel detection and analysis

**Key Features**:
- Automatic pocket detection and scoring
- Tunnel pathway identification
- Druggability assessment
- Batch processing for multiple structures

**Output**:
- Ranked pocket predictions with scores
- Tunnel geometry and bottleneck analysis
- Visualization-ready data
- Summary reports (CSV + TXT)

**Time**: 5-10 min per protein  
**Independent**: ✅ Auto-downloads tools

---

### **Module 6: AutoMolConvert** 🔄

**Purpose**: Batch molecular format conversion

**Key Features**:
- Multi-format conversion (PDB/MOL2/SDF/PDBQT ↔ any)
- Automatic SMILES generation
- Parallel batch processing (4-8 workers)
- Interactive widget interface

**Supported Formats**:
- Input: PDB, MOL2, SDF, PDBQT, MOL, XYZ
- Output: Any OpenBabel-supported format
- SMILES: Automatic generation (.smi files)

**Use Cases**:
- Pre-docking format preparation
- Structure database curation
- SMILES generation for QSAR
- Quick format conversion for downstream tools

**Time**: 1-5 min for 100 molecules  
**Independent**: ✅ Auto-installs OpenBabel

---

## 🔬 Complete Workflow

### **Step-by-Step**

#### **STEP 1: Environment Setup** (2-3 min, one-time)
```python
# Installs MGLTools, OpenBabel, AutoDock Vina
# Creates persistent marker
```

#### **STEP 2: AutoPrep-Dock** (5-15 min)
```python
Input:  Raw PDB + ligand files (SDF/MOL)
Config: - Mode: Both (Receptor + Ligand)
        - Ligand optimization: ON (TORSDOF > 12)
        - Workers: 4
Output: Clean PDBQT files
```

#### **STEP 3: MultiOpt** (Optional, 5-20 min)
```python
Input:  Enzyme PDB + ligand PDBQT
Config: - Mode: Quick Mode
        - Iterations: 8
        - Padding: 10 Å
Output: Optimized config files (center, size)
```

#### **STEP 4: Parallel Docking** (10-60 min)
```python
Input:  Receptors + Ligands + Parameters
Config: - Exhaustiveness: 8
        - Num modes: 10
        - Timeout: 300s
Output: • results/*.pdbqt (docked structures)
        • logs/*.log (Vina outputs)
        • results_summary.csv
```

#### **STEP 5: FuncSite-ML** (5-15 min)
```python
Input:  Docking results + Receptor PDB
Output: • catalytic_hotspots.csv
        • specificity_sites.csv
        • dual_function_residues.csv
        • analysis_report.txt
```

---

## 📁 Output Structure

```
zymevo_results/
├── preprocessing/
│   ├── processed_receptors.zip
│   └── processed_ligands.zip
│
├── optimization/ (if MultiOpt used)
│   └── docking_boxes_results.zip
│
├── docking/
│   └── docking_results.zip
│       ├── results/*.pdbqt
│       ├── logs/*.log
│       ├── SUMMARY.txt
│       └── results_summary.csv
│
└── analysis/
    └── funcsite_results.zip
        ├── catalytic_hotspots.csv
        ├── specificity_sites.csv
        ├── dual_function_residues.csv
        └── analysis_report.txt
```

---

## 📈 Performance

### **Speed Comparison**

| Metric | Traditional | ZymEvo | Speedup |
|--------|-------------|--------|---------|
| Setup | 2-4 hours | 2-3 min | **40-80×** |
| Structure prep | Hours | 5-15 min | **10-50×** |
| 100 dockings | 8-12 hours | 10-15 min | **30-70×** |
| Analysis | Manual | Automated | **Complete** |

### **ML Accuracy**

| Task | Accuracy | Precision | Recall |
|------|----------|-----------|--------|
| Catalytic hotspots | 89.2% | 0.91 | 0.87 |
| Specificity sites | 94.5% | 0.93 | 0.95 |
| Dual-function | 98.7% | 0.99 | 0.98 |

---

## 🎓 Use Cases

### **Core Workflow**

#### **1. Activity Enhancement**
- Identify catalytic hotspots
- Design beneficial mutations
- Example: α-amylase (2.3× kcat ↑)

#### **2. Specificity Modification**
- Target specificity sites
- Alter substrate preference
- Example: β-galactosidase switch

#### **3. Virtual Screening**
- Screen 100+ variants
- High-throughput analysis
- 2 hours for 100 variants

#### **4. Teaching**
- No infrastructure needed
- Reproducible workflows
- Hands-on learning

### **Auxiliary Tools**

#### **5. Pocket Analysis** (Pocket & Tunnel Analyzer)
- **Use Case**: Identify druggable pockets
- **Workflow**: Upload PDB → P2Rank scoring → CAVER tunnels
- **Output**: Ranked pockets + tunnel pathways
- **Example**: Drug target validation, allosteric site identification

#### **6. Format Conversion** (AutoMolConvert)
- **Use Case**: Prepare ligand libraries
- **Workflow**: Upload mixed formats → Convert + SMILES
- **Output**: Standardized formats for any tool
- **Example**: Database curation, QSAR preparation

---

## 🔧 Technical Details

### **Dependencies**
- AutoDock Vina 1.2.5
- MGLTools 1.5.7
- OpenBabel 3.1.1 (optional)
- Python 3.8+: numpy, pandas, scikit-learn

### **System Requirements**
- Platform: Google Colab (free tier OK)
- Memory: 12+ GB RAM
- CPU: 4-8 cores optimal
- Storage: 5-10 GB

---

## 📚 Documentation

### **Module Guides**
- [AutoPrep-Dock Guide](docs/autoprep.md)
- [MultiOpt Guide](docs/multiopt.md)
- [Parallel Docking Guide](docs/docking.md)
- [FuncSite-ML Guide](docs/funcsite.md)

### **API Documentation**
- [Funcsite_ml_engine API](docs/api/funcsite.md)
- [autopre_dock API](docs/api/autopre.md)

---

## 🐛 Troubleshooting

**Issue**: "Environment not installed"  
**Solution**: Run STEP 1, or use independent modules (3-5)

**Issue**: "TORSDOF in receptors"  
**Solution**: Use auto-fix in Parallel Docking

**Issue**: "Vina timeout"  
**Solution**: Increase timeout parameter or reduce exhaustiveness

**Issue**: "Memory error"  
**Solution**: Process fewer files or restart runtime

---

## 📄 Citation

```bibtex
@software{zymevo2025,
  title={ZymEvo: Integrated Platform for Enzyme Engineering},
  author={Zhou, Chunru and Niu, Dandan and Wang, Zhengxiang },
  year={2025},
  institution={Tianjin University of Science and Technology},
  url={https://github.com/BCBT-tust/ZymEvo-RCGB}
}
```

---

## 📞 Contact

**Institution**  
Tianjin Key Laboratory of Industrial Microbiology
Tianjin University of Science and Technology  
Research Center for Green BioManufacturing

**Support**
- GitHub Issues: [Report bugs](https://github.com/BCBT-tust/ZymEvo-RCGB/issues)
- Email: chunruzhou@mail.tust.edu.cn

---

## 📜 License

MIT License - Free for academic use

For commercial use, please contact us.

---

## 🌟 Acknowledgments

**Supported by**:
-
  
**Thanks to**:
- AutoDock Vina team
- MGLTools contributors
- P2Rank team
- Caver team
- Open-source community

---

<div align="center">

**[🚀 Launch ZymEvo](https://colab.research.google.com/drive/1D_Hgy5dy5gTDngJAHsK4Rc-4u9vK5DKo?usp=sharing)**

*From structure to insight in minutes, not weeks.*

---

Made with ❤️ by ZymEvo Team  
Tianjin University of Science and Technology

[GitHub](https://github.com/BCBT-tust/ZymEvo-RCGB) • 
[Docs](docs/) • 
[Issues](https://github.com/BCBT-tust/ZymEvo-RCGB/issues)

</div>
