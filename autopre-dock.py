# =============================================================================
# ZymEvo Batch Preprocessing ———— AutoPre—Dock
# =============================================================================

# @title 🧪 ZymEvo Batch Preprocessing ———— AutoPre—Dock { display-mode: "form" }
# @markdown ### Batch Processing Options
# @markdown Configure your preprocessing parameters below

processing_mode = "Prepare Receptors (PDB → PDBQT)"  # @param ["Prepare Receptors (PDB → PDBQT)", "Prepare Ligands (→ PDBQT)", "Convert Format Only", "Interactive Mode"]
remove_water = True  # @param {type:"boolean"}
remove_nonstd = False  # @param {type:"boolean"}
auto_convert = True  # @param {type:"boolean"}
output_format = "pdb"  # @param ["pdb", "mol2", "sdf", "pdbqt"]

import subprocess
import sys

print("📥 Downloading ZymEvo preprocessing script...")
result = subprocess.run([
    "wget", "-q",
    "https://raw.githubusercontent.com/BCBT-tust/ZymEvo-RCGB/main/autopre-dock.py",
    "-O", "/tmp/autopre-dock.py""
], capture_output=True)

if result.returncode == 0:
    print("✓ Download complete\n")
    
    sys.path.insert(0, '/tmp')
    from autopre-dock import MolecularPreprocessor, print_status
    
    preprocessor = MolecularPreprocessor()
    
    if not preprocessor.upload_files():
        sys.exit(1)
    
    input_files = preprocessor.list_input_files()
    
    if not input_files:
        print_status("No valid input files found", "error")
        sys.exit(1)
    
    if processing_mode == "Prepare Receptors (PDB → PDBQT)":
        pdb_files = [f for f in input_files if f.endswith('.pdb')]
        if pdb_files:
            results = preprocessor.prepare_receptor_batch(pdb_files, remove_water, remove_nonstd)
            preprocessor.generate_summary(results)
            preprocessor.download_results("*.pdbqt")
        else:
            print_status("No PDB files found for receptor preparation", "warning")
    
    elif processing_mode == "Prepare Ligands (→ PDBQT)":
        ligand_files = [f for f in input_files if not f.endswith('.pdbqt')]
        if ligand_files:
            results = preprocessor.prepare_ligand_batch(ligand_files, auto_convert)
            preprocessor.generate_summary(results)
            preprocessor.download_results("*.pdbqt")
        else:
            print_status("No ligand files found", "warning")
    
    elif processing_mode == "Convert Format Only":
        results = preprocessor.batch_convert(input_files, output_format)
        preprocessor.generate_summary(results)
        preprocessor.download_results(f"*.{output_format}")
    
    elif processing_mode == "Interactive Mode":
        print_status("Interactive mode activated. Use preprocessor object for custom operations.", "info")
        print("\nAvailable methods:")
        print("  - preprocessor.prepare_receptor_batch(pdb_files)")
        print("  - preprocessor.prepare_ligand_batch(ligand_files)")
        print("  - preprocessor.batch_convert(files, format)")
        print("  - preprocessor.download_results(pattern)")
        
else:
    print("❌ Failed to download preprocessing script")
    print("Manual download: https://github.com/BCBT-tust/ZymEvo-RCGB/blob/main/autopre-dock.py"")
