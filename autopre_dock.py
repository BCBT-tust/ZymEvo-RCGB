#!/usr/bin/env python3
"""
ZymEvo Unified Preprocessing Script
GitHub: https://github.com/BCBT-tust/ZymEvo-RCGB
Batch processing for receptors and ligands with OpenBabel format conversion
"""

import os
import sys
import subprocess
import shutil
import threading
import concurrent.futures
import time
from pathlib import Path
from IPython.display import HTML, display
from google.colab import files

# Environment variables
PYTHONPATH = "/usr/local/autodocktools/MGLToolsPckgs"
MGLTOOLS_PATH = "/usr/local/autodocktools/bin/pythonsh"
PREPARE_RECEPTOR = "/usr/local/autodocktools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"
PREPARE_LIGAND = "/usr/local/autodocktools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"

os.environ['PYTHONPATH'] = PYTHONPATH


# ==================== Progress & Status ====================

class ProcessingProgress:
    def __init__(self):
        self.completed = 0
        self.total = 0
        self.failed = 0
        self.converted = 0
        self.lock = threading.Lock()
        self.start_time = time.time()

    def update(self, success=True, converted=False):
        with self.lock:
            self.completed += 1
            if not success:
                self.failed += 1
            if converted:
                self.converted += 1

            if self.total > 0:
                progress_pct = (self.completed / self.total) * 100
                elapsed = time.time() - self.start_time
                print(f"📊 Progress: {self.completed}/{self.total} ({progress_pct:.0f}%) | "
                      f"✅ {self.completed - self.failed} | ❌ {self.failed} | ⏱️ {elapsed:.1f}s")

    def reset(self):
        self.completed = 0
        self.total = 0
        self.failed = 0
        self.converted = 0
        self.start_time = time.time()

    def upload_files(self, prompt="📤 Please upload your input files (.pdb, .sdf, .mol, etc.):"):
        try:
            print(prompt)
            uploaded = files.upload()
            if not uploaded:
                print("⚠️ No files uploaded.")
                return None
            print(f"✅ Uploaded {len(uploaded)} file(s): {', '.join(uploaded.keys())}")
            return uploaded
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return None


def print_status(message, status="info"):
    """Print colored status messages"""
    colors = {"success": "#4CAF50", "info": "#2196F3", "warning": "#FF9800", "error": "#F44336"}
    icons = {"success": "✓", "info": "🔄", "warning": "⚠️", "error": "✗"}

    color = colors.get(status, colors["info"])
    icon = icons.get(status, "🔄")

    display(HTML(f"""
    <div style='padding:8px; margin:5px 0; border-radius:4px; 
                background-color:{color}20; border-left:5px solid {color};'>
        <span style='color:{color}; font-weight:bold;'>{icon} </span>{message}
    </div>
    """))


# ==================== Receptor Processing ====================

def process_single_receptor(receptor_file, output_dir):
    """Process a single receptor file"""
    try:
        filename = Path(receptor_file).stem
        output_path = os.path.join(output_dir, f"{filename}.pdbqt")

        cmd = [
            MGLTOOLS_PATH, PREPARE_RECEPTOR,
            "-r", receptor_file,
            "-o", output_path,
            "-A", "hydrogens",
            "-U", "nphs_lps_waters"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True,
                              env=os.environ, timeout=300)

        if result.returncode == 0 and os.path.exists(output_path):
            progress.update(success=True)
            return output_path, None
        else:
            error_msg = result.stderr.strip() or "Unknown error"
            progress.update(success=False)
            return None, error_msg

    except subprocess.TimeoutExpired:
        progress.update(success=False)
        return None, "Timeout (>5 min)"
    except Exception as e:
        progress.update(success=False)
        return None, str(e)


def batch_process_receptors(uploaded_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    progress.total = len(uploaded_files)
    print(f"🚀 Processing {len(uploaded_files)} receptor(s)...")

    successful, failed = [], []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {
            executor.submit(process_single_receptor, filename, output_dir): filename
            for filename in uploaded_files.keys()
        }
        for future in concurrent.futures.as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                output_path, error = future.result()
                if output_path:
                    successful.append(output_path)
                else:
                    failed.append((filename, error))
            except Exception as e:
                failed.append((filename, str(e)))

    return successful, failed


# ==================== Ligand Processing ====================

def setup_openbabel():
    try:
        result = subprocess.run(['obabel', '-V'], capture_output=True)
        if result.returncode == 0:
            print_status("OpenBabel already available", "success")
            return True
    except:
        pass

    print_status("Installing OpenBabel via apt-get...", "info")
    try:
        subprocess.run(['apt-get', 'update', '-qq'], check=True, timeout=60)
        subprocess.run(['apt-get', 'install', '-y', 'openbabel', 'python3-openbabel'],
                       check=True, timeout=180)
        result = subprocess.run(['obabel', '-V'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0]
            print_status(f"OpenBabel installed: {version}", "success")
            return True
        else:
            print_status("OpenBabel installation verification failed", "error")
            return False
    except Exception as e:
        print_status(f"OpenBabel installation failed: {e}", "error")
        return False


def convert_to_pdb_openbabel(input_file, output_dir, clean_filename):
    try:
        pdb_file = os.path.join(output_dir, f"{clean_filename}.pdb")
        cmd = ['obabel', input_file, '-O', pdb_file, '--gen3d']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and os.path.exists(pdb_file) and os.path.getsize(pdb_file) > 50:
            return pdb_file, None
        else:
            return None, result.stderr.strip() or "OpenBabel conversion failed"
    except subprocess.TimeoutExpired:
        return None, "OpenBabel timeout (>60 s)"
    except Exception as e:
        return None, f"OpenBabel error: {str(e)}"


def fix_pdb_file(pdb_file):
    try:
        with open(pdb_file, 'r') as f:
            lines = f.readlines()
        fixed_lines = []
        atom_count = 0
        has_issues = False
        for line in lines:
            if line.startswith('HETATM'):
                line = 'ATOM  ' + line[6:]
                has_issues = True
                atom_count += 1
            elif line.startswith('ATOM'):
                atom_count += 1
            if line.startswith(('CONECT', 'MASTER', 'SSBOND', 'LINK', 'CISPEP')):
                has_issues = True
                continue
            if line.startswith('END'):
                if not any(l.startswith('TER') for l in fixed_lines):
                    fixed_lines.append(f"TER   {atom_count+1:5d}      UNL A   1\n")
                    has_issues = True
            fixed_lines.append(line)
        if has_issues:
            with open(pdb_file, 'w') as f:
                f.writelines(fixed_lines)
            return True
        return False
    except Exception:
        return False


def process_single_ligand(input_file, output_dir):
    try:
        original_name = Path(input_file).name
        clean_name = Path(input_file).stem
        import re
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', clean_name)
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')
        file_ext = Path(input_file).suffix.lower()
        print(f"\n🔄 Processing {original_name}")
        converted = False
        if file_ext == '.pdb':
            work_file = os.path.join(output_dir, f"{clean_name}.pdb")
            shutil.copy2(input_file, work_file)
            pdb_file = work_file
            fixed = fix_pdb_file(pdb_file)
            if fixed:
                print("   🔧 Fixed PDB format")
            print("   ✅ PDB format verified")
        elif file_ext in ['.sdf', '.mol', '.mol2', '.xyz', '.cml', '.smi']:
            pdb_file, error = convert_to_pdb_openbabel(input_file, output_dir, clean_name)
            if pdb_file is None:
                progress.update(success=False)
                return None, f"OpenBabel conversion failed: {error}"
            converted = True
            print(f"   ✅ Converted {file_ext.upper()} → PDB (OpenBabel)")
        else:
            progress.update(success=False)
            return None, f"Unsupported format: {file_ext}"
        if not os.path.exists(pdb_file) or os.path.getsize(pdb_file) < 50:
            progress.update(success=False)
            return None, "Invalid PDB file generated"
        output_path = os.path.join(output_dir, f"{clean_name}.pdbqt")
        abs_pdb = os.path.abspath(pdb_file)
        abs_output = os.path.abspath(output_path)
        cmd = [
            MGLTOOLS_PATH, PREPARE_LIGAND,
            "-l", abs_pdb,
            "-o", abs_output,
            "-A", "hydrogens",
            "-U", "nphs"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                              env=os.environ, timeout=300, cwd=output_dir)
        if result.returncode == 0 and os.path.exists(abs_output):
            size = os.path.getsize(abs_output)
            print(f"   ✅ Generated PDBQT ({size} bytes)")
            progress.update(success=True, converted=converted)
            return abs_output, None
        else:
            msg = result.stderr.strip() or "MGLTools failed"
            print(f"   ❌ MGLTools: {msg[:100]}")
            progress.update(success=False)
            return None, msg
    except subprocess.TimeoutExpired:
        progress.update(success=False)
        return None, "Timeout (>5 min)"
    except Exception as e:
        progress.update(success=False)
        return None, str(e)


def batch_process_ligands(uploaded_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    progress.total = len(uploaded_files)
    print(f"🚀 Processing {len(uploaded_files)} ligand(s)...")
    successful, failed = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {
            executor.submit(process_single_ligand, filename, output_dir): filename
            for filename in uploaded_files.keys()
        }
        for future in concurrent.futures.as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                output_path, error = future.result()
                if output_path:
                    successful.append(output_path)
                else:
                    failed.append((filename, error))
            except Exception as e:
                failed.append((filename, str(e)))
    return successful, failed

def show_summary(successful, failed, file_type):
    total_time = time.time() - progress.start_time
    print(f"\n{'='*60}")
    print(f"📋 {file_type} Processing Summary")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    if hasattr(progress, 'converted') and progress.converted > 0:
        print(f"🔄 Format conversions: {progress.converted}")
    print(f"⏱️ Total time: {total_time:.1f}s")
    if failed:
        print("\n⚠️ Failed files:")
        for i, (filename, error) in enumerate(failed[:5], 1):
            print(f"   {i}. {filename}: {error}")
        if len(failed) > 5:
            print(f"   ... and {len(failed) - 5} more")


def create_package(file_type, output_dir):
    try:
        archive_name = f"zymevo_{file_type}"
        shutil.make_archive(archive_name, 'zip', output_dir)
        zip_file = f"{archive_name}.zip"
        if os.path.exists(zip_file):
            return zip_file
    except Exception as e:
        print(f"❌ Packaging failed: {e}")
    return None


def main():
    display(HTML("""
    <div style="text-align:center; padding:15px; background-color:#f0f7ff; border-radius:8px;">
        <h2 style="color:#1a5fb4;">🧬 ZymEvo Unified Preprocessing</h2>
        <p>Batch processing for receptors and ligands</p>
    </div>
    """))
    print("\n📋 Select processing type:")
    print("1️⃣  Receptor preprocessing (PDB → PDBQT)")
    print("2️⃣  Ligand preprocessing (SDF/MOL/MOL2/PDB → PDBQT)")
    print("3️⃣  Both (receptors + ligands)")
    choice = input("\nEnter choice (1/2/3): ").strip()
    openbabel_ready = False
    if choice in ['2', '3']:
        openbabel_ready = setup_openbabel()
        if not openbabel_ready:
            print_status("⚠️ OpenBabel not available. Only PDB ligands can be processed.", "warning")
    all_successful, all_failed = [], []
    if choice in ['1', '3']:
        print("\n🧪 Upload receptor files (.pdb)...")
        receptor_files = files.upload()
        if receptor_files:
            progress.reset()
            successful, failed = batch_process_receptors(receptor_files, 'processed_receptors')
            show_summary(successful, failed, "Receptor")
            all_successful.extend(successful)
            all_failed.extend(failed)
            if successful:
                pkg = create_package("receptors", "processed_receptors")
                if pkg:
                    files.download(pkg)
                    print_status(f"✓ Downloaded: {pkg}", "success")
    if choice in ['2', '3']:
        print("\n💊 Upload ligand files...")
        ligand_files = files.upload()
        if ligand_files:
            progress.reset()
            successful, failed = batch_process_ligands(ligand_files, 'processed_ligands')
            show_summary(successful, failed, "Ligand")
            all_successful.extend(successful)
            all_failed.extend(failed)
            if successful:
                pkg = create_package("ligands", "processed_ligands")
                if pkg:
                    files.download(pkg)
                    print_status(f"✓ Downloaded: {pkg}", "success")
    if all_successful:
        display(HTML(f"""
        <div style="text-align:center; padding:15px; background-color:#e8f5e9;
                    border-radius:8px; border:1px solid #4CAF50; margin-top:20px;">
            <h3 style="color:#2E7D32;">🎉 Processing Complete!</h3>
            <p><strong>{len(all_successful)}</strong> files processed successfully</p>
        </div>
        """))
    else:
        display(HTML("""
        <div style="text-align:center; padding:15px; background-color:#ffebee;
                    border-radius:8px; border:1px solid #F44336; margin-top:20px;">
            <h3 style="color:#C62828;">⚠️ No files processed successfully</h3>
        </div>
        """))
        
progress = ProcessingProgress()

if __name__ == "__main__":
    main()
