#!/usr/bin/env python3
"""
ZymEvo Environment Setup Script
Tianjin University of Science and Technology</h4>
Research Center for Green BioManufacturing</h5>
Automated installation of enzyme evolution tools for Google Colab (Ubuntu Linux)
"""

import os
import sys
import subprocess
from IPython.display import HTML, display

def print_status(message, status="info"):
    colors = {
        "success": "#4CAF50",
        "info": "#2196F3",
        "warning": "#FF9800",
        "error": "#F44336",
    }
    icons = {
        "success": "✓",
        "info": "🔄",
        "warning": "⚠️",
        "error": "✗"
    }
    
    color = colors.get(status, colors["info"])
    icon = icons.get(status, "🔄")
    
    display(HTML(f"""
    <div style='padding:8px; margin:5px 0; border-radius:4px; 
                background-color:{color}20; border-left:5px solid {color};'>
        <span style='color:{color}; font-weight:bold;'>{icon} </span>{message}
    </div>
    """))

def run_cmd(command, description=None):
    if description:
        print_status(f"{description}...", "info")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        if description:
            print_status(f"{description} completed", "success")
        return True
    else:
        if description:
            print_status(f"{description} failed: {result.stderr[:100]}", "error")
        return False

def main():
    display(HTML("""
    <div style="text-align:center; padding:15px; background-color:#f0f7ff; border-radius:8px;">
        <h2 style="color:#1a5fb4;">🧬 ZymEvo Environment Setup</h2>
        <p>Automated installation of enzyme engineering tools</p>
        <p style="font-size:0.9em; color:#666;">
            GitHub: <a href="https://github.com/BCBT-tust/ZymEvo-RCGB" target="_blank">BCBT-tust/ZymEvo-RCGB</a>
        </p>
    </div>
    """))
    
    print_status("Step 1/6: Installing Miniconda")
    run_cmd("wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh")
    run_cmd("bash Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local")
    os.environ['PATH'] = '/usr/local/bin:' + os.environ['PATH']
    
    print_status("Step 2/6: Installing Python 2.7 and pip2")
    run_cmd("apt-get update -qq && apt-get install -y python2.7 csh", 
            "Installing Python 2.7 and csh")
    run_cmd("curl -s https://bootstrap.pypa.io/pip/2.7/get-pip.py | python2.7", 
            "Installing pip2")
    run_cmd("pip2 install numpy", "Installing numpy for Python 2.7")

    print_status("Step 3/6: Installing OpenBabel")
    run_cmd("apt-get install -y openbabel python3-openbabel", 
            "Installing OpenBabel via apt-get")
    
    print_status("Step 4/6: Installing AutoDockTools")
    if not os.path.exists('/usr/local/autodocktools'):
        run_cmd("wget -q https://ccsb.scripps.edu/mgltools/download/491/tars/releases/REL1.5.7/mgltools_x86_64Linux2_1.5.7.tar.gz")
        run_cmd("mkdir -p /usr/local/autodocktools")
        run_cmd("tar -xzf mgltools_x86_64Linux2_1.5.7.tar.gz -C /usr/local/autodocktools --strip-components=1")
        run_cmd("tar -xzf /usr/local/autodocktools/MGLToolsPckgs.tar.gz -C /usr/local/autodocktools/")
    else:
        print_status("AutoDockTools already installed", "success")

    print_status("Step 5/6: Configuring pythonsh")
    pythonsh_path = "/usr/local/autodocktools/bin/pythonsh"
    with open(pythonsh_path, "w") as f:
        f.write("#!/bin/bash\n/usr/bin/python2.7 \"$@\"\n")
    os.chmod(pythonsh_path, 0o755)

    print_status("Step 6/6: Setting environment variables")
    os.environ['PYTHONPATH'] = "/usr/local/autodocktools/MGLToolsPckgs"

    print_status("Verifying installation...")
    checks = {
        "Python 2.7": "python2.7 --version",
        "pip2": "pip2 --version",
        "OpenBabel": "obabel -V",
        "pythonsh": f"test -x {pythonsh_path} && echo 'OK'"
    }
    
    results = []
    for name, cmd in checks.items():
        result = subprocess.run(cmd, shell=True, capture_output=True)
        status = "✓" if result.returncode == 0 else "✗"
        color = "#4CAF50" if result.returncode == 0 else "#F44336"
        results.append((name, status, color))
    
    html = """
    <div style="padding:10px; margin:10px 0; background-color:#f9f9f9; border-radius:4px;">
    <h3>Installation Verification</h3>
    <table style="width:100%; border-collapse:collapse;">
    <tr style="background-color:#e0e0e0;">
        <th style="padding:8px; text-align:left; border:1px solid #ddd;">Component</th>
        <th style="padding:8px; text-align:center; border:1px solid #ddd;">Status</th>
    </tr>
    """
    for name, status, color in results:
        html += f"""
    <tr>
        <td style="padding:8px; border:1px solid #ddd;">{name}</td>
        <td style="padding:8px; text-align:center; color:{color}; font-weight:bold; border:1px solid #ddd;">{status}</td>
    </tr>
    """
    html += "</table></div>"
    display(HTML(html))
    
    run_cmd("rm -f Miniconda3-latest-Linux-x86_64.sh mgltools_x86_64Linux2_1.5.7.tar.gz", 
            "Cleaning up")
    
    if all(status == "✓" for _, status, _ in results):
        display(HTML("""
        <div style="text-align:center; padding:15px; background-color:#e8f5e9; 
                    border-radius:8px; border:1px solid #4CAF50;">
            <h2 style="color:#2E7D32;">🎉 Setup Complete!</h2>
            <p>All components installed successfully. ZymEvo environment is ready.</p>
        </div>
        """))
    else:
        display(HTML("""
        <div style="text-align:center; padding:15px; background-color:#ffebee; 
                    border-radius:8px; border:1px solid #F44336;">
            <h2 style="color:#C62828;">⚠️ Some components failed</h2>
            <p>Please check error messages above and try again</p>
        </div>
        """))
    
    print("\n" + "="*60)
    print("✓ ZymEvo Environment Variables:")
    print("="*60)
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
    print(f"pythonsh: {pythonsh_path}")
    print(f"prepare_receptor4: /usr/local/autodocktools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py")
    print(f"prepare_ligand4: /usr/local/autodocktools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py")
    print("="*60)
    
    return all(status == "✓" for _, status, _ in results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
