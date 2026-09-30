#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
from IPython.display import HTML, display

MGLTOOLS_URL = (
    "https://ccsb.scripps.edu/mgltools/download/491/tars/releases/"
    "REL1.5.7/mgltools_x86_64Linux2_1.5.7.tar.gz"
)
MGLTOOLS_TARBALL = "mgltools_x86_64Linux2_1.5.7.tar.gz"
MGLTOOLS_DIR = "/usr/local/autodocktools"
PREPARE_RECEPTOR = f"{MGLTOOLS_DIR}/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"
PREPARE_LIGAND = f"{MGLTOOLS_DIR}/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py"
PYTHONSH_PATH = f"{MGLTOOLS_DIR}/bin/pythonsh"

# Miniconda is installed into /usr/local; Python 2.7 lives in its own env.
CONDA_BIN = "/usr/local/bin/conda"
PY27_ENV = "/usr/local/envs/py27"
PY27_BIN = f"{PY27_ENV}/bin/python2.7"
PIP2_BIN = f"{PY27_ENV}/bin/pip"

# Scripps server has an SSL SAN mismatch; --no-check-certificate is required.
WGET_OPTS = "--no-check-certificate -q"


def print_status(message, status="info"):
    colors = {
        "success": "#4CAF50",
        "info": "#2196F3",
        "warning": "#FF9800",
        "error": "#F44336",
    }
    icons = {"success": "✓", "info": "🔄", "warning": "⚠️", "error": "✗"}
    color = colors.get(status, colors["info"])
    icon = icons.get(status, "🔄")
    display(HTML(f"""
    <div style='padding:8px; margin:5px 0; border-radius:4px;
                background-color:{color}20; border-left:5px solid {color};'>
        <span style='color:{color}; font-weight:bold;'>{icon} </span>{message}
    </div>
    """))


def run_cmd(command, description=None, check=True):
    """Run a shell command. If check=True, raise on non-zero exit."""
    if description:
        print_status(f"{description}...", "info")

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        if description:
            print_status(f"{description} completed", "success")
        return result

    # Non-zero
    err_tail = (result.stderr or result.stdout or "").strip()[-300:]
    if description:
        print_status(f"{description} failed: {err_tail}", "error")
    if check:
        raise RuntimeError(f"Command failed: {command}\n{err_tail}")
    return result


def install_python27():
    """Create a conda env with Python 2.7 + numpy.

    Ubuntu 24.04 (current Colab) removed python2.7 from apt, so conda is used.
    conda-forge is tried first (no Terms-of-Service prompt); if that fails,
    fall back to Anaconda's default channel after accepting its ToS.
    """
    if os.path.exists(PY27_BIN):
        print_status("Python 2.7 conda env already exists", "success")
    else:
        r = run_cmd(
            f"{CONDA_BIN} create -y -q -p {PY27_ENV} "
            f"--override-channels -c conda-forge python=2.7 numpy=1.16 pip",
            "Creating Python 2.7 env (conda-forge)",
            check=False,
        )
        if r.returncode != 0 or not os.path.exists(PY27_BIN):
            print_status("conda-forge failed, trying Anaconda defaults channel", "warning")
            run_cmd(
                f"{CONDA_BIN} tos accept --override-channels "
                f"--channel https://repo.anaconda.com/pkgs/main "
                f"--channel https://repo.anaconda.com/pkgs/r",
                check=False,
            )
            run_cmd(
                f"{CONDA_BIN} create -y -q -p {PY27_ENV} "
                f"--override-channels -c defaults python=2.7 numpy=1.16 pip",
                "Creating Python 2.7 env (defaults)",
            )

    if not os.path.exists(PY27_BIN):
        raise RuntimeError(f"Python 2.7 not found at {PY27_BIN}")

    # Compatibility links so later notebook cells that call `python2.7` / `pip2`
    # directly keep working.
    run_cmd(f"ln -sf {PY27_BIN} /usr/local/bin/python2.7", check=True)
    run_cmd(f"ln -sf {PIP2_BIN} /usr/local/bin/pip2", check=True)


def install_mgltools():
    """Download + extract MGLTools. Idempotent: skips if already installed,
    otherwise cleans partial state before re-downloading."""
    if os.path.exists(PREPARE_RECEPTOR):
        print_status("AutoDockTools already installed (verified by file check)", "success")
        return

    # Clean any partial residue (empty dir, stale tarball) before retry.
    if os.path.exists(MGLTOOLS_DIR):
        print_status("Removing incomplete AutoDockTools directory...", "warning")
        run_cmd(f"rm -rf {MGLTOOLS_DIR}", check=True)
    if os.path.exists(MGLTOOLS_TARBALL):
        run_cmd(f"rm -f {MGLTOOLS_TARBALL}", check=True)

    # Download (SSL cert on Scripps has SAN mismatch; must bypass).
    run_cmd(
        f"wget {WGET_OPTS} {MGLTOOLS_URL} -O {MGLTOOLS_TARBALL}",
        "Downloading MGLTools 1.5.7 (~60MB)",
    )

    # Sanity-check file size. If Scripps returned an error page, it'll be tiny.
    size = os.path.getsize(MGLTOOLS_TARBALL) if os.path.exists(MGLTOOLS_TARBALL) else 0
    if size < 50_000_000:
        raise RuntimeError(
            f"MGLTools download appears incomplete ({size} bytes). "
            f"Check connectivity to ccsb.scripps.edu."
        )

    # Extract (two-layer: outer tar + inner MGLToolsPckgs.tar.gz).
    run_cmd(f"mkdir -p {MGLTOOLS_DIR}", check=True)
    run_cmd(
        f"tar -xzf {MGLTOOLS_TARBALL} -C {MGLTOOLS_DIR} --strip-components=1",
        "Extracting MGLTools (layer 1)",
    )
    run_cmd(
        f"tar -xzf {MGLTOOLS_DIR}/MGLToolsPckgs.tar.gz -C {MGLTOOLS_DIR}/",
        "Extracting MGLTools (layer 2)",
    )

    if not os.path.exists(PREPARE_RECEPTOR):
        raise RuntimeError(
            f"Expected script not found after extraction: {PREPARE_RECEPTOR}"
        )


def configure_pythonsh():
    """Create the pythonsh wrapper that routes AutoDockTools scripts to Python 2.7.

    PYTHONPATH is set inside the wrapper so it works even if the notebook's
    environment variables are lost (e.g. when called via subprocess).
    """
    os.makedirs(os.path.dirname(PYTHONSH_PATH), exist_ok=True)
    with open(PYTHONSH_PATH, "w") as f:
        f.write(
            "#!/bin/bash\n"
            f"export PYTHONPATH=\"{MGLTOOLS_DIR}/MGLToolsPckgs\"\n"
            "unset PYTHONHOME\n"
            f"exec \"{PY27_BIN}\" \"$@\"\n"
        )
    os.chmod(PYTHONSH_PATH, 0o755)


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

    try:
        print_status("Step 1/6: Installing Miniconda")
        if not os.path.exists(CONDA_BIN):
            run_cmd(
                f"wget {WGET_OPTS} "
                "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh",
                "Downloading Miniconda",
            )
            run_cmd(
                "bash Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local",
                "Installing Miniconda",
            )
        else:
            print_status("Miniconda already installed", "success")
        os.environ["PATH"] = "/usr/local/bin:" + os.environ["PATH"]

        print_status("Step 2/6: Installing Python 2.7 (conda env) and csh")
        run_cmd(
            "apt-get update -qq; apt-get install -y -qq csh",
            "Installing csh",
        )
        install_python27()

        print_status("Step 3/6: Installing OpenBabel")
        run_cmd(
            "apt-get install -y -qq openbabel python3-openbabel",
            "Installing OpenBabel via apt-get",
        )

        print_status("Step 4/6: Installing AutoDockTools")
        install_mgltools()

        print_status("Step 5/6: Configuring pythonsh")
        configure_pythonsh()

        print_status("Step 6/6: Setting environment variables")
        os.environ["PYTHONPATH"] = f"{MGLTOOLS_DIR}/MGLToolsPckgs"

    except Exception as e:
        print_status(f"Setup aborted: {e}", "error")
        return False

    print_status("Verifying installation...")
    checks = {
        "Python 2.7":           "python2.7 --version",
        "pip2":                 "pip2 --version",
        "OpenBabel":            "obabel -V",
        "pythonsh":             f"test -x {PYTHONSH_PATH} && echo 'OK'",
        "prepare_receptor4.py": f"test -f {PREPARE_RECEPTOR} && echo 'OK'",
        "prepare_ligand4.py":   f"test -f {PREPARE_LIGAND} && echo 'OK'",
        # Functional test: the scripts' imports actually load under Python 2.7.
        "AutoDockTools import": (
            f"{PYTHONSH_PATH} -c "
            "\"import numpy, MolKit, AutoDockTools.MoleculePreparation; print('OK')\""
        ),
    }

    results = []
    for name, cmd in checks.items():
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        ok = result.returncode == 0
        if not ok:
            print_status(f"{name} check failed: {(result.stderr or '').strip()[-300:]}", "error")
        results.append((name, "✓" if ok else "✗", "#4CAF50" if ok else "#F44336"))

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

    # Always clean up tarballs, regardless of success.
    run_cmd(
        f"rm -f Miniconda3-latest-Linux-x86_64.sh {MGLTOOLS_TARBALL}",
        "Cleaning up downloads",
        check=False,
    )

    all_ok = all(status == "✓" for _, status, _ in results)

    if all_ok:
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
            <p>Please check error messages above and re-run.</p>
        </div>
        """))

    print("\n" + "=" * 60)
    print("✓ ZymEvo Environment Variables:")
    print("=" * 60)
    print(f"PYTHONPATH:        {os.environ.get('PYTHONPATH')}")
    print(f"python2.7:         {PY27_BIN}")
    print(f"pythonsh:          {PYTHONSH_PATH}")
    print(f"prepare_receptor4: {PREPARE_RECEPTOR}")
    print(f"prepare_ligand4:   {PREPARE_LIGAND}")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
